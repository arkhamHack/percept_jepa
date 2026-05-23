"""
Tracking Head + Track Memory Bank
===================================
Produces per-object identity embeddings for data association across frames.

Tracking paradigm: Embedding-based Re-Identification
------------------------------------------------------
Instead of running a separate detection-and-association pipeline, we train
the model to produce discriminative embeddings for each detected object.
At inference, embeddings from frame t are matched to embeddings from frame
t-1 via cosine similarity.  This is the approach used in:
  - FairMOT (Zhang et al. 2021)
  - CenterTrack (Zhou et al. 2020)
  - TrackFormer (Meinhardt et al. 2022)

Key design choices:
  1. Track embeddings are derived from object query embeddings (output of
     detection head decoder).  This shares representation and avoids
     double-counting scene features.
  2. A contrastive training loss (InfoNCE or triplet) ensures that
     embeddings of the same instance across frames are similar, while
     embeddings of different instances are dissimilar.
  3. The TrackMemoryBank stores embeddings from previous frames and
     provides IoU + cosine similarity for multi-frame association.

Track Association Algorithm (simplified Hungarian):
  1. Compute cosine similarity matrix [Q_curr, M_active_tracks]
  2. Apply IoU constraint from detection boxes (optional gating)
  3. Hungarian matching on similarity matrix
  4. Update active tracks; create new tracks for unmatched detections
  5. Kill tracks older than max_age frames

TODO:
  - Transformer-based temporal query propagation (TrackFormer-style)
  - Kalman filter state for position prediction between frames
  - ReID feature bank per track (weighted average of N past embeddings)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track Memory Bank
# ---------------------------------------------------------------------------

class TrackMemoryBank:
    """
    Runtime store of active track states for inference-time association.

    Each track stores:
      - track_id:       unique integer
      - embedding:      Tensor[E]   averaged identity embedding
      - last_box:       Tensor[7]   (cx, cy, cz, dx, dy, dz, yaw) last seen
      - age:            int         frames since last matched
      - n_matched:      int         total number of matched frames
    """

    def __init__(
        self,
        embed_dim: int,
        similarity_threshold: float = 0.6,
        max_age: int = 5,
        device: str = 'cpu',
    ):
        self.embed_dim = embed_dim
        self.sim_thresh = similarity_threshold
        self.max_age = max_age
        self.device = device

        self._tracks: List[Dict] = []
        self._next_id = 0

    def reset(self):
        """Clear all tracks (call at scene boundaries)."""
        self._tracks = []
        self._next_id = 0

    @property
    def num_active(self) -> int:
        return len(self._tracks)

    def get_embeddings(self) -> Optional[torch.Tensor]:
        """Return stacked embeddings of active tracks. [M, E]"""
        if not self._tracks:
            return None
        return torch.stack([t['embedding'] for t in self._tracks]).to(self.device)

    def get_boxes(self) -> Optional[torch.Tensor]:
        """Return last-seen boxes of active tracks. [M, 7]"""
        if not self._tracks:
            return None
        return torch.stack([t['last_box'] for t in self._tracks]).to(self.device)

    def get_ids(self) -> List[int]:
        """Return list of active track IDs."""
        return [t['track_id'] for t in self._tracks]

    def update(
        self,
        det_embeddings: torch.Tensor,   # [Q, E]  current frame detections
        det_boxes: torch.Tensor,        # [Q, 7]  current frame boxes
        det_scores: torch.Tensor,       # [Q]     objectness scores
        score_threshold: float = 0.3,
    ) -> torch.Tensor:
        """
        Match current detections to active tracks and update the memory bank.

        Returns:
            track_ids: Tensor[Q] — track ID per detection (-1 = new track)
        """
        Q = det_embeddings.shape[0]
        assigned_ids = torch.full((Q,), -1, dtype=torch.long)

        # Filter low-confidence detections
        valid_mask = det_scores > score_threshold  # [Q] bool

        if not self._tracks or not valid_mask.any():
            # No active tracks: create a new track for every valid detection
            for q in range(Q):
                if valid_mask[q]:
                    tid = self._create_track(det_embeddings[q], det_boxes[q])
                    assigned_ids[q] = tid
            return assigned_ids

        # ---- Cosine similarity matrix --------------------------------
        track_embeds = self.get_embeddings()          # [M, E]
        det_norm   = F.normalize(det_embeddings, dim=-1)  # [Q, E]
        track_norm = F.normalize(track_embeds, dim=-1)    # [M, E]
        sim_matrix = det_norm @ track_norm.T              # [Q, M]

        # ---- Hungarian matching (greedy approximation) ---------------
        matched_dets  = set()
        matched_tracks = set()

        # Sort by similarity descending
        flat_idx = sim_matrix.reshape(-1).argsort(descending=True)
        for idx in flat_idx:
            q = idx // sim_matrix.shape[1]
            m = idx %  sim_matrix.shape[1]
            q, m = q.item(), m.item()

            if q in matched_dets or m in matched_tracks:
                continue
            if not valid_mask[q]:
                continue
            if sim_matrix[q, m] < self.sim_thresh:
                break

            # Match det q → track m
            tid = self._tracks[m]['track_id']
            assigned_ids[q] = tid
            matched_dets.add(q)
            matched_tracks.add(m)

            # Update track embedding (EMA)
            alpha = 0.7
            self._tracks[m]['embedding'] = (
                alpha * self._tracks[m]['embedding']
                + (1 - alpha) * det_embeddings[q].detach()
            )
            self._tracks[m]['last_box'] = det_boxes[q].detach()
            self._tracks[m]['age'] = 0
            self._tracks[m]['n_matched'] += 1

        # ---- Age unmatched tracks -----------------------------------
        for m in range(len(self._tracks)):
            if m not in matched_tracks:
                self._tracks[m]['age'] += 1

        # ---- Create new tracks for unmatched detections -------------
        for q in range(Q):
            if q not in matched_dets and valid_mask[q]:
                tid = self._create_track(det_embeddings[q], det_boxes[q])
                assigned_ids[q] = tid

        # ---- Kill old tracks ----------------------------------------
        self._tracks = [t for t in self._tracks if t['age'] <= self.max_age]

        return assigned_ids

    def _create_track(
        self, embedding: torch.Tensor, box: torch.Tensor
    ) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks.append({
            'track_id':  tid,
            'embedding': embedding.detach().clone(),
            'last_box':  box.detach().clone(),
            'age':       0,
            'n_matched': 1,
        })
        return tid


# ---------------------------------------------------------------------------
# Tracking Head (neural module)
# ---------------------------------------------------------------------------

class TrackingHead(nn.Module):
    """
    Produces per-detection identity embeddings from object query vectors.

    The embedding space is trained with a contrastive loss so that:
      - same instance in different frames → high cosine similarity
      - different instances → low cosine similarity

    Args:
        cfg: full config
    """

    def __init__(self, cfg):
        super().__init__()
        tcfg = cfg.model.tracking
        D    = cfg.model.hidden_dim
        E    = tcfg.embed_dim
        L    = tcfg.proj_layers

        # Projection MLP: D → E
        layers = []
        in_d = D
        for i in range(L):
            out_d = E if i == L - 1 else D
            layers.extend([nn.Linear(in_d, out_d), nn.GELU()])
            in_d = out_d
        # Replace last GELU with LayerNorm for stable embedding space
        layers[-1] = nn.LayerNorm(E)
        self.projection = nn.Sequential(*layers)

        self.embed_dim = E

    def forward(
        self,
        query_embeds: torch.Tensor,    # [B, Q, D]  from DetectionHead
    ) -> torch.Tensor:
        """
        Project object query embeddings into identity embedding space.

        Args:
            query_embeds: Tensor[B, Q, D]  decoded object query vectors

        Returns:
            Tensor[B, Q, E]  L2-normalised identity embeddings
              E = tracking embed_dim (e.g. 256)

        These embeddings are:
          - L2-normalised → cosine similarity = dot product
          - Trained with InfoNCE contrastive loss using GT track IDs as
            positives (same track_id in consecutive frames = positive pair)
        """
        embeds = self.projection(query_embeds)    # [B, Q, E]
        embeds = F.normalize(embeds, dim=-1)       # [B, Q, E]  unit vectors
        return embeds
