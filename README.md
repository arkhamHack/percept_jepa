# Rear Camera + Rear Radar Fusion via V-JEPA Latent Representations

> **Research PoC** — Latent-space multimodal fusion for detection, tracking,
> and trajectory prediction on nuScenes, emphasising **world-model-inspired**
> transformer-centric representations.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        INPUT SENSORS                                  │
│                                                                      │
│   CAM_BACK frames [B, T, 3, 224, 224]   RADAR_BACK_L/R [B, N, 6]  │
└──────────────┬──────────────────────────────────┬────────────────────┘
               │                                  │
               ▼                                  ▼
   ┌───────────────────────┐        ┌─────────────────────────┐
   │    V-JEPA Encoder     │        │     Radar Encoder        │
   │                       │        │                         │
   │  Spatial ViT (frozen) │        │  PointNet MLP backbone  │
   │       +               │        │  (per-point features)   │
   │  Temporal Transformer │        │       +                 │
   │  (factored attention) │        │  Soft Token Aggregation │
   │                       │        │  (K=64 learned tokens)  │
   └──────────┬────────────┘        └───────────┬─────────────┘
              │ [B, T*P, D]                      │ [B, K, D]
              │ T=4, P=196, D=768                │ K=64, D=768
              │                                  │
              │          ┌───────────────────────┤
              │          │   Ego Motion Encoder  │
              │          │   [B, 6] → [B, 1, D]  │
              │          └───────────┬───────────┘
              │                      │ [B, 1, D]
              ▼                      ▼
   ┌──────────────────────────────────────────────────────────┐
   │                  Fusion Transformer                       │
   │                                                          │
   │  1. Modality-type embeddings (visual / radar / ego)      │
   │  2. Cross-modal attention: radar↔visual (optional)       │
   │  3. Concatenate: [visual_tokens | radar_tokens | ego]    │
   │     Total: T*P + K + 1 = 784 + 64 + 1 = 849 tokens      │
   │  4. L=6 self-attention layers (pre-norm)                  │
   │                                                          │
   └──────────────────────┬───────────────────────────────────┘
                          │ [B, 849, D]  ← Shared Scene Latent
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
  ┌──────────────┐ ┌────────────┐ ┌───────────────────┐
  │  Detection   │ │  Tracking  │ │    Prediction      │
  │    Head      │ │    Head    │ │      Head          │
  │              │ │            │ │                    │
  │ DETR queries │ │ Identity   │ │ Future trajectory  │
  │ Q=100        │ │ projection │ │ MLP / Transformer  │
  │ Decoder L=3  │ │ D → E=256  │ │ [B, Q, K, 2]       │
  │              │ │ L2-normed  │ │ K=12 steps × 0.5s  │
  └──────┬───────┘ └─────┬──────┘ └────────┬───────────┘
         │               │                 │
         ▼               ▼                 ▼
  class_logits    track_embeds        pred_traj
  [B, Q, C+1]     [B, Q, 256]        [B, Q, K, 2]
  pred_boxes
  [B, Q, 8]
```

---

## Project Structure

```
radar_vjepa_fusion/
│
├── configs/
│   └── default.yaml              ← All hyperparameters
│
├── data/
│   ├── nuscenes_dataset.py       ← nuScenes temporal dataset loader
│   ├── radar_utils.py            ← Radar processing utilities
│   ├── transforms.py             ← Image + radar augmentations
│   └── collate.py                ← Custom batch collation
│
├── models/
│   ├── vjepa_encoder.py          ← V-JEPA temporal visual encoder
│   ├── radar_encoder.py          ← PointNet + soft aggregation
│   ├── fusion_transformer.py     ← Multimodal fusion transformer
│   ├── shared_scene_latent.py    ← Scene latent container
│   ├── detection_head.py         ← DETR-style object detector
│   ├── tracking_head.py          ← Identity embedding head
│   ├── prediction_head.py        ← Future trajectory decoder
│   └── full_model.py             ← End-to-end model assembly
│
├── trainers/
│   ├── train_multitask.py        ← Joint detection+tracking+prediction
│   ├── train_detection.py        ← Detection-only training
│   └── train_prediction.py       ← Prediction fine-tuning
│
├── losses/
│   ├── detection_loss.py         ← Hungarian + focal + GIoU
│   ├── tracking_loss.py          ← InfoNCE contrastive
│   └── prediction_loss.py        ← Smooth L1 trajectory
│
├── visualization/
│   ├── visualize_radar.py        ← BEV radar point plots
│   ├── visualize_attention.py    ← Query attention heatmaps
│   ├── visualize_tracking.py     ← Tracked box rendering
│   └── visualize_predictions.py  ← Trajectory plots
│
├── utils/
│   ├── geometry.py               ← Box/pose transforms
│   ├── positional_encoding.py    ← Sin, learnable, Fourier 3D
│   ├── metrics.py                ← mAP, ADE/FDE, MOTP
│   └── checkpointing.py          ← Save/load/best tracking
│
├── scripts/
│   ├── download_vjepa.sh         ← Download V-JEPA weights
│   ├── train.sh                  ← Training launcher
│   └── eval.sh                   ← Evaluation launcher
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd radar_vjepa_fusion

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Download nuScenes

```bash
# Register at https://www.nuscenes.org/download
# Download "nuScenes mini" (350 MB) for initial testing

# Expected structure:
# /path/to/nuscenes/
#   v1.0-mini/
#   samples/
#   sweeps/
#   maps/
```

Update `configs/default.yaml`:
```yaml
data:
  nuscenes_root: "/path/to/nuscenes"
  nuscenes_version: "v1.0-mini"
```

### 3. Download V-JEPA weights (optional)

```bash
chmod +x scripts/download_vjepa.sh
./scripts/download_vjepa.sh
```

> **Without V-JEPA weights:** The model automatically falls back to
> a timm ImageNet-pretrained ViT-Base, which still provides strong
> spatial features. Set `VJEPA_WEIGHTS=/path/to/weights.pt` to use
> official V-JEPA weights.

### 4. Select Visual Encoder Backend

This project now supports two visual encoder backends:

1. `timm_fallback` (default): ViT-based fallback path with optional V-JEPA-style checkpoint loading.
2. `official_vjepa2`: official V-JEPA2/2.1 frozen encoder path via HuggingFace `AutoModel`.

In `configs/default.yaml`:

```yaml
model:
  vjepa:
    backend: "timm_fallback"      # or "official_vjepa2"
    official_model_id: ""         # e.g. "facebook/vjepa2-vitb"
    official_local_path: ""       # local downloaded model dir
    frozen: true
```

Example using official backend:

```yaml
model:
  vjepa:
    backend: "official_vjepa2"
    official_model_id: "facebook/vjepa2-vitb"
    frozen: true
```

Notes:

- No pretraining is required for this PoC workflow.
- The official encoder can stay frozen while training fusion + task heads.
- If your official model already outputs temporal clip tokens, keep:
  `official_outputs_temporal_tokens: true` and `apply_temporal_on_official: false`.

---

## Training

### Multitask training (recommended)

```bash
./scripts/train.sh

# With custom config:
CONFIG=configs/default.yaml ./scripts/train.sh

# Resume from checkpoint:
./scripts/train.sh --resume checkpoints/latest.pt
```

### Detection-only (for quick verification)

```bash
python trainers/train_detection.py --config configs/default.yaml
```

### Monitor training

```bash
tensorboard --logdir runs/
```

---

## Evaluation

```bash
./scripts/eval.sh
# or
./scripts/eval.sh --checkpoint checkpoints/best.pt
```

---

## Quick Start: Smoke Test

Verify the forward pass works without nuScenes data:

```python
import torch, sys
sys.path.insert(0, '.')

# Minimal config dict
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/default.yaml')
cfg.data.nuscenes_root = '/tmp'   # won't be used

from models.full_model import RearFusionModel

model = RearFusionModel(cfg).eval()

B, T, C, H, W = 2, 4, 3, 224, 224
N = 256  # radar points

images     = torch.randn(B, T, C, H, W)
radar_pts  = torch.randn(B, N, 6)
radar_mask = torch.ones(B, N)
ego_motion = torch.randn(B, 6)

with torch.no_grad():
    out = model(images, radar_pts, radar_mask, ego_motion)

print("class_logits:", out['class_logits'].shape)   # [2, 100, 11]
print("pred_boxes:  ", out['pred_boxes'].shape)     # [2, 100, 8]
print("track_embeds:", out['track_embeds'].shape)   # [2, 100, 256]
print("pred_traj:   ", out['pred_traj'].shape)      # [2, 100, 12, 2]
```

---

## Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Visual backbone | V-JEPA (ViT-B/16) | World-model latents; temporally consistent |
| Temporal fusion | Factored temporal attention | O(T²) not O((T×P)²) |
| Radar encoding | PointNet + soft aggregation | Permutation-invariant; differentiable |
| Multimodal fusion | Transformer self-attention | Learns cross-modal alignment implicitly |
| Detection | DETR-style queries | No anchors; end-to-end set prediction |
| Tracking | Contrastive embeddings | Simple; scales to many classes |
| Prediction | MLP/Transformer decoder | Fast; competitive for 6s horizon |

---

## TODOs & Research Extensions

- [ ] **Masked latent prediction** — Add V-JEPA-style auxiliary loss to predict
  future scene latents (proper world model objective)
- [ ] **Temporal memory tokens** — Persistent memory across frames for long-horizon reasoning
- [ ] **Occupancy prediction** — Auxiliary BEV grid head from scene latent
- [ ] **BEV projection** — Lift-Splat-Shoot or BEVFormer-style explicit BEV generation
- [ ] **Deformable attention** — Replace full self-attention for O(N) complexity
- [ ] **Multi-camera fusion** — Add front/side cameras to scene latent
- [ ] **Uncertainty estimation** — GMM trajectory outputs (like Trajectron++)
- [ ] **Slot Attention aggregation** — Replace soft-attention token aggregation in radar encoder
- [ ] **Online calibration** — Learn camera↔radar alignment from data
- [ ] **Streaming inference** — Single-frame incremental update with scene latent cache

---

## Citation

If you use this codebase in research, please cite the foundational works:

```bibtex
@article{assran2023vjepa,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and others},
  journal={CVPR},
  year={2023}
}

@article{carion2020detr,
  title={End-to-End Object Detection with Transformers},
  author={Carion, Nicolas and others},
  journal={ECCV},
  year={2020}
}

@article{caesar2020nuscenes,
  title={nuScenes: A multimodal dataset for autonomous driving},
  author={Caesar, Holger and others},
  journal={CVPR},
  year={2020}
}
```

---

## License

MIT License — see LICENSE file.

---

*Built for research. Not production-ready.*
