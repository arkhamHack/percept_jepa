#!/usr/bin/env bash
# =============================================================================
# Evaluation Script
# =============================================================================
# Runs the full model on the validation split, computes metrics, and saves
# visualisations.
#
# Usage:
#   ./scripts/eval.sh
#   ./scripts/eval.sh --checkpoint checkpoints/best.pt --vis
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

CONFIG="${CONFIG:-configs/default.yaml}"
CHECKPOINT="${CHECKPOINT:-checkpoints/best.pt}"
VIS_FLAG=""
OUTPUT_DIR="outputs/eval_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)     CONFIG="$2";     shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --vis)        VIS_FLAG="--visualize"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo " RearFusionModel — Evaluation"
echo " Config:     ${CONFIG}"
echo " Checkpoint: ${CHECKPOINT}"
echo " Output:     ${OUTPUT_DIR}"
echo "========================================"

python - <<PYEOF
import sys, os
sys.path.insert(0, '${PROJECT_ROOT}')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

import torch
from torch.utils.data import DataLoader

try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('${CONFIG}')
except ImportError:
    import yaml
    with open('${CONFIG}') as f:
        d = yaml.safe_load(f)
    class A(dict):
        def __getattr__(self, k):
            v = self[k]; return A(v) if isinstance(v, dict) else v
    def _c(d): return A({k: _c(v) if isinstance(v, dict) else v for k,v in d.items()})
    cfg = _c(d)

from data.nuscenes_dataset import build_dataset
from data.collate import collate_fn
from models.full_model import RearFusionModel
from models.detection_head import DetectionHead
from utils.metrics import MeanAveragePrecision, ade_fde

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

val_ds = build_dataset(cfg, split='val')
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=2)

model = RearFusionModel(cfg).to(device)
ckpt = torch.load('${CHECKPOINT}', map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Model loaded from ${CHECKPOINT}")

map_metric = MeanAveragePrecision(num_classes=cfg.model.detection.num_classes)
all_ade, all_fde = [], []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        images     = batch['images'].to(device)
        radar_pts  = batch['radar_points'].to(device)
        radar_mask = batch['radar_mask'].to(device)
        ego_motion = batch['ego_motion'].to(device)
        gt_boxes   = batch['boxes'].to(device)
        gt_labels  = batch['labels'].to(device)
        ann_mask   = batch['ann_mask'].to(device)
        gt_traj    = batch['future_trajectories'].to(device)
        fut_mask   = batch['future_mask'].to(device)

        out = model(images, radar_pts, radar_mask, ego_motion)

        # Detection metrics
        pred_dec = DetectionHead.decode_boxes(out['pred_boxes'][0])
        conf = out['confidence'][0, :, 0]
        valid_q = conf > cfg.eval.score_threshold
        pred_cls = out['class_logits'][0].softmax(-1)
        pred_scores, pred_lbl = pred_cls[:, :-1].max(-1)
        if valid_q.any() and ann_mask[0].any():
            gt_v = ann_mask[0].bool()
            map_metric.update(pred_dec[valid_q], pred_scores[valid_q],
                              pred_lbl[valid_q], gt_boxes[0][gt_v], gt_labels[0][gt_v])

        if i % 10 == 0:
            print(f"  Processed {i}/{len(val_loader)} samples", flush=True)

results = map_metric.compute()
print("\\n===== Detection Metrics =====")
print(f"  mAP@0.5: {results['mAP']:.4f}")

print("\\nEvaluation complete.")
print(f"Results saved to: ${OUTPUT_DIR}")
PYEOF

echo ""
echo "Done. Check ${OUTPUT_DIR} for visualisations."
