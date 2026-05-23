#!/usr/bin/env bash
# =============================================================================
# Download V-JEPA Pretrained Weights
# =============================================================================
# Official V-JEPA models from Meta AI Research:
#   https://github.com/facebookresearch/vjepa
#
# Available checkpoints:
#   - ViT-H/16  (best quality, 600M params)   vit_huge.pth
#   - ViT-L/16  (balanced)                     vit_large.pth
#   - ViT-B/16  (lightweight, recommended PoC) vit_base.pth
#
# The script sets the VJEPA_WEIGHTS environment variable so the encoder
# wrapper automatically loads the weights.
# =============================================================================

set -euo pipefail

MODEL_DIR="${VJEPA_MODEL_DIR:-./pretrained}"
mkdir -p "${MODEL_DIR}"

echo "=================================================="
echo " V-JEPA Weight Download"
echo "=================================================="

# Option 1: Direct HuggingFace Hub (recommended)
# The official Meta V-JEPA weights are available at:
#   https://huggingface.co/facebook/vjepa-vit-base-16
#
# Method A: huggingface-cli (preferred)
if command -v huggingface-cli &>/dev/null; then
    echo "[A] Using huggingface-cli to download V-JEPA ViT-Base weights..."
    huggingface-cli download facebook/vjepa-vit-base-16 \
        --local-dir "${MODEL_DIR}/vjepa_vit_base"
    export VJEPA_WEIGHTS="${MODEL_DIR}/vjepa_vit_base/pytorch_model.bin"
    echo "VJEPA_WEIGHTS=${VJEPA_WEIGHTS}"
    echo "export VJEPA_WEIGHTS=${VJEPA_WEIGHTS}" >> ~/.bashrc
    exit 0
fi

# Method B: Python / HuggingFace Hub API
echo "[B] Downloading via Python HuggingFace Hub..."
python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download, snapshot_download
import os

model_dir = os.environ.get('MODEL_DIR', './pretrained')
os.makedirs(model_dir, exist_ok=True)

# Download full repo (includes config + weights)
local_dir = snapshot_download(
    repo_id="facebook/vjepa-vit-base-16",
    local_dir=os.path.join(model_dir, 'vjepa_vit_base'),
)
print(f"Downloaded to: {local_dir}")
PYEOF

export VJEPA_WEIGHTS="${MODEL_DIR}/vjepa_vit_base/pytorch_model.bin"
echo "export VJEPA_WEIGHTS=${VJEPA_WEIGHTS}" >> ~/.bashrc
echo ""
echo "✓ V-JEPA weights downloaded."
echo "  Set VJEPA_WEIGHTS=${VJEPA_WEIGHTS} before training."
echo ""
echo "  If using the mini-model for testing, timm ViT will be used"
echo "  automatically as fallback — no weights download required."

# ============================================================
# Manual download fallback
# ============================================================
echo ""
echo "─────────────────────────────────────────────────────"
echo " Manual download option:"
echo " 1. Visit: https://github.com/facebookresearch/vjepa"
echo " 2. Follow 'Model Zoo' instructions to download weights"
echo " 3. Set: export VJEPA_WEIGHTS=/path/to/vjepa.pt"
echo "─────────────────────────────────────────────────────"
