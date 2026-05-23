#!/usr/bin/env bash
# =============================================================================
# Train RearFusionModel (multitask)
# =============================================================================
# Usage:
#   ./scripts/train.sh                         # default config, start fresh
#   ./scripts/train.sh --resume latest         # resume latest checkpoint
#   ./scripts/train.sh --config my_config.yaml
# =============================================================================

set -euo pipefail

# ---- Project root (directory containing this script's parent) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# ---- Environment -------------------------------------------------------------
# Activate virtual environment if present
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# ---- Defaults ----------------------------------------------------------------
CONFIG="${CONFIG:-configs/default.yaml}"
RESUME="${RESUME:-}"

# ---- Parse arguments ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)  CONFIG="$2";  shift 2 ;;
        --resume)  RESUME="$2";  shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- GPU setup ---------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Using GPU(s): ${CUDA_VISIBLE_DEVICES}"

# ---- Logging -----------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="runs/train_${TIMESTAMP}.log"
mkdir -p runs

echo "========================================"
echo " RearFusionModel — Multitask Training"
echo " Config:  ${CONFIG}"
echo " Resume:  ${RESUME:-none}"
echo " Log:     ${LOG_FILE}"
echo "========================================"

RESUME_ARG=""
if [ -n "${RESUME}" ]; then
    RESUME_ARG="--resume ${RESUME}"
fi

# ---- Run training ------------------------------------------------------------
python trainers/train_multitask.py \
    --config "${CONFIG}" \
    ${RESUME_ARG} \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "Training complete. Logs saved to: ${LOG_FILE}"
echo "TensorBoard:  tensorboard --logdir runs/"
