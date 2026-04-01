#!/bin/bash
# DeltaCoder v1.2 — Vast.ai Provisioning Script
#
# Auto-installs Unsloth + GDN acceleration + VLM packing patch on boot.
# Set as PROVISIONING_SCRIPT env var when creating Vast.ai instances.
#
# Usage (in vastai create instance):
#   --env '-e PROVISIONING_SCRIPT="https://raw.githubusercontent.com/danielcherubini/DeltaCoder/main/v1.2/scripts/provision.sh"'
#
# Expects: vastai/pytorch image with matching CUDA toolkit
# Volume: 300GB at /workspace/ with training data already uploaded

set -euo pipefail

LOG="/workspace/provision.log"
echo "=== DeltaCoder v1.2 Provisioning ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

# ---------- Venv (on volume so it persists) ----------
VENV="/workspace/venv"
if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    echo "Venv already exists at $VENV, reusing..." | tee -a "$LOG"
else
    echo "Creating venv at $VENV..." | tee -a "$LOG"
    uv venv "$VENV" --python 3.12 2>&1 | tee -a "$LOG"
fi

source "$VENV/bin/activate"
echo "Python: $(python3 --version)" | tee -a "$LOG"

# ---------- Install Unsloth + deps ----------
echo "Installing unsloth..." | tee -a "$LOG"
uv pip install unsloth 2>&1 | tee -a "$LOG"

echo "Installing flash-linear-attention..." | tee -a "$LOG"
uv pip install flash-linear-attention 2>&1 | tee -a "$LOG"

echo "Installing causal-conv1d (CUDA compilation for SM 9.0 / H100 only)..." | tee -a "$LOG"
export TORCH_CUDA_ARCH_LIST="9.0"
export PATH="/usr/local/cuda/bin:$PATH"
uv pip install causal-conv1d --no-build-isolation 2>&1 | tee -a "$LOG"

# ---------- VLM Packing Patch ----------
PATCH_SCRIPT="/workspace/patch_vlm_packing.py"
if [ -f "$PATCH_SCRIPT" ]; then
    echo "Applying VLM packing unblock patch..." | tee -a "$LOG"
    python3 "$PATCH_SCRIPT" 2>&1 | tee -a "$LOG"
else
    echo "WARNING: patch_vlm_packing.py not found at $PATCH_SCRIPT" | tee -a "$LOG"
    echo "VLM packing will be blocked. Upload the patch script to /workspace/" | tee -a "$LOG"
fi

# ---------- Verify ----------
echo "" | tee -a "$LOG"
echo "=== Verification ===" | tee -a "$LOG"
python3 -c "
import unsloth; print(f'Unsloth: {unsloth.__version__}')
import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')
try:
    import causal_conv1d; print(f'causal-conv1d: OK')
except: print('causal-conv1d: FAILED')
try:
    import fla; print(f'flash-linear-attention: OK')
except: print('flash-linear-attention: FAILED')
" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" | tee -a "$LOG"
echo "Disk: $(df -h /workspace | tail -1)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "=== Provisioning complete: $(date) ===" | tee -a "$LOG"
echo "Activate venv: source /workspace/venv/bin/activate" | tee -a "$LOG"
