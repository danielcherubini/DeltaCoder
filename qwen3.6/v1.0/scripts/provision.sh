#!/bin/bash
# DeltaCoder Qwen3.6 v1.0 — Vast.ai Provisioning Script
#
# Auto-installs Unsloth + GDN acceleration + VLM packing patch on boot.
# Download and run manually after SSH:
#   curl -fsSL -o /workspace/provision.sh https://raw.githubusercontent.com/danielcherubini/DeltaCoder/main/qwen3.6/v1.0/scripts/provision.sh
#   bash /workspace/provision.sh
#
# Expects: vastai/pytorch image with matching CUDA toolkit (e.g. vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312-2026-03-26)
# Local disk: 80GB (no persistent volume needed — bootstrap from scratch)

set -euo pipefail

LOG="/workspace/provision.log"
echo "=== DeltaCoder Qwen3.6 v1.0 Provisioning ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

# ---------- HF cache on workspace (not root disk!) ----------
export HF_HOME="/workspace/.cache/huggingface"
mkdir -p "$HF_HOME"

# ---------- CUDA path ----------
# Find the CUDA toolkit (vastai/pytorch images put it in /usr/local/cuda-XX.Y)
export PATH="/usr/local/cuda/bin:$PATH"
echo "nvcc: $(which nvcc 2>/dev/null || echo 'NOT FOUND')" | tee -a "$LOG"
echo "nvcc version: $(nvcc --version 2>&1 | tail -1)" | tee -a "$LOG"

# ---------- Venv (on workspace so it persists across reconnects) ----------
VENV="/workspace/venv"
if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    echo "Venv already exists at $VENV, reusing..." | tee -a "$LOG"
else
    echo "Creating venv at $VENV..." | tee -a "$LOG"
    uv venv "$VENV" --python 3.12 2>&1 | tee -a "$LOG"
fi

source "$VENV/bin/activate"
echo "Python: $(python3 --version)" | tee -a "$LOG"

# ---------- Install Unsloth + flash-linear-attention ----------
echo "Installing unsloth..." | tee -a "$LOG"
uv pip install unsloth 2>&1 | tee -a "$LOG"

echo "Installing flash-linear-attention..." | tee -a "$LOG"
uv pip install flash-linear-attention 2>&1 | tee -a "$LOG"

# ---------- Detect GPU SM architecture ----------
SM_ARCH=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).decode().strip().split('\n')[0]
major, minor = out.strip().split('.')
print(f'sm_{major}{minor}')
" 2>/dev/null || echo "sm_90")
SM_COMPUTE=$(echo "$SM_ARCH" | sed 's/sm_/compute_/')
echo "Detected GPU architecture: $SM_ARCH ($SM_COMPUTE)" | tee -a "$LOG"

# ---------- Build causal-conv1d (detected arch) ----------
# causal-conv1d's setup.py HARDCODES all GPU architectures and ignores TORCH_CUDA_ARCH_LIST.
# We must clone, patch setup.py to only build for the current GPU.
# TODO: Verify causal-conv1d is still needed for Qwen3.6 (depends on GDN usage)
echo "Building causal-conv1d ($SM_ARCH only)..." | tee -a "$LOG"

CONV1D_DIR="/workspace/causal-conv1d"
if [ -d "$CONV1D_DIR" ]; then
    echo "  Removing old causal-conv1d clone..." | tee -a "$LOG"
    rm -rf "$CONV1D_DIR"
fi

git clone https://github.com/Dao-AILab/causal-conv1d.git "$CONV1D_DIR" 2>&1 | tee -a "$LOG"

# Patch setup.py: replace all hardcoded arch flags with detected GPU arch
SM_ARCH_VAL="$SM_ARCH" SM_COMPUTE_VAL="$SM_COMPUTE" python3 - "$CONV1D_DIR/setup.py" << 'PYEOF'
import sys, os

sm_arch = os.environ["SM_ARCH_VAL"]
sm_compute = os.environ["SM_COMPUTE_VAL"]

setup_py = sys.argv[1]
with open(setup_py, "r") as f:
    content = f.read()

# Replace the base hardcoded arch flags
old_block = '''        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")'''

new_block = f'''        cc_flag.append("-gencode")
        cc_flag.append("arch={sm_compute},code={sm_arch}")'''

if old_block not in content:
    if "arch=compute_75" not in content:
        print("Already patched or format changed — skipping")
        sys.exit(0)
    else:
        print("ERROR: Could not find expected arch block in setup.py")
        sys.exit(1)

content = content.replace(old_block, new_block)

# Remove conditional arch additions (sm_90 duplicate, sm_100, sm_120, sm_103, sm_110, sm_121)
lines = content.split('\n')
new_lines = []
skip = False
for i, line in enumerate(lines):
    if 'bare_metal_version >= Version' in line:
        # Check if next non-empty line has cc_flag — if so, skip this block
        for j in range(i+1, min(i+10, len(lines))):
            if lines[j].strip():
                if 'cc_flag' in lines[j]:
                    skip = True
                break
        if skip:
            continue
    if skip:
        if line.strip().startswith('cc_flag') or line.strip() == '':
            continue
        else:
            skip = False
    new_lines.append(line)

content = '\n'.join(new_lines)

with open(setup_py, "w") as f:
    f.write(content)

# Verify
archs = [l.strip() for l in content.split('\n') if 'arch=compute' in l]
print(f"Patched setup.py — {len(archs)} architecture(s):")
for a in archs:
    print(f"  {a}")
PYEOF

echo "  Installing from patched source..." | tee -a "$LOG"
uv pip install "$CONV1D_DIR" --no-build-isolation 2>&1 | tee -a "$LOG"

# ---------- Download scripts from GitHub ----------
REPO_RAW="https://raw.githubusercontent.com/danielcherubini/DeltaCoder/main"
echo "Downloading scripts from GitHub..." | tee -a "$LOG"

curl -fsSL -o /workspace/patch_vlm_packing.py "$REPO_RAW/qwen3.6/v1.0/scripts/patch_vlm_packing.py" 2>&1 | tee -a "$LOG"
curl -fsSL -o /workspace/train_unsloth.py "$REPO_RAW/qwen3.6/v1.0/scripts/train_unsloth.py" 2>&1 | tee -a "$LOG"

echo "  Downloaded patch_vlm_packing.py and train_unsloth.py" | tee -a "$LOG"

# ---------- VLM Packing Patch ----------
# TODO: Verify this patch is still needed for Qwen3.6 — Unsloth may have native support
echo "Applying VLM packing unblock patch..." | tee -a "$LOG"
python3 /workspace/patch_vlm_packing.py 2>&1 | tee -a "$LOG"

# ---------- Pre-download model ----------
# TODO: Update model name when Qwen3.6 open weights release
MODEL_NAME="Qwen/Qwen3.6-9B"
echo "Pre-downloading $MODEL_NAME model..." | tee -a "$LOG"
python3 -c "
from transformers import AutoTokenizer, AutoConfig
MODEL = '$MODEL_NAME'
print(f'Downloading tokenizer for {MODEL}...')
AutoTokenizer.from_pretrained(MODEL)
print(f'Downloading config for {MODEL}...')
AutoConfig.from_pretrained(MODEL)
print('Model files will be downloaded on first training run (FastVisionModel handles this)')
" 2>&1 | tee -a "$LOG"

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
