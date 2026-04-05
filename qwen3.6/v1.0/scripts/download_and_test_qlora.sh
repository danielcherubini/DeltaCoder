#!/bin/bash
# Download Qwen3.6-9B and run QLoRA dry run on A6000/L40S
# Run with: nohup bash /workspace/download_and_test_qlora.sh > /workspace/qlora_test.log 2>&1 &
set -euo pipefail

source /workspace/venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN="${HF_TOKEN:-}"

# TODO: Verify model name when Qwen3.6 open weights release
MODEL_NAME="Qwen/Qwen3.6-9B"

echo "=== Step 1: Download model to HF cache ==="
python3 << PYEOF
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
print("Downloading $MODEL_NAME to HF cache...")
path = snapshot_download(
    "$MODEL_NAME",
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"Cached at: {path}")
PYEOF
echo "Model download complete."

echo ""
echo "=== Step 2: Run QLoRA dry run (20 steps) ==="
python3 /workspace/train_unsloth.py \
    --data /workspace/v1.3_pretokenized_pruned.parquet \
    --max-steps 20 \
    --qlora \
    --logging-steps 1

echo ""
echo "=== Step 3: GPU memory usage ==="
nvidia-smi

echo ""
echo "=== Done ==="
