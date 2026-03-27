#!/usr/bin/env bash
# Vast.ai instance setup for DPO pair generation
# Run as root on a fresh CUDA 12.1+ instance with ≥24GB VRAM (RTX 4090 recommended)
#
# Usage:
#   bash vastai_setup.sh <HF_TOKEN>
#
# After setup, pair generation runs in the background.
# Monitor with: tail -f /workspace/logs/generate_dpo_pairs.log
# Download results with: rsync -avP root@<instance>:/workspace/data/dpo_pairs.jsonl ./data/

set -euo pipefail

HF_TOKEN="${1:-}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "Usage: $0 <HF_TOKEN>"
    exit 1
fi

WORKSPACE=/workspace
MODEL="danielcherubini/Qwen3.5-DeltaCoder-9B"
VLLM_PORT=8080
SCRIPT_DIR="$WORKSPACE/DeltaCoder/scripts"

echo "=== Step 1: Install dependencies ==="
pip install -q vllm huggingface_hub openai numpy datasets tqdm

echo "=== Step 2: Download model from HuggingFace ==="
huggingface-cli download "$MODEL" \
    --token "$HF_TOKEN" \
    --local-dir "$WORKSPACE/model" \
    --local-dir-use-symlinks False

echo "=== Step 3: Clone DeltaCoder repo ==="
git clone https://github.com/danielcherubini/DeltaCoder.git "$WORKSPACE/DeltaCoder" || true
mkdir -p "$WORKSPACE/DeltaCoder/data" "$WORKSPACE/DeltaCoder/logs"

echo "=== Step 4: Start vLLM server ==="
nohup python -m vllm.entrypoints.openai.api_server \
    --model "$WORKSPACE/model" \
    --served-model-name deltacoder \
    --port $VLLM_PORT \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --enable-prefix-caching \
    > "$WORKSPACE/logs/vllm.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "=== Step 5: Wait for vLLM to be ready ==="
echo "Waiting for vLLM on port $VLLM_PORT..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "vLLM is ready!"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "ERROR: vLLM did not start in 60s. Check $WORKSPACE/logs/vllm.log"
        exit 1
    fi
    sleep 5
done

echo "=== Step 6: Start pair generation ==="
cd "$WORKSPACE/DeltaCoder"
nohup env HF_TOKEN="$HF_TOKEN" python "$SCRIPT_DIR/generate_dpo_pairs.py" \
    --n-problems 10000 \
    --n-samples 8 \
    --api-base "http://localhost:$VLLM_PORT/v1" \
    --model deltacoder \
    > "$WORKSPACE/DeltaCoder/logs/generate_dpo_pairs.log" 2>&1 &

GEN_PID=$!
echo "Pair generation PID: $GEN_PID"

echo ""
echo "=== Setup complete ==="
echo "Monitor vLLM:      tail -f $WORKSPACE/logs/vllm.log"
echo "Monitor progress:  tail -f $WORKSPACE/DeltaCoder/logs/generate_dpo_pairs.log"
echo "Download results:  rsync -avP root@<instance>:$WORKSPACE/DeltaCoder/data/dpo_pairs.jsonl ./data/"
