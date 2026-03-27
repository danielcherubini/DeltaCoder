#!/usr/bin/env bash
# Vast.ai instance setup for DPO pair generation using llama.cpp + GGUF
# Run as root on a fresh CUDA 12.1+ instance
#
# Usage:
#   bash vastai_setup.sh <HF_TOKEN>
#
# After setup, pair generation runs in the background.
# Monitor with: tail -f /workspace/DeltaCoder/logs/generate_dpo_pairs.log
# Download results with: rsync -avP root@<instance>:/workspace/DeltaCoder/data/dpo_pairs.jsonl ./data/

set -euo pipefail

HF_TOKEN="${1:-}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "Usage: $0 <HF_TOKEN>"
    exit 1
fi

WORKSPACE=/workspace
GGUF_REPO="danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF"
GGUF_FILE="Qwen3.5-DeltaCoder-9B-Q4_K_M.gguf"
SERVER_PORT=8080
SCRIPT_DIR="$WORKSPACE/DeltaCoder/scripts"

echo "=== Step 1: Create directories ==="
mkdir -p "$WORKSPACE/logs" "$WORKSPACE/model"

echo "=== Step 2: Install Python dependencies ==="
pip install -q huggingface_hub openai numpy datasets tqdm

echo "=== Step 3: Download GGUF from HuggingFace ==="
huggingface-cli download "$GGUF_REPO" "$GGUF_FILE" \
    --token "$HF_TOKEN" \
    --local-dir "$WORKSPACE/model"

echo "=== Step 4: Download llama.cpp release binary ==="
LLAMA_RELEASE="b5569"
LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_RELEASE}/llama-${LLAMA_RELEASE}-bin-ubuntu-x64.zip"
curl -L "$LLAMA_URL" -o /tmp/llama.zip
unzip -o /tmp/llama.zip -d /workspace/llama.cpp
chmod +x /workspace/llama.cpp/build/bin/llama-server

echo "=== Step 5: Clone DeltaCoder repo ==="
if [[ ! -d "$WORKSPACE/DeltaCoder/.git" ]]; then
    git clone https://github.com/danielcherubini/DeltaCoder.git "$WORKSPACE/DeltaCoder"
fi
mkdir -p "$WORKSPACE/DeltaCoder/data" "$WORKSPACE/DeltaCoder/logs"

echo "=== Step 6: Start llama-server ==="
nohup /workspace/llama.cpp/build/bin/llama-server \
    -m "$WORKSPACE/model/$GGUF_FILE" \
    -ngl 999 \
    -c 4096 \
    --port $SERVER_PORT \
    --host 0.0.0.0 \
    > "$WORKSPACE/logs/llama-server.log" 2>&1 &

SERVER_PID=$!
echo "llama-server PID: $SERVER_PID"

echo "=== Step 7: Wait for server to be ready ==="
echo "Waiting for llama-server on port $SERVER_PORT..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
        echo "llama-server is ready!"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "ERROR: llama-server did not start in 60s. Check $WORKSPACE/logs/llama-server.log"
        exit 1
    fi
    sleep 5
done

echo "=== Step 8: Start pair generation ==="
cd "$WORKSPACE/DeltaCoder"
nohup env HF_TOKEN="$HF_TOKEN" python "$SCRIPT_DIR/generate_dpo_pairs.py" \
    --n-problems 10000 \
    --n-samples 8 \
    --api-base "http://localhost:$SERVER_PORT/v1" \
    --model deltacoder \
    > "$WORKSPACE/DeltaCoder/logs/generate_dpo_pairs.log" 2>&1 &

GEN_PID=$!
echo "Pair generation PID: $GEN_PID"

echo ""
echo "=== Setup complete ==="
echo "Monitor server:    tail -f $WORKSPACE/logs/llama-server.log"
echo "Monitor progress:  tail -f $WORKSPACE/DeltaCoder/logs/generate_dpo_pairs.log"
echo "Download results:  rsync -avP root@<instance>:$WORKSPACE/DeltaCoder/data/dpo_pairs.jsonl ./data/"
