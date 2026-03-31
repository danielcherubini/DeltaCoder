#!/bin/bash
# DeltaCoder-9B: Merge LoRA adapter + export to GGUF
# Run on the training box after training completes
#
# Usage: bash scripts/merge_and_export.sh

set -euo pipefail

OUTPUT_DIR="./outputs/deltacoder-9b"
MERGED_DIR="./outputs/deltacoder-9b-merged"
GGUF_DIR="./outputs/deltacoder-9b-gguf"

echo "=== Step 1: Merge LoRA adapter into base model ==="
python -m axolotl.cli.merge_lora configs/deltacoder-9b-lora.yaml \
  --lora_model_dir="$OUTPUT_DIR" \
  --output_dir="$MERGED_DIR"

echo "=== Step 2: Clone llama.cpp (if not present) ==="
if [ ! -d "llama.cpp" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp && make -j$(nproc) llama-quantize && cd ..
fi

echo "=== Step 3: Convert to GGUF (f16) ==="
mkdir -p "$GGUF_DIR"
python llama.cpp/convert_hf_to_gguf.py "$MERGED_DIR" \
  --outfile "$GGUF_DIR/DeltaCoder-9B-f16.gguf" \
  --outtype f16

echo "=== Step 4: Generate quants ==="
for QUANT in Q4_K_S Q4_K_M Q5_K_M Q6_K Q8_0; do
  echo "  Quantizing: $QUANT"
  ./llama.cpp/llama-quantize \
    "$GGUF_DIR/DeltaCoder-9B-f16.gguf" \
    "$GGUF_DIR/DeltaCoder-9B-${QUANT}.gguf" \
    "$QUANT"
done

echo ""
echo "=== Done! ==="
ls -lh "$GGUF_DIR"/*.gguf
echo ""
echo "Transfer GGUFs to your local machine:"
echo "  rsync -avP $GGUF_DIR/*.gguf user@host:/path/to/models/"
