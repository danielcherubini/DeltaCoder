# AGENTS.md — DeltaCoder v1.2 (v1.3 planned)

## 1. Project Overview

**DeltaCoder** is a code-specialized LLM trained on Qwen3.5-9B with:
- **v1.2**: SFT + DPO at 8192 context (truncated OCR traces)
- **v1.3**: SFT + DPO at 32768 context (full OCR reasoning traces)
- Target: GitHub + OCR code traces for complex reasoning

## 2. Repository Structure

```
DeltaCoder/
├── v1.1/
│   ├── configs/       # v1.1 Axolotl configs (SFT + DPO)
│   ├── scripts/       # v1.1 scripts (train_unsloth, merge_and_export, etc.)
│   ├── data/          # v1.1 DPO pairs (gitignored)
│   ├── outputs/       # v1.1 DPO adapter (gitignored)
│   └── logs/          # v1.1 training logs (gitignored)
├── v1.2/
│   ├── configs/       # v1.2 Axolotl SFT config
│   ├── scripts/       # v1.2 scripts (pretokenize, train_dpo, preprocess_*, etc.)
│   ├── data/          # v1.2 SFT training data + preprocessed datasets (gitignored)
│   ├── lora_adapter/  # v1.2 SFT LoRA adapter (gitignored)
│   ├── merged/        # v1.2 merged SFT model (17GB, gitignored)
│   └── v1.2_axolotl_train.log
├── v1.3/
│   ├── configs/       # v1.3 Axolotl config (sequence_len=32768)
│   └── scripts/       # v1.3 scripts (pretokenize.py with 32K context)
├── docs/              # Documentation + plans
├── AGENTS.md          # This file
└── README.md
```

## 3. Vast.ai Instance

**Connect**: `ssh -T -o StrictHostKeyChecking=no -p 28137 root@213.5.130.43`

**Find new instances**: `vastai search offers 'gpu_name=H200 num_gpus=1 dph<2.0'`

**Monitoring commands**:
```bash
# GPU health
nvidia-smi -q -d MEMORY,TEMPERATURE,FAN

# Training process
ps aux | grep accelerate
tail -f logs/*.log

# Disk usage
df -h
```

## 4. CRITICAL RULES — DO NOT VIOLATE

### NEVER use:
- **Unsloth DPOTrainer** — crashes with `KeyError: 'images'` on Qwen3.5 VLM
- **flash_attention_2** with Qwen3.5 GDN — causes `cudaErrorIllegalAddress`
- **vLLM stable releases** for Qwen3.5-9B — stable versions (0.11.0, 0.18.1) fail weight loading. **Must use vLLM nightly** (`uv pip install --pre -U vllm --extra-index-url https://wheels.vllm.ai/nightly`)

### Use vLLM nightly for inference (DPO pair generation):

Qwen3.5-9B text-only fine-tunes require special handling for vLLM:

1. **Separate venv** with vLLM nightly + transformers 5.x (system vLLM has transformers 4.57 which doesn't know `qwen3_5_text`)
2. **Wrapped config** — the merged SFT model outputs a flat `qwen3_5_text` config, but vLLM only supports the VL wrapper format (`qwen3_5` + `Qwen3_5ForConditionalGeneration` + `text_config`). Wrap using the official Qwen3.5-9B config as template.
3. **`--language-model-only`** flag to skip vision encoder loading
4. **First run JIT-compiles** FlashInfer GDN prefill kernels (~15min one-time cost)

**Setup venv (one-time):**
```bash
uv venv /workspace/vllm-env
source /workspace/vllm-env/bin/activate
uv pip install --pre -U vllm --extra-index-url https://wheels.vllm.ai/nightly
uv pip install 'transformers>=5.0'
```

**Wrap config.json (one-time, after SFT merge):**
```python
import json
from huggingface_hub import hf_hub_download

# Get official VL config as template
path = hf_hub_download('Qwen/Qwen3.5-9B', 'config.json')
with open(path) as f:
    official = json.load(f)

# Read flat text config from merged model
with open('/workspace/merged_v1.2/config.json') as f:
    text_config = json.load(f)

# Wrap it: text_config goes inside the VL wrapper
official['text_config'] = text_config
with open('/workspace/merged_v1.2/config.json', 'w') as f:
    json.dump(official, f, indent=2)
```

Also fix `tokenizer_config.json` if it has `"tokenizer_class": "TokenizersBackend"` (axolotl artifact — remove that key).

**Serve:**
```bash
source /workspace/vllm-env/bin/activate
vllm serve /workspace/merged_v1.2 \
    --port 18000 --host 0.0.0.0 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --reasoning-parser qwen3 \
    --dtype auto \
    --language-model-only
```

**Fallback:** If vLLM still fails, use ik_llama.cpp (build from main, NOT release):
```bash
git clone https://github.com/ikawrakow/ik_llama.cpp /workspace/ik_llama.cpp
cd /workspace/ik_llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
python3 /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/merged_v1.2 --outfile /workspace/merged_v1.2.Q8_0.gguf --outtype q8_0
/workspace/ik_llama.cpp/build/bin/llama-server -m /workspace/merged_v1.2.Q8_0.gguf --port 18000 --host 0.0.0.0 -ngl 999 -c 4096 --jinja -fa
```

### ALWAYS use:
- `attn_implementation: sdpa` (SDPA, not flash_attention)
- `micro_batch_size: 1` with sample packing (GDN limitation)
- `dataset_num_proc=1` for Qwen3.5 tokenizer (crashes with multiprocessing)

### GDN target modules (REQUIRED):
```python
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]
```

### Vast.ai env vars:
- Vast.ai scrubs inline env vars
- Must export `HF_TOKEN` separately (it's in `~/.bashrc` on remote)

### SSH:
- Use `ssh -T` not `kitten ssh` for non-interactive commands

## 5. Code Style Conventions

```python
"""
Docstring: One-paragraph summary of function/script.
Short, clear, no fluff.
"""

import argparse
import json
import os
import sys
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# Constants at top
BASE_MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 4096
LORA_R = 64
LORA_ALPHA = 32

def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="...")
    # ...
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    # ...
```

## 6. Key Scripts

| Script | Purpose |
|--------|---------|
| `v1.2/scripts/pretokenize.py` | Tokenize v1.2 data (8192 context) |
| `v1.2/scripts/train_dpo.py` | DPO training on top of SFT-merged model (supports `--ling-coder N` to mix in Ling-Coder-DPO) |
| `v1.2/scripts/merge_and_export_dpo.py` | Merge LoRA + export to GGUF |
| `v1.2/scripts/generate_dpo_pairs.py` | Generate on-policy DPO pairs via OpenAI-compatible API |
| `v1.3/scripts/pretokenize.py` | Tokenize v1.3 data (32768 context) |

## 7. Training Monitoring

```bash
# Watch training log in real-time
tail -f logs/v1.3_axolotl_train.log

# Check GPU memory
watch -n 1 'nvidia-smi'

# Training loss (grep from log)
grep -E "^\s*loss:" logs/v1.3_axolotl_train.log | tail -n 50
```

## 8. HuggingFace Repos

- `danielcherubini/Qwen3.5-DeltaCoder-9B` — DPO adapter
- `danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF` — GGUF quantizations

## 9. v1.3 Structure

- `v1.3/configs/axolotl.yaml` — 32768 sequence length config
- `v1.3/scripts/pretokenize.py` — 32K context pretokenization

## 10. Quick Commands

```bash
# v1.2 DPO training (on-policy only)
python v1.2/scripts/train_dpo.py --sft-model /workspace/merged_v1.2

# v1.2 DPO training (on-policy + 50K Ling-Coder-DPO)
python v1.2/scripts/train_dpo.py --sft-model /workspace/merged_v1.2 --ling-coder 50000

# v1.3 pretokenize (32K context)
python v1.3/scripts/pretokenize.py v1.2/data/v1.2_sft_train.jsonl /dev/shm/train_tokenized_v1.3.jsonl 1

# Dry run v1.3 (verify memory)
accelerate launch -m axolotl.cli.train v1.3/configs/axolotl.yaml --max_steps=20

# Merge + export to GGUF
python v1.2/scripts/merge_and_export_dpo.py --sft-model /workspace/merged_v1.2 \
    --dpo-adapter ./outputs/deltacoder-9b-v1.2-dpo/lora_adapter \
    --merged-dir ./outputs/deltacoder-9b-v1.2-dpo-merged \
    --gguf-dir ./outputs/deltacoder-9b-v1.2-dpo-gguf \
    --filename-prefix DeltaCoder-9B-v1.2-DPO \
    --llama-cpp-dir /workspace/llama.cpp \
    --keep-merged --upload --hf-token $HF_TOKEN
```