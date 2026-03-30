# AGENTS.md — DeltaCoder v1.2 (v1.3 planned)

## 1. Project Overview

**DeltaCoder** is a code-specialized LLM trained on Qwen3.5-9B with:
- **v1.2**: SFT + DPO at 8192 context (truncated OCR traces)
- **v1.3**: SFT + DPO at 32768 context (full OCR reasoning traces)
- Target: GitHub + OCR code traces for complex reasoning

## 2. Repository Structure

```
DeltaCoder/
├── configs/           # Axolotl training configs (v1.2, v1.3)
├── data/              # Raw training data (train.jsonl, dpo_pairs)
├── docs/              # Documentation
├── logs/              # Training logs
├── outputs/           # Checkpoints, merged models
├── scripts/           # v1.2 scripts (pretokenize.py, train_dpo.py, merge_and_export_dpo.py)
├── v1.3/
│   ├── configs/       # v1.3 Axolotl config (sequence_len=32768)
│   └── scripts/       # v1.3 scripts (pretokenize.py with 32K context)
├── AGENTS.md          # This file
└── README.md
```

## 3. Vast.ai Instance

**Connect**: `ssh -T -o StrictHostKeyChecking=no -p 28988 root@213.5.130.43`

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
| `scripts/pretokenize.py` | Tokenize v1.2 data (8192 context) |
| `scripts/train_dpo.py` | DPO training on top of SFT-merged model |
| `scripts/merge_and_export_dpo.py` | Merge LoRA + export to GGUF |
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
# v1.2 DPO training
python scripts/train_dpo.py --sft-model /workspace/merged_v1.2

# v1.3 pretokenize (32K context)
python v1.3/scripts/pretokenize.py data/v1.2_sft_train.jsonl /dev/shm/train_tokenized_v1.3.jsonl 1

# Dry run v1.3 (verify memory)
accelerate launch -m axolotl.cli.train v1.3/configs/axolotl.yaml --max_steps=20

# Merge + export to GGUF
python scripts/merge_and_export_dpo.py --sft-model /workspace/merged_v1.2 \
    --dpo-adapter ./outputs/deltacoder-9b-v1.2-dpo/lora_adapter \
    --merged-dir ./outputs/deltacoder-9b-v1.2-dpo-merged \
    --gguf-dir ./outputs/deltacoder-9b-v1.2-dpo-gguf \
    --filename-prefix DeltaCoder-9B-v1.2-DPO \
    --llama-cpp-dir /workspace/llama.cpp \
    --keep-merged --upload --hf-token $HF_TOKEN
```