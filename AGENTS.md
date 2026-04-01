# AGENTS.md — DeltaCoder v1.2

## 1. Project Overview

**DeltaCoder** is a code-specialized LLM trained on Qwen3.5-9B with:
- **v1.2**: SFT + DPO at 32768 context (full reasoning traces)
- Priorities (in order): (1) Coding, (2) Tool Calling, (3) Agentic Workflows
- Target: THE BEST 9B for those three tasks
- **MUST preserve vision capabilities** — Qwen3.5-9B is a VLM

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

## 3. Vast.ai Infrastructure

### Current Instance
**Connect**: `ssh -T -o StrictHostKeyChecking=no -p <PORT> root@<VAST_IP>`
**Image**: `nvcr.io/nvidia/pytorch:26.01-py3` (PyTorch NGC)
**Volume**: 300GB persistent volume mounted at `/workspace/`

### Instance Creation (with persistent volume)
```bash
# Search for H100 SXM offers (US/EU only for latency, skip host 68137 — broken SSH)
vastai search offers 'gpu_name=H100_SXM num_gpus=1 dph<2.5 reliability>0.95 geolocation in [US,DE,NL,GB,CZ]' --order 'dph' --raw

# Create instance WITH persistent volume (volume survives instance destruction)
vastai create instance <OFFER_ID> \
  --image nvcr.io/nvidia/pytorch:26.01-py3 \
  --env '-e DATA_DIRECTORY="/workspace/" -e JUPYTER_DIR="/"' \
  --onstart-cmd 'env >> /etc/environment;mkdir -p ${DATA_DIRECTORY:-/workspace};' \
  --disk 16 \
  --create-volume <VOLUME_ASK_ID> --volume-size 300 --mount-path '/workspace' \
  --ssh --direct

# Copy files to/from instances (better than scp)
vastai copy local:path/to/file C.<INSTANCE_ID>:/workspace/
vastai copy C.<INSTANCE_ID>:/workspace/output local:output/
```

### Volume Strategy
Persistent volume at `/workspace/` holds all data, scripts, models, and checkpoints.
Survives instance destruction — can swap templates without re-uploading.

**Current volume**: ID 33974210, 300GB, host 260094 (US), $0.20/GB/mo

**Swapping templates on the same volume:**
```bash
# Destroy instance (volume persists)
vastai destroy instance <INSTANCE_ID>

# Recreate on same host with different image, reattach volume
# NOTE: --link-volume may fail with "access denied" if the offer ID doesn't match
# the volume's host. Use an offer on the SAME host (check host_id in offer search).
# Also: the first offer ID tried (33748664) failed with access denied even on the
# same host. The second offer (32856289) on the same host worked. May be a timing/
# caching issue — try a different offer on the same host if one fails.
vastai create instance <OFFER_ID> \
  --image <NEW_IMAGE> \
  --link-volume <VOLUME_ID> \
  --mount-path '/workspace' \
  --ssh --direct
```

**Volume contents:**
```
/workspace/
├── venv/                    # Python venv (persists across restarts)
├── v1.2_sft_train.jsonl     # 262K training rows (5.1GB)
├── train_unsloth.py         # SFT training script
├── patch_vlm_packing.py     # VLM packing unblock patch
├── outputs/                 # LoRA adapters + checkpoints
├── merged_v1.2/             # Merged SFT model (~18GB)
└── logs/                    # Training logs
```

### Versioned Templates (for future phases)
Vast.ai has version-tagged Docker images with precise CUDA/PyTorch/Python combinations:
- **vastai/pytorch**: Tags like `2.10.0-cu128-cuda-12.9-mini-py312-2026-03-26` (PyTorch 2.10, CUDA toolkit 12.9, Python 3.12). Use this for training — CUDA toolkit matches PyTorch's compiled CUDA version, so `causal-conv1d` compiles without errors.
- **nvcr.io/nvidia/pytorch:26.01-py3**: NGC PyTorch — AVOID for compiling CUDA extensions. Ships CUDA toolkit 13.1 but PyTorch compiled for CUDA 12.8 → `causal-conv1d` fails with version mismatch.
- **vastai/vllm:nightly-2026-03-02-cuda-12.9**: vLLM nightly (for DPO pair generation)
- **vastai/base-image:cuda-13.2.0-auto**: CUDA 13.2, clean base
- **unsloth/unsloth:latest**: Unsloth Studio (NOTE: SSH may not work with Vast.ai — uses non-standard port mappings)

### Environment Variables (vastai/pytorch image)
- `WORKSPACE`: Change default working directory
- `PROVISIONING_SCRIPT`: Auto-run setup script from URL on instance boot (GitHub, Gist, any plain-text URL)
- `TENSORBOARD_LOG_DIR`: Customize Tensorboard log dir (defaults to /workspace)
- `ENABLE_HTTPS`: Force HTTPS connections

The `PROVISIONING_SCRIPT` is powerful — point it at a gist that installs unsloth + deps
and the instance is ready to train on boot. No manual SSH setup needed.

### CRITICAL: Match CUDA toolkit to PyTorch's compiled CUDA version
- `causal-conv1d` (required for GDN acceleration) compiles CUDA kernels at install time
- If the system CUDA toolkit version doesn't match PyTorch's compiled CUDA version, build fails
- NGC PyTorch `26.01-py3` has toolkit 13.1 but torch compiled for 12.8 → FAILS
- `vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312-2026-03-26` has toolkit 12.9 + torch for 12.8 → WORKS (close enough)

### CRITICAL: flash-linear-attention is REQUIRED for training

Without `flash-linear-attention` + `causal-conv1d`, Qwen3.5's GDN layers (24/32) fall back to
slow torch CPU implementation → 0% GPU utilization, training takes days instead of hours.
- `flash-linear-attention`: pure Python wheel, installs instantly
- `causal-conv1d`: requires CUDA compilation (~20-45 min depending on CPU)
- Install with: `uv pip install causal-conv1d flash-linear-attention --no-build-isolation`
- MUST use `--no-build-isolation` to avoid pip pulling wrong PyTorch/CUDA version
- Use `uv` instead of `pip` — much faster installs

### Monitoring commands
```bash
# GPU health
nvidia-smi -q -d MEMORY,TEMPERATURE,FAN

# Training process
ps aux | grep python
tail -f /workspace/logs/*.log

# Disk usage
df -h /workspace
```

## 4. CRITICAL RULES — DO NOT VIOLATE

### NEVER use:
- **Unsloth DPOTrainer** — crashes with `KeyError: 'images'` on Qwen3.5 VLM
- **flash_attention_2** with Qwen3.5 GDN — causes `cudaErrorIllegalAddress`
- **vLLM stable releases** for Qwen3.5-9B — stable versions (0.11.0, 0.18.1) fail weight loading. **Must use vLLM nightly** (`uv pip install --pre -U vllm --extra-index-url https://wheels.vllm.ai/nightly`)

### SFT Training Approach: Unsloth FastVisionModel + Packing Unblock

Qwen3.5-9B is a unified VLM — there is NO separate text-only model. Every variant uses
`Qwen3_5ForConditionalGeneration`. This creates issues for text-only fine-tuning with packing:

1. **Unsloth blocks sample packing for VLMs** — checks `ForConditionalGeneration` in architectures
   and `vision_config` in model config, plus `ProcessorMixin` check on tokenizer
2. **Without packing**: ~182 hours for 262K rows ($333) — too slow
3. **With packing**: ~38 hours ($69) — feasible

**Solution** (validated by community in unslothai/unsloth#4160):
- Load with `FastVisionModel` (preserves all vision weights)
- LoRA with `finetune_vision_layers=False` (only train language layers)
- Apply VLM packing unblock patch (`patch_vlm_packing.py`) to remove `is_vlm` check from `trainer.py`
- Pass `tokenizer` (not processor) to `SFTTrainer` to bypass `ProcessorMixin` check
- Use `packing=True`, `max_seq_length=32768`, `per_device_train_batch_size=1`

**NaN gradient risk**: Issue #4160 reports NaN gradients at >16K context, but this appears to be
a total-tokens-per-batch issue (~64K threshold). At batch_size=1 + 32K, total is ~32K — safely below.

**32K OOM background**: The VL model materializes full logits tensor (32K × 248K vocab ≈ 30GB)
before computing cross_entropy. Unsloth handles this internally with fused CE — no OOM.
Axolotl's Liger integration only patches `ForCausalLM`, not `ForConditionalGeneration`.

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
| `v1.2/scripts/train_unsloth.py` | SFT training with Unsloth FastVisionModel + packing at 32K |
| `v1.2/scripts/patch_vlm_packing.py` | Removes VLM packing block from unsloth/trainer.py |
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

## 11. Key Discoveries & Constraints

### 32K context cross_entropy OOM (Axolotl path — NOT used)
- The VL model (`Qwen3_5ForConditionalGeneration`) materializes full logits tensor
  (32K × 248K vocab ≈ 30GB) before computing cross_entropy loss
- Axolotl's Liger integration only patches `ForCausalLM`, not `ForConditionalGeneration`
- Axolotl PR #2908 added generic fused CE for arbitrary models, but still targets `ForCausalLM`
- **Solution**: Use Unsloth instead (handles fused CE internally)

### Unsloth VLM packing block
- Unsloth deliberately blocks sample packing for VLMs (issue #4120 — open feature request)
- Two checks: `is_vlm` (architectures + vision_config) and `isinstance(ProcessorMixin)`
- **Bypass**: `patch_vlm_packing.py` removes `is_vlm` check; passing tokenizer (not processor)
  to SFTTrainer bypasses the ProcessorMixin check

### NaN gradients at high total tokens per batch (issue #4160)
- At batch_size=4 + 17K context (~68K total tokens), gradients go NaN
- At batch_size=4 + 16K context (~64K total tokens), high grads but recovers
- At batch_size=1 + 32K context (~32K total tokens), safely below threshold
- **Mitigation**: Use batch_size=1 with packing

### Vast.ai volume limitations
- Regular volumes (`search volumes`) only attach to instances on the SAME physical machine
- No H100 SXM hosts currently offer regular volume storage
- **Solution**: Use `--create-volume` flag on `create instance` which creates a network volume
  that persists independently and can be reattached

### Unsloth Docker image SSH issues
- `unsloth/unsloth:latest` Docker image has its own port mappings that conflict with Vast.ai SSH
- The official Unsloth template exposes ports 1111, 6006, 8080, 8384, 8888, 72299 — NOT port 22
- SSH is handled by Vast.ai's proxy, not the container
- **Workaround**: Use PyTorch NGC image + `pip install unsloth` instead

### LLaMA-Factory Qwen3.5 support
- LLaMA-Factory supports Qwen3.5 fine-tuning (official blog post)
- But no evidence of 32K text-only training with packing at scale
- Unsloth remains the better option for our use case