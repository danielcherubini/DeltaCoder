# AGENTS.md — DeltaCoder

## 1. Project Overview

**DeltaCoder** is a code-specialized LLM fine-tune:
- **qwen3.5/v1.1**: SFT + DPO on Qwen3.5-9B at 32768 context (completed pipeline, all scripts validated)
- **qwen3.5/35b-a3b**: SFT fine-tune of Qwen3.5-35B-A3B MoE (NEW — plan written, scripts pending)
- **qwen3.6/v1.0**: SFT + DPO on **Qwen3.6** (BLOCKED — waiting for open weights release)
- Priorities (in order): (1) Coding, (2) Tool Calling, (3) Agentic Workflows
- Target: THE BEST 9B for those three tasks
- **MUST preserve vision capabilities** — base models are VLMs

## 2. Repository Structure

```
DeltaCoder/
├── qwen3.5/
│   ├── v1.0/              # Original 9B (SFT + DPO, Axolotl configs)
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── data/          # DPO pairs (gitignored)
│   │   ├── outputs/       # DPO adapter (gitignored)
│   │   └── logs/          # training logs (gitignored)
│   ├── v1.1/              # Revised 9B (Jackrong-inspired, Unsloth + packing at 32K)
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── data/          # SFT training data + preprocessed datasets (gitignored)
│   │   ├── lora_adapter/  # SFT LoRA adapter (gitignored)
│   │   └── merged/        # Merged SFT model (gitignored)
│   └── 35b-a3b/           # MoE fine-tune (NEW — plan written, scripts pending)
│       └── scripts/
├── qwen3.6/
│   └── v1.0/              # Qwen3.6 (BLOCKED — waiting for open weights)
│       ├── configs/
│       ├── scripts/
│       └── data/          # Training data (gitignored)
├── docs/              # Documentation + plans
├── AGENTS.md          # This file
└── README.md
```

## 3. Vast.ai Infrastructure

### Current Instance
**No active instance or volume.** All Vast.ai resources torn down (2026-04-02).

### Instance Creation (no volume — use local disk)
```bash
# Search for H100 SXM offers (US/EU only for latency, skip host 68137 — broken SSH)
vastai search offers 'gpu_name=H100_SXM num_gpus=1 dph<2.5 reliability>0.95 geolocation in [US,DE,NL,GB,CZ]' --order 'dph' --raw

# Create instance with 80GB local disk (no volume needed — bootstrap from scratch)
vastai create instance <OFFER_ID> \
  --image vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312-2026-03-26 \
  --env '-e DATA_DIRECTORY="/workspace/"' \
  --disk 80 \
  --ssh --direct

# Then SSH in and bootstrap:
curl -fsSL -o /workspace/provision.sh https://raw.githubusercontent.com/danielcherubini/DeltaCoder/main/qwen3.5/v1.1/scripts/provision.sh
bash /workspace/provision.sh
# Upload pre-tokenized data:
scp -P <PORT> qwen3.5/v1.1/data/v1.1_pretokenized/*.parquet root@<IP>:/workspace/v1.1_pretokenized/
```

**NOTE:** The `PROVISIONING_SCRIPT` env var does NOT auto-run on the `vastai/pytorch` image.
Must download and run `provision.sh` manually after SSH.

### Volume Strategy (DEPRECATED)
Previous approach used persistent volumes, but they are fragile on Vast.ai:
- `--create-volume` needs a volume ask on the EXACT same machine as the offer
- `--link-volume` often fails with "access denied" even on the same host
- Volume storage costs $0.20/GB/mo ($60/mo for 300GB)

**New approach:** Use 80GB local disk + bootstrap from scratch via `provision.sh`.
The provisioning script installs everything in ~4 minutes. Pre-tokenized data (4.4GB)
is uploaded via scp. No persistent volume needed.

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

The `PROVISIONING_SCRIPT` env var does NOT auto-run on the `vastai/pytorch` image despite docs.
Must download and run `provision.sh` manually after SSH.

**Provisioning script**: `qwen3.5/v1.1/scripts/provision.sh` — **validated 2026-04-02**, installs
everything in ~4 minutes on a fresh H100 instance:
1. Creates Python 3.12 venv at `/workspace/venv/`
2. Installs Unsloth 2026.3.18 + all dependencies
3. Clones + patches causal-conv1d for detected GPU SM arch (~3 min compile)
4. Installs flash-linear-attention
5. Downloads train_unsloth.py + patch_vlm_packing.py from GitHub
6. Applies VLM packing unblock patch
7. Pre-downloads Qwen3.5-9B tokenizer/config

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
- Install with: `TORCH_CUDA_ARCH_LIST="9.0" uv pip install causal-conv1d flash-linear-attention --no-build-isolation`
- MUST set `TORCH_CUDA_ARCH_LIST="9.0"` — only compile for H100 (Hopper). Without this it builds for all GPU architectures and takes forever.
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
with open('/workspace/merged_v1.1/config.json') as f:
    text_config = json.load(f)

# Wrap it: text_config goes inside the VL wrapper
official['text_config'] = text_config
with open('/workspace/merged_v1.1/config.json', 'w') as f:
    json.dump(official, f, indent=2)
```

Also fix `tokenizer_config.json` if it has `"tokenizer_class": "TokenizersBackend"` (axolotl artifact — remove that key).

**Serve:**
```bash
source /workspace/vllm-env/bin/activate
vllm serve /workspace/merged_v1.1 \
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
python3 /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/merged_v1.1 --outfile /workspace/merged_v1.1.Q8_0.gguf --outtype q8_0
/workspace/ik_llama.cpp/build/bin/llama-server -m /workspace/merged_v1.1.Q8_0.gguf --port 18000 --host 0.0.0.0 -ngl 999 -c 4096 --jinja -fa
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
| `qwen3.5/v1.1/scripts/train_unsloth.py` | SFT training with Unsloth FastVisionModel + packing at 32K |
| `qwen3.5/v1.1/scripts/pretokenize_for_sft.py` | Pre-tokenize training data to parquet shards (run on Romulus) |
| `qwen3.5/v1.1/scripts/patch_vlm_packing.py` | Removes VLM packing block from unsloth/trainer.py |
| `qwen3.5/v1.1/scripts/provision.sh` | Vast.ai bootstrap: installs all deps in ~4 min |
| `qwen3.5/v1.1/scripts/pretokenize.py` | Tokenize v1.1 data (8192 context) |
| `qwen3.5/v1.1/scripts/train_dpo.py` | DPO training on top of SFT-merged model (supports `--ling-coder N` to mix in Ling-Coder-DPO) |
| `qwen3.5/v1.1/scripts/merge_and_export_dpo.py` | Merge LoRA + export to GGUF |
| `qwen3.5/v1.1/scripts/generate_dpo_pairs.py` | Generate on-policy DPO pairs via OpenAI-compatible API |
| `qwen3.5/v1.1/scripts/build_training_mix.py` | Build final JSONL training mix from filtered sources |
| `qwen3.5/v1.1/scripts/filter_for_v12_pruned.py` | Apply tiered 8K/16K token filters to each source |
| `qwen3.5/v1.1/scripts/preprocess_competitive_programming.py` | Download + convert Jackrong competitive programming dataset |
| `qwen3.5/v1.1/scripts/preprocess_qwen3_coder_distill.py` | Download + convert Jackrong Qwen3-Coder-480B distill dataset |
| `qwen3.6/v1.0/scripts/pretokenize_for_sft.py` | Pre-tokenize Qwen3.6 v1.0 data (32768 context) |
| `qwen3.6/v1.0/scripts/train_unsloth.py` | Qwen3.6 v1.0 SFT training (adapted for Qwen3.6) |
| `qwen3.6/v1.0/scripts/provision.sh` | Qwen3.6 v1.0 Vast.ai bootstrap (adapted for Qwen3.6) |
| `qwen3.6/v1.0/scripts/train_dpo.py` | Qwen3.6 v1.0 DPO training |
| `qwen3.6/v1.0/scripts/merge_and_export_dpo.py` | Qwen3.6 v1.0 merge LoRA + GGUF export |
| `qwen3.6/v1.0/scripts/generate_dpo_pairs.py` | Qwen3.6 v1.0 DPO pair generation |
| `qwen3.6/v1.0/scripts/build_training_mix.py` | Qwen3.6 v1.0 training mix builder |
| `qwen3.6/v1.0/scripts/filter_for_v12_pruned.py` | Qwen3.6 v1.0 tiered 8K/16K token filters |
| `qwen3.6/v1.0/scripts/preprocess_competitive_programming.py` | Qwen3.6 v1.0 competitive programming preprocessing |
| `qwen3.6/v1.0/scripts/preprocess_qwen3_coder_distill.py` | Qwen3.6 v1.0 Qwen3-Coder-480B distill preprocessing |

## 7. Training Monitoring

```bash
# Watch training log in real-time
tail -f /workspace/logs/*.log

# Check GPU memory
watch -n 1 'nvidia-smi'

# Training loss (grep from log)
grep -E "^\s*loss:" /workspace/logs/*.log | tail -n 50
```

## 8. HuggingFace Repos

- `danielcherubini/Qwen3.5-DeltaCoder-9B` — Qwen3.5 v1.0/v1.1 DPO adapter
- `danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF` — Qwen3.5 v1.0/v1.1 GGUF quantizations
- `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B` — 35B-A3B adapter (TODO: create when ready)
- `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B-GGUF` — 35B-A3B GGUFs (TODO: create when ready)
- `danielcherubini/Qwen3.6-DeltaCoder-9B` — Qwen3.6 v1.0 adapter (TODO: create when ready)
- `danielcherubini/Qwen3.6-DeltaCoder-9B-GGUF` — Qwen3.6 v1.0 GGUFs (TODO: create when ready)

## 9. Qwen3.6 v1.0 Structure

- `qwen3.6/v1.0/configs/` — Axolotl config placeholder (BLOCKED)
- `qwen3.6/v1.0/scripts/pretokenize_for_sft.py` — 32K context pretokenization for Qwen3.6
- `qwen3.6/v1.0/scripts/train_unsloth.py` — SFT training for Qwen3.6
- `qwen3.6/v1.0/scripts/provision.sh` — Vast.ai bootstrap for Qwen3.6
- `qwen3.6/v1.0/scripts/train_dpo.py` — DPO training for Qwen3.6
- `qwen3.6/v1.0/scripts/generate_dpo_pairs.py` — DPO pair generation
- `qwen3.6/v1.0/scripts/merge_and_export_dpo.py` — Merge + GGUF export

## 10. Quick Commands

```bash
# v1.1 DPO training (on-policy only)
python qwen3.5/v1.1/scripts/train_dpo.py --sft-model /workspace/merged_v1.1

# v1.1 DPO training (on-policy + 50K Ling-Coder-DPO)
python qwen3.5/v1.1/scripts/train_dpo.py --sft-model /workspace/merged_v1.1 --ling-coder 50000

# v1.1 pretokenize (32K context)
python qwen3.5/v1.1/scripts/pretokenize_for_sft.py --data qwen3.5/v1.1/data/v1.1_sft_train_pruned.jsonl --output qwen3.5/v1.1/data/v1.1_pretokenized.parquet

# v1.1 dry run
python qwen3.5/v1.1/scripts/train_unsloth.py --data /workspace/v1.1_pretokenized.parquet --max-steps 20

# Qwen3.6 v1.0 pretokenize (32K context)
python qwen3.6/v1.0/scripts/pretokenize_for_sft.py --data qwen3.6/v1.0/data/v1.0_sft_train_pruned.jsonl --output qwen3.6/v1.0/data/v1.0_pretokenized.parquet

# Qwen3.6 v1.0 dry run
python qwen3.6/v1.0/scripts/train_unsloth.py --data /workspace/v1.0_pretokenized.parquet --max-steps 20

# Merge + export to GGUF
python qwen3.5/v1.1/scripts/merge_and_export_dpo.py --sft-model /workspace/merged_v1.1 \
    --dpo-adapter ./outputs/deltacoder-9b-v1.1-dpo/lora_adapter \
    --merged-dir ./outputs/deltacoder-9b-v1.1-dpo-merged \
    --gguf-dir ./outputs/deltacoder-9b-v1.1-dpo-gguf \
    --filename-prefix DeltaCoder-9B-v1.1-DPO \
    --llama-cpp-dir /workspace/llama.cpp \
    --keep-merged --upload --hf-token $HF_TOKEN
```

## 11. Training Dataset & Strategy (Jackrong-Inspired, 2026-04-05)

### Core Philosophy: Quality > Quantity

After analyzing Jackrong's Qwopus3.5-9B-v3 (87.80% HumanEval vs our v1 regression to 50.6%),
we revised the entire training approach. Key changes:
- **`lora_alpha=64`** (1:1 ratio with r=64, was 0.5:1) — Jackrong-validated
- **`train_on_responses_only=True`** — mask user/system tokens, loss only on assistant responses
- **1 epoch** (157K rows is already 10x Jackrong's dataset size)
- **Tiered token limits** instead of uniform truncation

### New Dataset Mix (~157K rows, ~700M tokens)

**Tier 1 — ≤8K tokens (Coding + Tool Calling):**
| Source | Rows | Notes |
|--------|------|-------|
| nemotron_tool_calling | ~40,000 | Filtered by tool call count |
| competitive_programming | ~28,000 | NEW — Jackrong blend, 87.5% Nemotron Python competitive coding |
| nemotron_agentic | ~18,850 | All kept (99.1% naturally ≤8K) |
| xlam | ~15,000 | All kept |
| code_feedback | ~14,985 | Multi-turn ≥4 messages |
| qwen3_coder_distill | ~9,500 | NEW — distilled from Qwen3-Coder-480B via rStar-Coder |
| magicoder | ~5,000 | Top 5K by length |

**Tier 2 — ≤16K tokens (Agentic/SWE):**
| Source | Rows | Notes |
|--------|------|-------|
| opencoder_reasoning | ~16,025 | 64.1% of 25K survive 16K filter |
| swesmith | ~9,780 | 48.9% of 20K survive 16K filter |

**Dropped entirely:** `nemotron_swe` — 100% of rows exceed 16K (median 43K).

### New Jackrong Datasets
- **`Jackrong/Competitive-Programming-python-blend`**: ~28K rows, already in `messages` format
  with `<think>` blocks, apache-2.0/cc-by-4.0. Proved to boost HumanEval by +4.87pp.
- **`Jackrong/qwen3-coder-480b-distill-mini`**: 9,543 rows, distilled from Qwen3-Coder-480B.
  Uses `Input`/`code_output` format — converted by `preprocess_qwen3_coder_distill.py`.

### `train_on_responses_only` Implementation
```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```
Applied after `SFTTrainer(...)` creation, before `trainer.train()`.
Use `--no-response-only` flag on `train_unsloth.py` to disable for ablation.

### Cost Reduction
~700M tokens vs old 1.4B = ~half the training steps → **~$100-130** (was ~$200-260).

## 12. Key Discoveries & Constraints

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

### Dry Run Results (2026-04-02, validated)
20-step dry run on fresh H100 SXM 80GB (no volume, 80GB local disk):
- **Bootstrap**: provision.sh installs everything from scratch in ~4 min
- **Data loading**: 261,998 rows from 6 parquet shards, ~5s
- **Packing**: 262K rows -> 42,976 packed 32K sequences
- **Step time**: ~59s/step steady state (first step 166s due to JIT)
- **VRAM**: 60-63 GB / 80 GB (plenty of headroom)
- **Loss**: 1.101 (step 10) -> 0.522 (step 20), avg 0.811
- **Grad norm**: ~0.10 (healthy, no NaN)
- **Trainable params**: 173M / 9.6B (1.81%)
- **LoRA**: r=64, alpha=32, all GDN + attention + MLP targets

### Training Cost Analysis (149K pruned dataset)
Full training (1 epoch), 43,115 packed examples, batch_size=1, grad_accum=4:

**Single GPU:**
- 1x H100 SXM: 10,779 steps × 59s = ~177 hrs
- 1x A100 SXM: 10,779 steps × 84s = ~252 hrs

**Multi-GPU DDP (validated — see below):**
- 2x A100 SXM: 5,390 steps × 101s = ~151 hrs
- 2x H100 SXM: 5,390 steps × ~72s = ~108 hrs (estimated)
- 4x H100 SXM: 2,695 steps × ~72s = ~54 hrs (estimated)

batch_size must stay at 1 (GDN + packing limitation). Changing grad_accum changes optimizer
steps but not total forward/backward passes.

### DDP Multi-GPU Support (VALIDATED 2026-04-02)

**Unsloth DDP works with our FastVisionModel + packing + frozen vision setup.**

Tested on 2x A100 SXM4 80GB, 20 steps, using `torchrun --nproc_per_node=2`:
- **~101s/step** steady state (converged to 100.86s)
- **~65 GB VRAM per GPU**, 100% utilization on both
- **Loss: 0.4821** (step 20), avg 0.8716, grad_norm 0.0312 — healthy
- **No errors, no OOM, no NaN**
- Total batch size = batch_size × grad_accum × num_GPUs (1 × 4 × 2 = 8)
- Steps halved vs single GPU (data parallelism)

**Required settings for DDP:**
- `ddp_find_unused_parameters=False` in SFTConfig — frozen vision encoder creates unused params
- Use `torchrun --nproc_per_node=N` or `accelerate launch` to start training
- Each GPU needs full model in VRAM (~65GB) — DDP does NOT pool VRAM

**Known issues (do NOT affect our setup):**
- GitHub #4485: VLM DDP slow with actual vision data — we do text-only, no issue
- GitHub #4066: VLM DDP device mismatch with `device_map="balanced"` — we don't use device_map

### RTX PRO 6000 Blackwell Compatibility (VALIDATED 2026-04-02)
- **96GB GDDR7 VRAM** — uses only 59GB with Unsloth's smart gradient offloading (40% headroom)
- **81.7s/step** — faster than A100 SXM (84s), slower than H100 SXM (59s)
- **SM 12.0** compute capability — causal-conv1d compiles for it (compute_120)
- provision.sh auto-detects GPU SM arch, works on Blackwell
- **Must use `vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312-2026-03-26`** — CUDA 13.1 image causes mismatch. CUDA 12.9 toolkit supports SM 12.0.
- Vast.ai: $0.83/hr (Spain), RunPod: $1.64/hr
- Full training: 10,779 steps × 82s = ~245h × $0.83 = **~$204** (cheapest validated option)