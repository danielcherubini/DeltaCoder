# DeltaCoder-9B

> Reliable tool-calling for agentic coding — LoRA fine-tune of Qwen3.5-9B

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-Qwen3.5--9B-purple)](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-DeltaCoder--9B-yellow)](https://huggingface.co/danielcherubini/DeltaCoder-9B-GGUF)

Small language models can reason about code, but they struggle to **call tools reliably**. DeltaCoder takes a strong reasoning base and teaches it to produce correctly-formatted JSON tool calls — the kind that coding agents like [OpenCode](https://github.com/opencode-ai/opencode), [Pi](https://github.com/badlogic/pi-mono), and [Cline](https://github.com/cline/cline) depend on.

## The Problem

[Jackrong's Qwen3.5-9B reasoning distill](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2) scores **53.7% on HumanEval** — best-in-class at 9B. But when used as a coding agent, it frequently produces malformed JSON tool calls:

```
tool=edit, error=JSON Parse error: Property name must be a string literal
tool=bash, error=JSON Parse error: Expected '}'
```

The model picks the right tool and the right content, but breaks on JSON serialization — especially when arguments contain multi-line code, nested quotes, or special characters. [OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B) solves the agentic side (23.6% Terminal-Bench) but sacrifices reasoning (36% HumanEval).

**DeltaCoder aims to do both.**

## Target Benchmarks

| Benchmark | Jackrong v2 (base) | OmniCoder-9B | DeltaCoder-9B (target) |
|---|---|---|---|
| HumanEval | **53.7%** | 36.0% | >50% |
| Terminal-Bench 2.0 | — | **23.6%** | >30% |
| SWE-Bench Verified | — | — | >25% |

## Approach

**LoRA SFT** on ~238K tool-calling trajectories, normalized to OpenAI JSON `tool_calls` format.

### Base Model

- **Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2**
- Architecture: Qwen3.5-9B hybrid (Gated Delta Networks + Gated Attention)
- Claude Opus reasoning distillation with concise thinking chains

### Training Data

Four open-source datasets, all converted to OpenAI JSON format:

| Dataset | Filtered Rows | Source |
|---|---|---|
| [CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) | ~155K | togethercomputer |
| [Nemotron-SWE-v1](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1) | ~51K | NVIDIA |
| [Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1) | ~19K | NVIDIA |
| [SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) | ~13K | Nebius |

> [!NOTE]
> Each dataset uses a different tool-call format (XML, JSON, plain text). The preprocessing scripts normalize everything to OpenAI `tool_calls` JSON — the format that local inference servers (llama.cpp, Ollama) expose.

### Training Config

- **Framework**: HuggingFace PEFT + Axolotl
- **Method**: LoRA (r=64, alpha=32) targeting all GDN, attention, and MLP layers
- **Hardware**: A100 80GB (~3-5 hours, ~$2-4 on Vast.ai)
- **Precision**: bf16 with gradient checkpointing

## Project Structure

```
configs/
  deltacoder-9b-lora.yaml        # Axolotl training configuration
scripts/
  dry_run.py                      # Pre-flight validation (no GPU needed)
  preprocess_coderforge.py        # XML → OpenAI JSON
  preprocess_nemotron_swe.py      # Schema normalization
  preprocess_nemotron_agentic.py  # Strip reasoning_content
  preprocess_sweagent.py          # Plain text → OpenAI JSON
  merge_datasets.py               # Combine and shuffle all datasets
  merge_and_export.sh             # LoRA merge + GGUF quantization
```

## Getting Started

### 1. Validate before spending on GPU

```bash
pip install transformers peft datasets
python scripts/dry_run.py
```

This checks model architecture, LoRA target modules, chat template support, preprocessing correctness, and VRAM estimates — all on CPU.

### 2. Preprocess datasets

```bash
pip install datasets
python scripts/preprocess_coderforge.py
python scripts/preprocess_nemotron_swe.py
python scripts/preprocess_nemotron_agentic.py
python scripts/preprocess_sweagent.py
python scripts/merge_datasets.py
```

### 3. Train on cloud GPU

```bash
pip install axolotl peft transformers accelerate datasets
pip install flash-linear-attention==0.4.1

# Quick test (verify loss decreases)
accelerate launch -m axolotl.cli.train configs/deltacoder-9b-lora.yaml \
  --max_steps=50 --val_set_size=0.1

# Full training
accelerate launch -m axolotl.cli.train configs/deltacoder-9b-lora.yaml
```

### 4. Export to GGUF

```bash
bash scripts/merge_and_export.sh
```

Produces Q4_K_S, Q4_K_M, Q5_K_M, Q6_K, and Q8_0 quantizations.

## Key Findings

A few things discovered during development that may help others working with Qwen3.5:

> [!IMPORTANT]
> **Unsloth does not support Qwen3.5.** Its custom kernels only handle standard transformer attention/MLP. Use HuggingFace PEFT instead.

> [!WARNING]
> **Do not use `flash_attention_2` with sample packing on Qwen3.5** — this causes training loss to go to 0 ([axolotl#3453](https://github.com/axolotl-ai-cloud/axolotl/issues/3453)). Use `attn_implementation: sdpa` instead.

- Qwen3.5 uses **Gated Delta Networks** (not Mamba) — requires `flash-linear-attention` for sample packing
- LoRA kernel optimizations must be disabled (`lora_*_kernel: false`) to avoid assertion errors
- The GDN layer names are `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` — not the commonly assumed `in_proj`, `x_proj`, `dt_proj`

## Status

- [x] Dataset research and format analysis
- [x] Preprocessing scripts
- [x] Dry-run validation (layer names, chat template, VRAM)
- [x] Axolotl training config
- [ ] Full dataset preprocessing
- [ ] LoRA fine-tune
- [ ] GGUF export and quantization
- [ ] Benchmarking
- [ ] HuggingFace release
