# Qwen3.5-DeltaCoder-9B

> Reliable tool-calling for agentic coding — LoRA fine-tune of Qwen3.5-9B
> **v1.1-DPO in progress** — DPO alignment underway to improve code correctness

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-Qwen3.5--9B-purple)](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-GGUF-yellow)](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF)
[![LoRA](https://img.shields.io/badge/HuggingFace-LoRA-orange)](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B)

Small language models can reason about code, but they struggle to **call tools reliably**. DeltaCoder takes a strong reasoning base and teaches it to produce correctly-formatted JSON tool calls — the kind that coding agents like [OpenCode](https://github.com/opencode-ai/opencode), [Pi](https://github.com/badlogic/pi-mono), and [Cline](https://github.com/cline/cline) depend on.

## Downloads

| Format | Link | Size |
|--------|------|------|
| GGUF Q4_K_M | [HuggingFace](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF) | 5.3 GB |
| GGUF Q5_K_M | [HuggingFace](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF) | 6.1 GB |
| GGUF Q6_K | [HuggingFace](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF) | 6.9 GB |
| GGUF Q8_0 | [HuggingFace](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF) | 8.9 GB |
| LoRA adapter | [HuggingFace](https://huggingface.co/danielcherubini/Qwen3.5-DeltaCoder-9B) | 661 MB |

## The Problem

[Jackrong's Qwen3.5-9B reasoning distill](https://huggingface.co/Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2) scores **53.7% on HumanEval** — best-in-class at 9B. But when used as a coding agent, it frequently produces malformed JSON tool calls:

```
tool=edit, error=JSON Parse error: Property name must be a string literal
tool=bash, error=JSON Parse error: Expected '}'
```

The model picks the right tool and the right content, but breaks on JSON serialization — especially when arguments contain multi-line code, nested quotes, or special characters.

**DeltaCoder aims to fix this.**

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3.5-9B (hybrid GDN architecture) |
| Method | LoRA (r=64, alpha=32) |
| Dataset | [CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) `filtered_reward1` (50K subset) |
| Sequence length | 4096 |
| Effective batch size | 16 (batch=2 x grad_accum=8) |
| Learning rate | 1e-4 (cosine schedule) |
| Epochs | 1 |
| Optimizer | AdamW |
| Precision | BF16 |
| Hardware | NVIDIA H200 140GB (Vast.ai) |
| Training time | ~10 hours |
| Training cost | ~$25 |
| Framework | Unsloth 2026.3.10 + HuggingFace Transformers 5.3.0 |
| Final loss | ~0.94 |

### LoRA Target Modules

All major weight matrices are adapted across the hybrid architecture:

- **Full Attention** (8/32 layers): `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Gated Delta Net** (24/32 layers): `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- **MLP** (all 32 layers): `gate_proj`, `up_proj`, `down_proj`

## Usage

### Ollama

```bash
ollama create deltacoder -f Modelfile
```

### llama.cpp / ik_llama.cpp

```bash
./llama-server -m Qwen3.5-DeltaCoder-9B-Q5_K_M.gguf -ngl 999 -c 131072 -ctk f16 -ctv q4_0 -fa 1 --jinja
```

### With PEFT (Python)

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "danielcherubini/Qwen3.5-DeltaCoder-9B")
```

## Project Structure

```
configs/
  deltacoder-9b-lora.yaml        # Axolotl training configuration (legacy)
  deltacoder-9b-dpo.yaml         # DPO hyperparameter reference
scripts/
  train_unsloth.py                # Unsloth SFT training script (v1)
  train_dpo.py                    # DPO training script (v1.1)
  generate_dpo_pairs.py           # On-policy DPO pair generation (async)
  merge_and_export_dpo.py         # Merge LoRA + export GGUFs (v1.1)
  pretokenize.py                  # Pre-tokenization script
  preprocess_coderforge.py        # XML -> OpenAI JSON
  preprocess_nemotron_swe.py      # Schema normalization
  preprocess_nemotron_agentic.py  # Strip reasoning_content
  preprocess_sweagent.py          # Plain text -> OpenAI JSON
  merge_datasets.py               # Combine and shuffle all datasets
data/
  dpo_pairs.jsonl                 # Generated DPO pairs (not committed)
```

## Key Findings

Things discovered during development that may help others working with Qwen3.5:

> [!NOTE]
> **Unsloth now supports Qwen3.5** (as of 2026.3.10) with custom Triton kernels. It's significantly faster than Axolotl for this architecture.

> [!WARNING]
> **Do not use `flash_attention_2` with sample packing on Qwen3.5** — this causes training loss to go to 0. Use `attn_implementation: sdpa` instead.

- Qwen3.5 uses **Gated Delta Networks** (not Mamba) — the GDN layer names are `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- The tokenizer is a `Qwen3VLProcessor` — standard `DataCollatorForSeq2Seq` won't work (lacks `.pad()`), use a custom collator or Unsloth's native pipeline
- `bitsandbytes adamw_8bit` may fail on some CUDA setups — `adamw_torch` is a safe fallback
- `causal-conv1d` may not build if system CUDA and PyTorch CUDA versions mismatch — flash-linear-attention falls back to torch (slower but works)
- Unsloth skips sample packing for processor-based models — still fast enough on H200

## Recommended Sampling Settings

Validated through testing with [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [Kronk](https://github.com/danielcherubini/kronk) on an RTX 3080 10GB.

| Profile | temperature | top_k | top_p | min_p | presence_penalty |
|---------|-------------|-------|-------|-------|-----------------|
| **Coding** | 0.6 | 20 | 0.95 | 0.0 | 0.0 |
| **Chat** | 1.0 | 20 | 0.95 | 0.0 | 1.5 |

> [!WARNING]
> **Do not use temperature below 0.5** — low temperatures (e.g., 0.3) cause deterministic looping in multi-turn agentic use, where the model repeats the same tool call indefinitely.

### KV Cache Quantization

For VRAM-constrained GPUs, use quantized KV cache keys/values:

| Context Length | KV Cache | VRAM (Q4_K_M) | Generation Speed |
|---------------|----------|---------------|-----------------|
| 102,400 | f16/q4_0 | ~8.5 GB | ~111 tok/s |
| 131,072 | f16/q4_0 | ~9.1 GB | ~110 tok/s |

```bash
# llama.cpp / ik_llama.cpp flags
-ctk f16 -ctv q4_0
```

## Benchmarks

Evaluated using [EvalPlus](https://github.com/evalplus/evalplus) against the Q4_K_M GGUF via ik_llama.cpp.

| Model | HumanEval | HumanEval+ |
|-------|-----------|------------|
| Jackrong v2 (base) | 53.7% | — |
| OmniCoder-9B | 36.0% | — |
| **DeltaCoder-9B** (temp=0.6) | **50.6%** | **49.4%** |
| DeltaCoder-9B (greedy) | 43.9% | 42.1% |

DeltaCoder retains most of the base model's code reasoning ability while adding reliable tool-call JSON formatting. Use the recommended sampling settings (temp=0.6) — greedy decoding significantly underperforms.

## v1.1-DPO (In Progress)

DeltaCoder v1 scored **2/4 (50%) on [Terminal-Bench](https://github.com/terminal-bench/terminal-bench) easy tasks**. The two failures were:
- `overfull-hbox`: hallucinated a word not in the allowed synonym list
- `cobol-modernization`: bytes/int type error in generated code — didn't catch its own bug

v1.1 applies **Direct Preference Optimization (DPO)** to improve code correctness and self-verification:

| Step | Status |
|------|--------|
| Generate DPO pairs from AceCode-V2-122K (10K problems, 8 samples each) | In progress |
| DPO training on Vast.ai H100 | Pending |
| GGUF export (Q4_K_M, Q5_K_M, Q6_K, Q8_0) | Pending |
| HumanEval + Terminal-Bench evaluation | Pending |
| HuggingFace release | Pending |

### v1.1 Training Plan

| Parameter | Value |
|-----------|-------|
| Method | DPO (Direct Preference Optimization) |
| Dataset | On-policy pairs from [AceCode-V2-122K](https://huggingface.co/datasets/TIGER-Lab/AceCode-V2-122K) |
| Pair generation | 8 samples/problem, keep if ≥1 pass AND ≥1 fail |
| Beta | 0.1 |
| Loss type | sigmoid |
| Hardware | Vast.ai H100 80GB |
| Framework | Unsloth + TRL ≥0.19.0 |

## Status

- [x] Dataset research and format analysis
- [x] Preprocessing scripts (238K rows from 4 datasets)
- [x] Dry-run validation (layer names, chat template, VRAM)
- [x] Training config (Axolotl -> Unsloth migration)
- [x] Full LoRA fine-tune (50K CoderForge examples, H200, ~10hrs)
- [x] GGUF export (Q4_K_M, Q5_K_M, Q6_K, Q8_0)
- [x] HuggingFace release
- [x] Benchmarking (HumanEval via EvalPlus)
- [x] Terminal-Bench evaluation (2/4 easy tasks, 50%)
- [ ] DPO pair generation (in progress)
- [ ] DPO fine-tune (v1.1)
- [ ] v1.1 GGUF export
- [ ] v1.1 HumanEval + Terminal-Bench evaluation

## Acknowledgements

- [Unsloth](https://unsloth.ai) for Qwen3.5 training support
- [Together AI](https://together.ai) for the CoderForge dataset
- [Jackrong](https://huggingface.co/Jackrong) for the base model
- [NVIDIA](https://nvidia.com) for Nemotron datasets
- [Nebius](https://nebius.com) for SWE-agent trajectories
