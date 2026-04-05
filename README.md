# Qwen3.5-DeltaCoder-9B

> Reliable tool-calling for agentic coding — LoRA fine-tune of Qwen3.5-9B
> **qwen3.5/v1.1 in progress** — retrained from clean base with 157K coding-focused dataset

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-Qwen3.5--9B-purple)](https://huggingface.co/Qwen/Qwen3.5-9B)
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

## Version History

### v1.2 (In Progress)

Complete retrain from clean `Qwen/Qwen3.5-9B` base with a curated, 100% coding-focused dataset:

| Dataset | Rows | Purpose |
|---------|------|---------|
| [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | 65K | Code reasoning with `<think>` traces (R1-generated) |
| [Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | 50K | Single-turn code generation |
| [CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview) | 50K | SWE-agent tool-use trajectories |
| [Code-Feedback](https://huggingface.co/datasets/m-a-p/Code-Feedback) | 50K | Multi-turn code revision |
| [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | 15K | Verified function calling |
| **Total** | **230K** | |

**Key improvements over v1:**
- Clean `Qwen/Qwen3.5-9B` base (not a third-party distill)
- 230K rows (vs 50K) — 4.6x more training data
- 100% coding-focused (v1 had generic function calling waste)
- R1 reasoning traces for chain-of-thought coding
- Execution-verified function calling (xlam)
- Axolotl with sample packing at 8192 seq length

| Step | Status |
|------|--------|
| Dataset preprocessing (5 datasets → 230K rows) | ✅ Done |
| SFT training (Axolotl, H200 141GB, batch=1+packing) | ✅ Done (~46hrs, ~$87) |
| On-policy DPO pair generation (4,521 pairs, 45.2% keep) | ✅ Done |
| DPO training (H200, TRL 1.0 + PEFT) | 🔄 Running |
| Merge + GGUF export (local, CPU) | ⏳ Pending |
| HuggingFace upload | ⏳ Pending |
| Terminal-Bench + HumanEval evaluation | ⏳ Pending |

### v1.1-DPO

DPO alignment on top of v1 to improve code correctness and self-verification.

- **DPO pairs:** 4,519 from AceCode-V2-122K (on-policy, 45% keep rate)
- **Training:** 3.7hrs on H100, final loss 0.538
- **Terminal-Bench:** 2/4 easy tasks (50%) — same score as v1 but different failure mode (timeouts from self-correction attempts vs immediate failures)
- **Status:** ✅ Released

### v1

Initial LoRA fine-tune on 50K CoderForge-Preview trajectories.

- **Base:** Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2
- **HumanEval:** 50.6% (temp=0.6)
- **Terminal-Bench:** 2/4 easy tasks (50%)
- **Status:** ✅ Released (superseded by v1.1-DPO)

## Training Details (v1.2)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3.5-9B (hybrid GDN architecture) |
| Method | LoRA (r=64, alpha=32) |
| Framework | Axolotl 0.15.0 (sample packing) |
| Dataset | 230K rows (5 coding datasets) |
| Sequence length | 8192 |
| Sample packing | true |
| Micro batch size | 1 (GDN limitation — cannot go higher with packing) |
| Gradient accumulation | 4 |
| Learning rate | 1e-4 (cosine schedule) |
| Epochs | 1 |
| Precision | BF16 |
| Attention | SDPA (FA2 causes CUDA errors with GDN) |
| group_by_length | true (reduces padding waste) |
| Hardware | NVIDIA H200 141GB (Vast.ai, France) |
| Estimated training time | ~46 hours |
| Estimated cost | ~$87 |

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
    "Qwen/Qwen3.5-9B",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, "danielcherubini/Qwen3.5-DeltaCoder-9B")
```

## Project Structure

```
qwen3.5/
  v1.0/                              # Original 9B (SFT + DPO, Axolotl configs)
    configs/                         # Axolotl configs (SFT + DPO)
    scripts/                         # train_unsloth, merge_and_export, etc.
    data/                            # DPO pairs (gitignored)
    outputs/                         # DPO adapter (gitignored)
    logs/                            # training logs (gitignored)
  v1.1/                              # Revised 9B (Jackrong-inspired, Unsloth + packing at 32K)
    configs/
      deltacoder-9b-lora-v1.1.yaml   # Axolotl SFT config
    scripts/
      train_dpo.py                   # DPO training (HF+PEFT+TRL, no Unsloth)
      generate_dpo_pairs.py          # On-policy DPO pair generation (async)
      merge_and_export_dpo.py        # LoRA merge + GGUF export
      pretokenize_for_sft.py         # Pre-tokenization (32768 context)
      preprocess_*.py                # Dataset preprocessing scripts
      merge_datasets.py              # Combine and shuffle all datasets
    data/                            # SFT training data + preprocessed datasets (gitignored)
    merged/                          # Merged SFT model (gitignored)
    lora_adapter/                    # SFT LoRA adapter (gitignored)
  35b-a3b/                           # MoE fine-tune (NEW — scripts pending)
    scripts/
qwen3.6/
  v1.0/                              # Qwen3.6 (BLOCKED — waiting for open weights)
    configs/
    scripts/
    data/                            # Training data (gitignored)
docs/
  plans/                             # Implementation plans
```

## Key Findings

Things discovered during development that may help others working with Qwen3.5:

> [!WARNING]
> **Qwen3.5 GDN does not support `micro_batch_size > 1` with sample packing.** The Gated Delta Net layers use `cu_seqlens` which requires batch dimension = 1. This is confirmed by [NVIDIA/Megatron-LM #3798](https://github.com/nvidia/megatron-lm/issues/3798), [axolotl #3453](https://github.com/axolotl-ai-cloud/axolotl/issues/3453), and [ms-swift docs](https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_5-Best-Practice.html). Use `group_by_length: true` to reduce padding waste.

> [!WARNING]
> **Do not use `flash_attention_2` with Qwen3.5 GDN layers** — causes `cudaErrorIllegalAddress`. Use `attn_implementation: sdpa` instead. Sample packing with SDPA works at batch=1.

> [!WARNING]
> **Do not use Unsloth for large-scale SFT on Qwen3.5.** Unsloth's VLM collator doesn't support sample packing. Without packing, training 230K rows takes 132+ hours. Axolotl with `sample_packing: true` reduces this to ~46 hours.

> [!WARNING]
> **Do not use Unsloth DPOTrainer with Qwen3.5** — it detects the model as a VLM and crashes with `KeyError: 'images'`. Use plain HuggingFace + PEFT + TRL instead.

- Qwen3.5 uses **Gated Delta Networks** (not Mamba) — the GDN layer names are `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- **All Qwen3.5 models are VLMs** (`Qwen3_5ForConditionalGeneration`) — this causes compatibility issues with many training frameworks
- Vast.ai scrubs inline `HF_TOKEN` env vars — must `export` separately
- Vast.ai HF dataset cache (`/workspace/.hf_home/datasets`) can silently grow to 60GB+ and fill the disk — clean it after preprocessing
- Use Vast.ai **shared drives** to persist datasets across instance swaps — avoids re-uploading 3.6GB+ each time
- `dataset_num_proc` must be `1` for Qwen3.5 tokenizer (crashes with multiprocessing)
- nvidia/OpenCodeReasoning R1 traces are 8K-32K tokens — NVIDIA trained at 32K context. Truncating to 4096 wastes most of the reasoning.
- For the Axolotl template on Vast.ai, the telemetry whitelist file may be missing — fix with `echo 'organizations: []' > .../axolotl/telemetry/whitelist.yaml`

## Recommended Sampling Settings

| Parameter | Value |
|-----------|-------|
| temperature | 0.6 |
| top_k | 20 |
| top_p | 0.95 |
| min_p | 0.0 |
| presence_penalty | 0.0 |
| repeat_penalty | 1.0 |

> [!WARNING]
> **Do not use temperature below 0.5** — low temperatures cause deterministic looping in multi-turn agentic use.

### KV Cache Quantization

For VRAM-constrained GPUs:

| Context Length | KV Cache | VRAM (Q4_K_M) | Generation Speed |
|---------------|----------|---------------|-----------------|
| 102,400 | f16/q4_0 | ~8.5 GB | ~111 tok/s |
| 131,072 | f16/q4_0 | ~9.1 GB | ~110 tok/s |

## Benchmarks

### v1 (current release)

| Model | HumanEval | HumanEval+ |
|-------|-----------|------------|
| Jackrong v2 (base) | 53.7% | — |
| OmniCoder-9B | 36.0% | — |
| **DeltaCoder-9B v1** (temp=0.6) | **50.6%** | **49.4%** |

### Terminal-Bench (easy tasks)

| Model | Score |
|-------|-------|
| DeltaCoder v1 | 2/4 (50%) |
| DeltaCoder v1.1-DPO | 2/4 (50%) — different failure mode |
| DeltaCoder Qwen3.5 v1.1 | TBD |

## Acknowledgements

- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) for sample packing and efficient training
- [Together AI](https://together.ai) for the CoderForge dataset
- [NVIDIA](https://nvidia.com) for OpenCodeReasoning and Nemotron datasets
- [Salesforce](https://salesforce.com) for xlam function-calling dataset
- [Qwen](https://huggingface.co/Qwen) for the Qwen3.5-9B base model
- [Unsloth](https://unsloth.ai) for Qwen3.5 support (used in v1 and DPO)
