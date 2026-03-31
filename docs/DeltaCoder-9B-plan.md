# DeltaCoder-9B

## Goal
Combine Jackrong's superior code reasoning (53.7% HumanEval) with OmniCoder's agentic tool-use ability (23.6% Terminal-Bench) into a single 9B model optimized for agentic coding in Claude Code, OpenCode, and Cline.

## Why
The Jackrong v2 base model is excellent at code reasoning but frequently produces **malformed JSON tool calls**. When the `arguments` field contains code strings (newlines, quotes, backslashes), the model fails to escape them properly, resulting in JSON parse errors:
```
JSON Parse error: Property name must be a string literal
```
The model tries to call the right tool with the right content, but the serialization breaks. This is especially bad for edit/replace tools where `oldString`/`newString` contain multi-line code blocks.

### Failure Example 1: Edit tool — unescaped code in arguments
```
tool=edit, error=Invalid input for tool edit: JSON parsing failed
{"filePath":"...router.rs","oldString":"pub fn build_router() -> Router {\n    Router::new()\n
.route(\n            \"/v1/chat/completions\",\n  ...
Error message: JSON Parse error: Property name must be a string literal
```
Multi-line Rust code with escaped quotes inside `oldString` breaks JSON parsing.

### Failure Example 2: Bash tool — truncated JSON, never closes object
```
tool=bash, error=Invalid input for tool bash: JSON parsing failed
{"command":"gh pr create --title 'fix: address CodeRabbit review round 10 findings' --body '## Summary\n\n...
- cargo build --workspace ✅\n- cargo test --workspace ✅ (75 passed)\n...','description"
Error message: JSON Parse error: Expected '}'
```
Long bash command with embedded markdown, emojis, and single quotes — model cuts off mid-key (`'description"`) without closing the JSON object.

### Common patterns
- Fails on **long string values** with special characters (newlines, quotes, emojis)
- Fails on **nested quoting** (single quotes inside JSON strings inside arguments)
- Sometimes **truncates** the JSON object before closing it
- The model's *intent* is correct — it picks the right tool and the right content — but the serialization is broken

**The fix**: SFT on ~238K correctly-formatted OpenAI JSON tool_call examples to teach the model reliable JSON serialization, especially for code-heavy arguments.

## Base Model
- **Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2**
- Architecture: Qwen3.5-9B hybrid (Gated Delta Networks + Gated Attention — NOT Mamba)
- Why: Best HumanEval score (53.7%) at 9B, Claude Opus reasoning distill, concise thinking chains

## Training Method
- **LoRA SFT** (r=64, alpha=32) — same config as OmniCoder
- **Framework**: HuggingFace PEFT + Axolotl (on cloud GPU via SSH)
- **Precision**: bf16
- **Optimizer**: AdamW (lr=2e-4, cosine schedule)

> **NOTE**: Unsloth does NOT support Qwen3.5's hybrid GDN architecture — its custom kernels
> only handle standard transformer attention/MLP. Use HuggingFace PEFT directly instead.
> PEFT's LoRA works on all `nn.Linear` layers including GDN projections.

### LoRA Target Modules (verified via dry run)
Architecture: 32 layers total — 24 GDN (`linear_attention`) + 8 Full Attention (`full_attention`)
Pattern: `[GDN, GDN, GDN, FullAttn]` × 8

- **Full Attention (8 layers)**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **GDN (24 layers)**: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- **MLP (all 32 layers)**: `gate_proj`, `up_proj`, `down_proj`

```python
target_modules = [
    # Full Attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # GDN
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    # MLP
    "gate_proj", "up_proj", "down_proj",
]
```

## Datasets (Open Source)

**Target format: OpenAI JSON tool_calls** — this is what local inference servers (llama.cpp, Ollama) expose, and what OpenCode/Pi use when talking to local models.

| Dataset | Rows | Size | Native Format | Conversion Needed |
|---------|------|------|---------------|-------------------|
| CoderForge-Preview | 413K | ~90GB | XML tags (`<function_calls><invoke>`) | Parse XML → OpenAI JSON |
| Nemotron-SWE-v1 | 51K | ~11GB | OpenAI JSON `tool_calls` | None (ready to use) |
| Nemotron-Agentic-v1 | 335K | ~5.5GB | OpenAI JSON `tool_calls` + `reasoning_content` | Strip `reasoning_content` field |
| SWE-agent-trajectories | 80K | ~5.3GB | Plain text commands, custom roles (`ai` not `assistant`) | Wrap commands as tool_calls, fix roles |

### Dataset Details

1. **CoderForge-Preview** (togethercomputer) — 413K trajectories, OpenHands scaffold
   - https://huggingface.co/datasets/togethercomputer/CoderForge-Preview
   - Tools: `execute_bash`, `str_replace_editor`, `think`, `finish`
   - Tool calls are XML inline in assistant content: `<function_calls><invoke name="execute_bash"><parameter name="command">...</parameter></invoke></function_calls>`
   - Has `reward` field (0.0/1.0) — filter to reward=1.0 for quality

2. **Nemotron-SWE-v1** (NVIDIA) — 51K trajectories, OpenHands scaffold
   - https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1
   - Already OpenAI JSON format with `tool` role messages
   - Tools: `execute_bash`, `str_replace_based_edit_tool`, `read_file`
   - Generated by Qwen3-Coder-480B

3. **Nemotron-Agentic-v1** (NVIDIA) — 335K trajectories (19K interactive_agent + 316K tool_calling)
   - https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1
   - Already OpenAI JSON format with `reasoning_content` field
   - Diverse tool definitions per row (not just coding tools)
   - CC-BY-4.0 license

4. **SWE-agent-trajectories** (nebius) — 80K trajectories
   - https://huggingface.co/datasets/nebius/SWE-agent-trajectories
   - Custom format: roles are `system/user/ai`, no structured tool calls
   - Agent writes `DISCUSSION` then a command in a code block
   - Has `target` field (bool) — filter to target=True (16.8%) for quality
   - Commands: `open`, `edit`, `create`, `search_dir`, `search_file`, `submit`, plus bash

## Data Preprocessing Pipeline

All datasets get normalized to a single JSONL format with OpenAI-style tool_calls:
```
scripts/
  preprocess_coderforge.py     # XML tags → OpenAI JSON tool_calls
  preprocess_nemotron_swe.py   # Pass-through (already correct format)
  preprocess_nemotron_agentic.py # Strip reasoning_content
  preprocess_sweagent.py       # Plain text → OpenAI JSON tool_calls, ai→assistant
  merge_datasets.py            # Combine all, shuffle, apply mixing ratios
```

### Quality Filters
- CoderForge: only `reward == 1.0` rows (~155K of 413K)
- SWE-agent: only `target == True` rows (~13K of 80K)
- Nemotron-SWE: use all (already curated)
- Nemotron-Agentic: use `interactive_agent` split (19K) — the `tool_calling` split (316K) is general-purpose, not coding-focused

### Estimated Training Mix
| Dataset | Filtered Rows | Weight |
|---------|--------------|--------|
| CoderForge (reward=1) | ~155K | Primary |
| Nemotron-SWE-v1 | ~51K | Primary |
| Nemotron-Agentic (interactive) | ~19K | Secondary |
| SWE-agent (target=True) | ~13K | Secondary |
| **Total** | **~238K** | |

## Cloud Training Setup

**Platform**: Vast.ai with "Axolotl - LLM Fine Tuning" Docker template (CUDA 12.6, 200GB container)

Template includes: Axolotl, PyTorch, Transformers, PEFT, bitsandbytes, flash-attention, DeepSpeed, wandb

### Steps
1. Rent A100 80GB on Vast.ai with the Axolotl template
2. SSH into instance
3. Install Qwen3.5 dependencies (not included in template):
   ```bash
   pip install flash-linear-attention==0.4.1
   pip uninstall causal-conv1d -y  # conflicts with flash-linear-attention
   ```
4. Upload `data/train.jsonl` and `configs/deltacoder-9b-lora.yaml`
5. Login to HuggingFace (for base model download):
   ```bash
   huggingface-cli login
   ```
6. Test run (verify loss decreases before committing to full training):
   ```bash
   accelerate launch -m axolotl.cli.train configs/deltacoder-9b-lora.yaml \
     --max_steps=50 --val_set_size=0.1
   ```
7. Full training:
   ```bash
   accelerate launch -m axolotl.cli.train configs/deltacoder-9b-lora.yaml
   ```
8. Merge LoRA + export GGUF:
   ```bash
   bash scripts/merge_and_export.sh
   ```
9. Download GGUFs locally, terminate instance

| Provider | GPU | Cost/hr | Est. Time | Total Cost |
|----------|-----|---------|-----------|------------|
| Vast.ai | A100 80GB | $0.67 | 3-5 hrs | **$2-4** |
| Vast.ai | H100 SXM | $1.67 | 1.5-3 hrs | $3-5 |

## Output
- Export as GGUF via llama.cpp (`convert_hf_to_gguf.py` + `llama-quantize`)
- Quants: Q4_K_S, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- Publish: `danielcherubini/DeltaCoder-9B-GGUF`
- llama.cpp has full Qwen3.5 GGUF support (merged Feb 2026, dedicated `GATED_DELTA_NET` op)
- Ollama also supports Qwen3.5 natively

## Target Benchmarks
| Benchmark | Jackrong v2 (base) | OmniCoder-9B | DeltaCoder-9B (target) |
|-----------|-------------------|--------------|----------------------|
| HumanEval (local) | 53.7% | 36.0% | >50% |
| Terminal-Bench 2.0 | — | 23.6% | >30% |
| SWE-Bench Verified | — | — | >25% |

## Evaluation Plan
1. Run local HumanEval via eval_humaneval.py
2. Run EvalPlus codegen + evaluate
3. If possible, run Terminal-Bench 2.0 via Docker
4. Compare against base Jackrong v2 and OmniCoder-9B

## Sampling Recommendations (Qwen3.5 official)
- Coding: temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
- Chat: temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5

## Known Quirks
- Qwen3.5-9B sometimes outputs tool calls as XML instead of JSON when thinking is enabled (llama.cpp #20837) — training on OpenAI JSON format should help
- Ollama is ~40-65% slower than raw llama.cpp for Qwen3.5 — prefer llama-server for benchmarking
- VRAM-constrained GPUs can slow down after long prompts (>10K tokens) due to SSM state checkpoints

## Training Issues & Fixes
- **PyArrow schema conflict on `tools` field**: Different datasets have different tool definitions (different parameter schemas), causing `TypeError: Couldn't cast array of type struct<...>` when HF datasets tries to infer a unified schema. **Fix**: Strip `tools` field from train.jsonl and remove `field_tools` from Axolotl config. The model still learns correct tool_call JSON format from the assistant messages — tool definitions are provided at inference time anyway. **Future improvement**: Normalize tool schemas across datasets so `tools` can be included in training (would teach the model to use tools it hasn't seen before).
- **Empty turn warnings**: Some messages have `null` content (e.g., assistant turns that only contain `tool_calls`). Axolotl warns "Content end boundary is the same as start boundary" — harmless, training continues fine.

## Status
- [x] Inspect base model layers — LoRA target names confirmed via dry run
- [x] Verify GGUF/llama.cpp/Ollama support — confirmed working
- [x] Verify chat template supports tool_calls — confirmed (4047 char template)
- [x] Verify preprocessing scripts — all parsers working
- [x] Verify VRAM fits A100 80GB — ~28-29GB estimated, fits easily
- [x] Run preprocessing scripts on full datasets — 238,590 rows, 14.7GB train.jsonl
- [x] Write Axolotl config with verified target_modules
- [x] Rent cloud GPU — Vast.ai A100 SXM4 80GB, then H200 140GB ($2.50/hr)
- [x] Set up training environment — Unsloth 2026.3.10 + Transformers 5.3.0
- [x] Upload training data — MD5 verified
- [x] Run full LoRA fine-tune — 3,125 steps, 10.14hrs on H200, final loss ~0.94
- [x] Export to GGUF — Q4_K_M (5.3GB), Q5_K_M (6.1GB), Q6_K (6.9GB), Q8_0 (8.9GB)
- [x] Publish to HuggingFace — danielcherubini/Qwen3.5-DeltaCoder-9B + GGUF
- [ ] Benchmark against baselines (HumanEval, Terminal-Bench, SWE-Bench)

## Training History
- **Attempt 1**: Axolotl on A100 — O(n^2) tokenization bug, pre-tokenized to work around it
- **Attempt 2**: Axolotl on H200 — too slow (74hr estimate, ~$185), killed
- **Attempt 3**: Unsloth on H200 with pre-tokenized data — multiple issues (SFTTrainer incompatible, bitsandbytes broken)
- **Attempt 4 (final)**: Unsloth on H200 with CoderForge from HF — 10.14hrs, ~$25, success
