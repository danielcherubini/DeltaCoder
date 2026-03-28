# DeltaCoder v1.2 — Broader SFT Plan

**Goal:** Retrain DeltaCoder from `Qwen/Qwen3.5-9B` base with a broader, higher-quality SFT dataset covering code instruction following, tool-calling, and chain-of-thought reasoning — then apply the existing DPO pipeline on top.

**Architecture:** Single SFT run on merged/shuffled dataset → same DPO pipeline as v1.1 → same GGUF export pipeline.

**Tech Stack:** Unsloth (SFT), HuggingFace + PEFT + TRL (DPO), llama.cpp (GGUF export), Vast.ai H100

---

## Dataset Mix

### Code Instruction Following (~60% of training)

| Dataset | Rows (capped) | Notes |
|---------|--------------|-------|
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 50K | High quality synthetic coding instructions |
| `togethercomputer/CoderForge-Preview` | 50K | Tool-call reliability (proven from v1) |
| `m-a-p/Code-Feedback` | 50K | Multi-turn code revision — teaches self-correction |

### Tool-Calling (~20% of training)

| Dataset | Rows (capped) | Notes |
|---------|--------------|-------|
| `NousResearch/hermes-function-calling-v1` | 13K | JSON function calling, all rows |
| `glaiveai/glaive-function-calling-v2` | 50K | Multi-turn tool use |

### Reasoning / Chain-of-Thought (~20% of training)

| Dataset | Rows (capped) | Notes |
|---------|--------------|-------|
| `nohurry/Opus-4.6-Reasoning-3000x-filtered` | 3K | Claude 4.6 Opus reasoning traces |
| `Jackrong/Qwen3.5-reasoning-700x` | 700 | Qwen3.5-specific reasoning patterns |

**Total: ~217K rows, ~500-700M tokens (est)**

---

## Training Config

### SFT (Stage 1)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | `Qwen/Qwen3.5-9B` | Clean start, not Jackrong distill |
| Method | LoRA (r=64, alpha=32) | Same as v1 |
| Target modules | All attention + GDN + MLP | Same as v1 — include GDN projections |
| Sequence length | 4096 | Increase to 8192 if VRAM allows |
| Batch size | 2 (effective 16 with grad_accum=8) | |
| Learning rate | 1e-4 (cosine) | |
| Epochs | 1 | |
| Precision | BF16 | No QLoRA |
| Framework | Unsloth | Faster SFT |
| Hardware | Vast.ai H100 80GB | ~12-15hrs, ~$20-25 |

### DPO (Stage 2)

Reuse existing pipeline from v1.1:
- `scripts/generate_dpo_pairs.py` — regenerate pairs using v1.2 SFT model as generator
- `scripts/train_dpo.py` — same config (beta=0.1, lr=5e-6)
- Dataset: `TIGER-Lab/AceCode-V2-122K` — 10K problems, 8 samples each

---

## Key Differences from v1

| | v1/v1.1 | v1.2 |
|---|---|---|
| Base model | Jackrong distill (reasoning pre-tuned) | `Qwen/Qwen3.5-9B` (clean base) |
| Dataset size | 50K (CoderForge only) | ~217K (multi-source) |
| Code diversity | Low (CoderForge focused) | High (Magicoder + Code-Feedback + Glaive) |
| Reasoning | Inherited from Jackrong | Explicitly included in SFT mix |
| Tool-calling | Strong (CoderForge) | Strong (CoderForge + Hermes + Glaive) |

---

## Preprocessing Steps

Each dataset needs to be converted to the Qwen3.5 chat template format before training. Write a preprocessing script for each:

### Task 1: Preprocess Magicoder-OSS-Instruct-75K
- Format: `instruction` → `response` (single-turn)
- Convert to: `[{"role": "user", "content": instruction}, {"role": "assistant", "content": response}]`
- Cap at 50K rows, filter empty responses

### Task 2: Preprocess Code-Feedback
- Format: multi-turn conversations
- Already in conversation format — verify schema and normalize
- Cap at 50K rows

### Task 3: Preprocess Hermes function-calling-v1
- Format: tool-call conversations
- Normalize to Qwen3.5 tool-call format
- Use all ~13K rows

### Task 4: Preprocess Glaive function-calling-v2
- Format: `system` + multi-turn with function calls
- Normalize to Qwen3.5 tool-call format
- Cap at 50K rows

### Task 5: Merge + shuffle all datasets
- Extend `scripts/merge_datasets.py`
- Weight by dataset (60% code, 20% tool, 20% reasoning)
- Final output: `data/v1.2_sft_train.jsonl`

---

## SFT Training Script Changes

- Update `scripts/train_unsloth.py` to point at merged dataset
- Add `--dataset` argument for flexibility
- Consider `dataset_num_proc=1` (Qwen3.5 multiprocessing issues)

---

## Estimated Timeline & Cost

| Step | Time | Cost |
|------|------|------|
| Dataset preprocessing (local) | 1-2hrs | $0 |
| SFT training (H100 80GB) | ~12-15hrs | ~$20-25 |
| DPO pair generation (H100, vLLM) | ~4-6hrs | ~$7-10 |
| DPO training (H100 80GB) | ~4hrs | ~$6-8 |
| GGUF export + upload | ~1hr | ~$1.50 |
| **Total** | **~22-28hrs** | **~$35-45** |

---

## Expected Improvements

- **HumanEval**: 50.6% → 55-60% (broader code training)
- **Tool-calling**: Maintained or improved (more diverse tool-call data)
- **Terminal-Bench**: 2-3/4 → 3-4/4 (self-correction from Code-Feedback)
- **Reasoning**: Maintained (explicit reasoning data in mix)

---

## Files to Create/Modify

- `scripts/preprocess_magicoder.py` — new
- `scripts/preprocess_code_feedback.py` — new
- `scripts/preprocess_hermes.py` — new
- `scripts/preprocess_glaive.py` — new
- `scripts/merge_datasets.py` — extend existing
- `scripts/train_unsloth.py` — update dataset path + args
- `configs/deltacoder-9b-lora-v1.2.yaml` — new config

---

## Open Questions

1. **Base model choice**: `Qwen/Qwen3.5-9B` vs keeping Jackrong distill as base?
   - Clean base = more control, but loses Jackrong's reasoning efficiency
   - Jackrong base = keep reasoning gains, just add code data
   - **Recommendation**: Start from `Qwen/Qwen3.5-9B` for a clean v1.2

2. **Sequence length**: 4096 vs 8192?
   - 8192 fits more complete code examples but halves effective batch size
   - **Recommendation**: 4096 for first run, increase if loss plateaus

3. **DPO regeneration**: Reuse v1.1 pairs or regenerate with v1.2 SFT model?
   - On-policy pairs (from v1.2 model) will be better
   - **Recommendation**: Regenerate — pairs from a better generator = better DPO signal
