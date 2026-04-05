# DeltaCoder 35B-A3B — Qwen3.5-35B-A3B SFT Plan

**Goal:** Fine-tune Qwen3.5-35B-A3B (MoE, 35B total / 3B active params) with a coding-heavy ~5K row dataset using the DeltaCoder Jackrong-inspired approach. Produce a merged bf16 model + GGUF quantizations.

**Architecture:** Use Unsloth's `FastLanguageModel` with bf16 LoRA (no QLoRA — not viable for MoE). Load the full MoE VLM model, apply LoRA to attention + MoE expert layers (GDN modules omitted by default — untested with MoE, opt-in via flag). Train with `train_on_responses_only=True`. Save merged 16bit, convert to GGUF via llama.cpp.

**Tech Stack:** Unsloth (FastLanguageModel), TRL SFTTrainer, flash-linear-attention + causal-conv1d (GDN acceleration), llama.cpp (GGUF conversion), Vast.ai H100 SXM 80GB.

**Note on `FastModel` vs `FastLanguageModel`:** Unsloth's newer docs reference `FastModel` as the unified API. The official MoE Colab notebook still uses `FastLanguageModel`. Both should work (likely aliased). We use `FastLanguageModel` to match the validated notebook.

**Status:** READY TO IMPLEMENT — Jackrong validated this exact approach on the same model ([Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled](https://huggingface.co/Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled)).

---

## Architecture Notes

### Qwen3.5-35B-A3B Key Facts
- **Model class:** `Qwen3_5MoeForConditionalGeneration` (VLM + MoE)
- **Layers:** 40 total (30 GDN + 10 full attention, same 3:1 ratio as 9B)
- **MoE:** 256 experts, 8 routed + 1 shared per token, intermediate_size=512 per expert
- **GDN layers:** YES — same Gated DeltaNet as 9B. Requires flash-linear-attention + causal-conv1d.
- **Hidden size:** 2048
- **Vocab:** 248,320
- **VRAM:** ~74GB at 2K context (Unsloth docs), likely ~78-80GB at 8K with gradient checkpointing. **Very tight** on H100 80GB — Jackrong confirmed it works. Fallback: reduce to 4096 context.
- **QLoRA:** NOT viable — Unsloth explicitly warns against it for MoE
- **Disk:** ~250GB needed for merge+GGUF pipeline (model cache 70GB + merged 70GB + GGUF 70GB)

### Key Differences from 9B Dense Pipeline
| Aspect | 9B (qwen3.5/v1.1) | 35B-A3B |
|--------|-----------|---------|
| Loader | `FastVisionModel` | `FastLanguageModel` |
| VLM packing hack | Required | Not needed — `FastLanguageModel` doesn't check `is_vlm` |
| LoRA targets | attention + GDN + MLP | attention + GDN + MLP + `gate_up_proj` (MoE experts) |
| `fast_inference` | N/A | Must be `False` (not supported for MoE yet) |
| Packing | Yes (VLM hack) | **Not used** (Jackrong didn't, Unsloth notebook doesn't, unclear if stable) |
| GGUF export | Manual config wrap | `save_pretrained_merged` + manual `convert_hf_to_gguf.py` |
| Trainable params | 173M / 9.6B (1.81%) | ~465M / 35.5B (~1.31%) at r=64 |

### LoRA Target Modules (MoE-aware)

**Default — matches official Unsloth MoE notebook (validated):**
```python
LORA_TARGET_MODULES = [
    # Attention (10 full + 30 GDN layers — attention projections only)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # MoE expert layers (all 40 layers) — Unsloth auto-detects and maps to
    # mlp.experts.gate_up_proj and mlp.experts.down_proj
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
]
```

**Optional GDN modules (untested with MoE — use `--include-gdn-modules` flag):**
```python
LORA_GDN_MODULES = [
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]
```

**Why GDN modules are opt-in:** The 35B-A3B has 30/40 GDN layers (same ratio as 9B), but the
official Unsloth MoE notebook does NOT include GDN-specific targets. Adding them may conflict
with Unsloth's MoE auto-detection (e.g., `out_proj` is ambiguous — exists in both attention and
GDN layers). Jackrong's validated 35B-A3B model (107 likes, 19K downloads) did NOT use GDN targets.

Unsloth auto-detects MoE and maps `gate_proj`/`up_proj`/`down_proj`/`gate_up_proj` to
`mlp.experts.gate_up_proj` and `mlp.experts.down_proj` internally.

---

## Dataset: Coding-Heavy ~5K Row Subset

Curated from our existing preprocessed v1.1 data sources:

| Source | Rows | Notes |
|--------|------|-------|
| competitive_programming | 2,000 | Jackrong blend with `<think>` blocks, HumanEval boost |
| qwen3_coder_distill | 1,500 | Distilled from Qwen3-Coder-480B via rStar-Coder |
| magicoder | 1,000 | Top by length (hardest problems) |
| code_feedback | 500 | Multi-turn ≥4 messages |
| **Total** | **5,000** | |

All rows pre-filtered to ≤8K tokens. Formatted with Qwen3.5 chat template,
`train_on_responses_only=True`.

---

## Training Config

| Parameter | Value | Notes |
|-----------|-------|-------|
| `base_model` | `Qwen/Qwen3.5-35B-A3B` | or `unsloth/Qwen3.5-35B-A3B` |
| `loader` | `FastLanguageModel` | NOT FastVisionModel |
| `load_in_4bit` | `False` | QLoRA not viable for MoE |
| `load_in_16bit` | Not passed | Unsloth auto-detects bf16 when `load_in_4bit=False` (notebook doesn't pass this) |
| `fast_inference` | `False` | Not supported for MoE |
| `max_seq_length` | 8192 | Context window |
| `lora_r` | 64 | Same as 9B |
| `lora_alpha` | 64 | 1:1 ratio (Jackrong-validated) |
| `use_gradient_checkpointing` | `"unsloth"` | Required for VRAM |
| `per_device_train_batch_size` | 1 | MoE + VRAM constraint |
| `gradient_accumulation_steps` | 4 | Effective batch = 4 |
| `num_train_epochs` | 2 | Small dataset → 2 epochs (Jackrong style) |
| `learning_rate` | 2e-5 | Long training rate (not 2e-4 demo rate) |
| `lr_scheduler_type` | `"linear"` | Jackrong used linear |
| `warmup_ratio` | 0.05 | |
| `optim` | `"adamw_8bit"` | |
| `weight_decay` | 0.001 | |
| `packing` | `False` | Not validated for MoE |
| `train_on_responses_only` | `True` | Same as 9B |
| `dataset_num_proc` | 1 | Qwen3.5 tokenizer constraint |
| `ddp_find_unused_parameters` | `False` | Vision encoder frozen |

### Estimated Training Time
- 5,000 rows × 2 epochs = 10,000 samples
- batch_size=1, grad_accum=4 → 2,500 optimizer steps
- ~15-20s/step at 8K context on H100 SXM
- **~10-14 hours total**
- At ~$2/hr for H100 SXM → **~$20-28**

---

## GGUF Export Strategy

Unsloth's `save_pretrained_gguf` is broken for MoE (bug #4294). Manual approach:

1. `model.save_pretrained_merged("merged_dir", tokenizer, save_method="merged_16bit")`
2. Build llama.cpp on the instance: `git clone https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j`
3. Convert: `python3 llama.cpp/convert_hf_to_gguf.py merged_dir --outfile model.gguf --outtype bf16`
4. Quantize: `llama.cpp/build/bin/llama-quantize model.gguf model-Q4_K_M.gguf Q4_K_M`
5. Upload to `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B-GGUF`

Note: MoE GGUF files are large even quantized (~20GB for Q4_K_M). The model needs ~20GB RAM to run at Q4.

---

## Tasks

### Task 1: Create `qwen3.5/35b-a3b/scripts/build_35b_subset.py` — Curate 5K Row Training Subset

**Context:**
We need a script that samples our existing preprocessed `*_converted.jsonl` files to create a small, coding-focused 5K row dataset for the 35B-A3B fine-tune. This script reads from the same data sources as `build_training_mix.py` but targets specific sources and row counts. The output is a single shuffled JSONL file.

**Files:**
- Create: `qwen3.5/35b-a3b/scripts/build_35b_subset.py`

**What to implement:**
A script that:
- Takes `--data-dir` (default: `qwen3.5/v1.1/data/v1.1_pruned`) and `--output` (default: `qwen3.5/35b-a3b/data/v1.1_35b_sft.jsonl`)
- Loads from these filtered files (already 8K token-filtered by `filter_for_v12_pruned.py`):
  - `competitive_programming_converted.jsonl` → sample 2,000 rows
  - `qwen3_coder_distill_converted.jsonl` → sample 1,500 rows
  - `magicoder_filtered.jsonl` → sample 1,000 rows
  - `code_feedback_filtered.jsonl` → sample 500 rows
- Falls back to `qwen3.5/v1.1/data/` (unfiltered) if filtered files don't exist
- Validates each row has `messages` with at least one assistant turn
- Shuffles with seed=42
- Writes single JSONL output
- Prints summary with per-source counts and total

The structure should follow the same patterns as `build_training_mix.py` — `load_jsonl()`, `validate_row()`, argparse CLI, seed-based shuffle.

**Steps:**
- [ ] Implement `qwen3.5/35b-a3b/scripts/build_35b_subset.py` following the spec above
- [ ] Run `python qwen3.5/35b-a3b/scripts/build_35b_subset.py --dry-run` to verify it finds files (or reports MISSING gracefully)
- [ ] Commit with message: `Add build_35b_subset.py: curate 5K coding-heavy dataset for 35B-A3B fine-tune`

**Acceptance criteria:**
- [ ] Script runs with `--dry-run` and reports expected source files + row counts
- [ ] Output JSONL has exactly 5,000 rows when all sources are available
- [ ] Each row has `messages` and `source` fields
- [ ] Sources are: competitive_programming (2K), qwen3_coder_distill (1.5K), magicoder (1K), code_feedback (500)

---

### Task 2: Create `qwen3.5/35b-a3b/scripts/train_35b_a3b.py` — SFT Training Script for MoE Model

**Context:**
This is the main training script for DeltaCoder 35B-A3B. It's modeled on `qwen3.5/v1.1/scripts/train_unsloth.py` (the 9B script) but adapted for the MoE architecture. The key differences are: (1) use `FastLanguageModel` instead of `FastVisionModel`, (2) include `gate_up_proj` in LoRA targets, (3) disable packing, (4) set `fast_inference=False`, (5) set default context to 8192 instead of 32768, (6) no VLM packing hack needed.

The script loads raw JSONL data (not pre-tokenized parquet — dataset is only 5K rows so tokenization is instant), applies the Qwen3.5 chat template, and trains with `train_on_responses_only=True`.

**Files:**
- Create: `qwen3.5/35b-a3b/scripts/train_35b_a3b.py`

**What to implement:**
A training script with these specifics:

```python
BASE_MODEL = "unsloth/Qwen3.5-35B-A3B"  # or "Qwen/Qwen3.5-35B-A3B"
MAX_SEQ_LENGTH = 8192
LORA_R = 64
LORA_ALPHA = 64
OUTPUT_DIR = "./outputs/deltacoder-35b-a3b-v1.1"

# Default: matches official Unsloth MoE notebook (validated)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",           # Attention
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",  # MoE experts
]

# Optional GDN modules — add with --include-gdn-modules flag (untested with MoE)
LORA_GDN_MODULES = [
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
]
```

Model loading:
```python
from unsloth import FastLanguageModel

model, processor = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=args.max_seq_length,
    load_in_4bit=False,         # QLoRA NOT viable for MoE; auto-selects bf16 LoRA
    fast_inference=False,       # Not supported for MoE
)
tokenizer = processor.tokenizer  # Extract tokenizer from processor

# NOTE: Do NOT pass load_in_16bit=True — the official notebook doesn't pass it,
# and Unsloth auto-detects bf16 when load_in_4bit=False.
```

LoRA application:
```python
targets = LORA_TARGET_MODULES
if args.include_gdn_modules:
    targets = LORA_TARGET_MODULES + LORA_GDN_MODULES

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_r,
    target_modules=targets,
    lora_alpha=args.lora_alpha,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # "unsloth" saves more VRAM than True (needed for 8K on 80GB)
    random_state=42,
)
# NOTE: Official notebook uses use_gradient_checkpointing=True. We use "unsloth" for 
# lower VRAM at 8K context. If issues, fall back to True.
```

SFTConfig differences from 9B script:
- `packing=False` (MoE packing not validated)
- `max_seq_length=8192` (default)
- `num_train_epochs=2` (small dataset)
- `learning_rate=2e-5` (not 1e-4)
- `lr_scheduler_type="linear"` (Jackrong used linear)
- `weight_decay=0.001`
- `dataset_text_field="text"` (always raw JSONL, no pre-tokenized mode needed)

train_on_responses_only — same as 9B:
```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

Save — LoRA adapter first, then merged 16bit:
```python
# Save LoRA adapter (small, fast)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

# Save merged 16bit (large, slow but needed for GGUF)
if args.save_merged:
    merged_dir = args.output_dir + "-merged"
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
```

CLI args should include all the same ones as the 9B script (`--data`, `--output-dir`, `--max-steps`, `--max-seq-length`, `--batch-size`, `--grad-accum`, `--lr`, `--lora-r`, `--lora-alpha`, `--save-steps`, `--logging-steps`, `--warmup-ratio`, `--no-response-only`) PLUS:
- `--save-merged` flag to trigger merged 16bit save after training
- `--include-gdn-modules` flag to add GDN-specific LoRA targets (untested, ablation)
- NO `--qlora` flag (not viable for MoE)

The script should include the same `normalize_messages()` and `build_text_dataset()` functions from the 9B script (for tool_calls handling), but NOT the VLM packing hack or pre-tokenized parquet support (not needed for 5K rows).

Use `FastLanguageModel.for_training(model)` before creating the trainer.

**Steps:**
- [ ] Implement `qwen3.5/35b-a3b/scripts/train_35b_a3b.py` following the spec above
- [ ] Verify it imports cleanly: `python -c "import ast; ast.parse(open('qwen3.5/35b-a3b/scripts/train_35b_a3b.py').read()); print('OK')"`
- [ ] Commit with message: `Add train_35b_a3b.py: SFT training script for Qwen3.5-35B-A3B MoE`

**Acceptance criteria:**
- [ ] Script parses without syntax errors
- [ ] Uses `FastLanguageModel` (not FastVisionModel)
- [ ] Default LoRA targets match Unsloth notebook: `q/k/v/o_proj` + `gate/up/down/gate_up_proj`
- [ ] `--include-gdn-modules` flag adds GDN-specific targets as opt-in
- [ ] Does NOT pass `load_in_16bit` (auto-detected)
- [ ] Packing is disabled by default
- [ ] Default context is 8192, epochs=2, lr=2e-5
- [ ] `--save-merged` flag exists for merged 16bit export
- [ ] No `--qlora` flag
- [ ] `train_on_responses_only` is applied by default

---

### Task 3: Create `qwen3.5/35b-a3b/scripts/provision.sh` — Vast.ai Bootstrap for 35B-A3B

**Context:**
This is the Vast.ai provisioning script for the 35B-A3B fine-tune. It's based on `qwen3.5/v1.1/scripts/provision.sh` (the 9B version) but adapted for the 35B-A3B model. The key differences: (1) downloads `train_35b_a3b.py` instead of `train_unsloth.py`, (2) NO VLM packing patch needed, (3) pre-downloads `Qwen/Qwen3.5-35B-A3B` tokenizer/config (NOT `Qwen/Qwen3.5-9B`), (4) also downloads `build_35b_subset.py` so we can build the training data on-instance.

The script should be almost identical to `provision.sh` but with these changes. It still needs: venv, unsloth, flash-linear-attention, causal-conv1d (GDN layers exist in 35B-A3B too), GPU arch detection.

**Instance creation note:** Must use `--disk 250` (not 80) to accommodate merge+GGUF pipeline.

Also add a verification step for `transformers>=5.0` (required for Qwen3.5).

**Files:**
- Create: `qwen3.5/35b-a3b/scripts/provision.sh`

**What to implement:**
Copy the structure of `qwen3.5/v1.1/scripts/provision.sh` line-by-line, with these changes:

1. Header comment: "DeltaCoder Qwen3.5 v1.1 — 35B-A3B Vast.ai Provisioning Script"
2. Script downloads section — download these scripts:
   - `train_35b_a3b.py` (instead of `train_unsloth.py`)
   - `build_35b_subset.py` (for building training data on-instance)
   - Do NOT download `patch_vlm_packing.py` (not needed)
3. Remove the VLM packing patch step entirely
4. Pre-download model section — change from `Qwen/Qwen3.5-9B` to `Qwen/Qwen3.5-35B-A3B`
5. Also install llama.cpp for GGUF conversion:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
   cd /workspace/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
   ```

Everything else stays the same: venv creation, unsloth install, flash-linear-attention, causal-conv1d build with detected arch patching, verification checks.

**Steps:**
- [ ] Implement `qwen3.5/35b-a3b/scripts/provision.sh` following the spec above
- [ ] Verify it's valid bash: `bash -n qwen3.5/35b-a3b/scripts/provision.sh`
- [ ] Commit with message: `Add provision.sh: Vast.ai bootstrap for 35B-A3B fine-tune`

**Acceptance criteria:**
- [ ] Script passes `bash -n` (no syntax errors)
- [ ] Downloads `train_35b_a3b.py` and `build_35b_subset.py` (not `train_unsloth.py`)
- [ ] Does NOT apply VLM packing patch
- [ ] Pre-downloads `Qwen/Qwen3.5-35B-A3B` tokenizer/config
- [ ] Builds llama.cpp with CUDA support
- [ ] Still installs flash-linear-attention + causal-conv1d (GDN acceleration)

---

### Task 4: Create `qwen3.5/35b-a3b/scripts/merge_and_export.py` — Merge LoRA + GGUF Export for MoE

**Context:**
Unsloth's `save_pretrained_gguf` is broken for MoE models (bug #4294). This script handles the manual merge + GGUF conversion using llama.cpp directly. Unlike the 9B version (`merge_and_export_dpo.py`) which does a two-stage merge (SFT + DPO), this script only does a single merge (SFT LoRA → merged) since we're not doing DPO for the 35B-A3B initially.

If training used `--save-merged`, the merged model already exists and we skip straight to GGUF conversion. Otherwise, we merge the LoRA adapter first using PEFT.

**Files:**
- Create: `qwen3.5/35b-a3b/scripts/merge_and_export.py`

**What to implement:**
A script with:
- `--adapter-dir` (LoRA adapter directory, default: `./outputs/deltacoder-35b-a3b-v1.1`)
- `--merged-dir` (where to save merged model, default: `./outputs/deltacoder-35b-a3b-v1.1-merged`)
- `--gguf-dir` (where to save GGUFs, default: `./outputs/deltacoder-35b-a3b-v1.1-gguf`)
- `--llama-cpp-dir` (path to llama.cpp, default: `/workspace/llama.cpp`)
- `--filename-prefix` (default: `DeltaCoder-35B-A3B-v1.1`)
- `--skip-merge` flag (if merged model already exists from `--save-merged` in training)
- `--upload` flag + `--hf-token` for HuggingFace upload
- `--keep-merged` flag to keep the merged bf16 model (otherwise deleted to save disk)

**Recommended flow:** Use `--save-merged` during training (saves merged 16bit while model is in VRAM),
then `--skip-merge` in the export script. This avoids reloading 70GB weights.

Merge step (if not `--skip-merge` — fallback only):
```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# NOTE: Use AutoModel (not AutoModelForCausalLM) — the model is Qwen3_5MoeForConditionalGeneration
# which is a VLM class, not CausalLM. AutoModel auto-dispatches correctly.
base = AutoModel.from_pretrained("Qwen/Qwen3.5-35B-A3B", torch_dtype=torch.bfloat16,
                                  device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, args.adapter_dir)
model = model.merge_and_unload()
model.save_pretrained(args.merged_dir)
tokenizer.save_pretrained(args.merged_dir)
```

Note: This merge step needs ~70GB RAM. On an H100 with 80GB VRAM, `device_map="auto"` should work. May need CPU offloading.

GGUF conversion:
```bash
python3 llama.cpp/convert_hf_to_gguf.py merged_dir --outfile gguf_dir/prefix.gguf --outtype bf16
llama.cpp/build/bin/llama-quantize gguf_dir/prefix.gguf gguf_dir/prefix-Q4_K_M.gguf Q4_K_M
llama.cpp/build/bin/llama-quantize gguf_dir/prefix.gguf gguf_dir/prefix-Q8_0.gguf Q8_0
# ... etc for other quants
```

Quant list (same as 9B): Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S, Q4_K_M, Q5_K_S, Q5_0, Q5_K_M, Q6_K, Q8_0, BF16.

Upload (if `--upload`): Use `huggingface_hub.HfApi().upload_file()` to `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B-GGUF`.

**Steps:**
- [ ] Implement `qwen3.5/35b-a3b/scripts/merge_and_export.py` following the spec above
- [ ] Verify it imports cleanly: `python -c "import ast; ast.parse(open('qwen3.5/35b-a3b/scripts/merge_and_export.py').read()); print('OK')"`
- [ ] Commit with message: `Add merge_and_export.py: merge LoRA + GGUF export for 35B-A3B MoE`

**Acceptance criteria:**
- [ ] Script parses without syntax errors
- [ ] Supports `--skip-merge` for pre-merged models
- [ ] Uses llama.cpp for GGUF conversion (not Unsloth's broken `save_pretrained_gguf`)
- [ ] Generates all standard quant levels
- [ ] Upload targets `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B-GGUF`

---

### Task 5: Update AGENTS.md — Document 35B-A3B Pipeline

**Context:**
AGENTS.md is the single source of truth for the project. It needs a new section documenting the 35B-A3B pipeline, including the architecture differences, training approach, known issues (GGUF bug, no QLoRA, packing unclear), and quick commands.

**Files:**
- Modify: `AGENTS.md`

**What to implement:**
Add a new section `## 13. DeltaCoder 35B-A3B (Qwen3.5-35B-A3B MoE)` after the existing section 12, with:

1. **Overview**: 35B total / 3B active MoE model, same GDN architecture as 9B, bf16 LoRA only
2. **Key differences from 9B**: table comparing loader, LoRA targets, VRAM, packing, GGUF export
3. **Known issues**:
   - Unsloth bug #4294: LoRA-to-GGUF broken for MoE → use manual merge + llama.cpp
   - QLoRA not viable (BnB doesn't support MoE nn.Parameter format)
   - Packing not validated for MoE — disabled
   - `fast_inference=False` required (not supported for MoE)
   - Triton kernel needs 131072 bytes shared memory — may fail on some GPUs (need 80GB class)
4. **Quick commands**:
    ```bash
    # Build 5K subset
    python qwen3.5/35b-a3b/scripts/build_35b_subset.py --data-dir qwen3.5/v1.1/data/v1.1_pruned --output qwen3.5/35b-a3b/data/v1.1_35b_sft.jsonl

    # Dry run (20 steps)
    python qwen3.5/35b-a3b/scripts/train_35b_a3b.py --data qwen3.5/35b-a3b/data/v1.1_35b_sft.jsonl --max-steps 20

    # Full training
    python qwen3.5/35b-a3b/scripts/train_35b_a3b.py --data qwen3.5/35b-a3b/data/v1.1_35b_sft.jsonl --save-merged

    # GGUF export
    python qwen3.5/35b-a3b/scripts/merge_and_export.py --skip-merge --merged-dir ./outputs/deltacoder-35b-a3b-v1.1-merged
    ```

Also update the **Key Scripts** table (section 6) to add the four new scripts (`qwen3.5/35b-a3b/scripts/build_35b_subset.py`, `qwen3.5/35b-a3b/scripts/train_35b_a3b.py`, `qwen3.5/35b-a3b/scripts/provision.sh`, `qwen3.5/35b-a3b/scripts/merge_and_export.py`).

Also update **HuggingFace Repos** (section 8) to add:
- `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B` — 35B-A3B adapter
- `danielcherubini/Qwen3.5-DeltaCoder-35B-A3B-GGUF` — 35B-A3B GGUFs

**Steps:**
- [ ] Add section 13 to AGENTS.md with all content above
- [ ] Update section 6 Key Scripts table with 4 new scripts
- [ ] Update section 8 HuggingFace Repos with 2 new repos
- [ ] Commit with message: `Update AGENTS.md: document 35B-A3B MoE pipeline`

**Acceptance criteria:**
- [ ] Section 13 exists with overview, differences table, known issues, quick commands
- [ ] Key scripts table includes `build_35b_subset.py`, `train_35b_a3b.py`, `provision_35b.sh`, `merge_and_export_35b.py`
- [ ] HF repos section lists 35B-A3B adapter and GGUF repos

---

## Execution Plan

### On local machine (before GPU rental):
1. Task 1: `build_35b_subset.py` — curate 5K subset
2. Task 2: `train_35b_a3b.py` — training script
3. Task 3: `provision.sh` — provisioning script
4. Task 4: `merge_and_export.py` — merge + GGUF
5. Task 5: Update AGENTS.md
6. Push to GitHub

### On Vast.ai H100 SXM 80GB (~$2/hr), **250GB+ disk**:
```bash
# IMPORTANT: Create instance with --disk 250 (not 80!) for merge+GGUF pipeline
# Model cache (70GB) + merged model (70GB) + GGUF bf16 (70GB) = 210GB peak

# 1. Bootstrap (~10 min — includes llama.cpp build)
curl -fsSL -o /workspace/provision.sh https://raw.githubusercontent.com/danielcherubini/DeltaCoder/main/qwen3.5/35b-a3b/scripts/provision.sh
bash /workspace/provision.sh

# 2. Upload data (~1 min)
# From local: scp the existing preprocessed data
scp -P <PORT> qwen3.5/v1.1/data/competitive_programming_converted.jsonl root@<IP>:/workspace/data/
scp -P <PORT> qwen3.5/v1.1/data/qwen3_coder_distill_converted.jsonl root@<IP>:/workspace/data/
scp -P <PORT> qwen3.5/v1.1/data/v1.1_pruned/magicoder_filtered.jsonl root@<IP>:/workspace/data/
scp -P <PORT> qwen3.5/v1.1/data/v1.1_pruned/code_feedback_filtered.jsonl root@<IP>:/workspace/data/

# 3. Build subset (seconds)
source /workspace/venv/bin/activate
python /workspace/build_35b_subset.py --data-dir /workspace/data --output /workspace/v1.1_35b_sft.jsonl

# 4. Dry run (20 steps, ~10 min)
nohup python /workspace/train_35b_a3b.py \
    --data /workspace/v1.1_35b_sft.jsonl \
    --max-steps 20 \
    --output-dir /workspace/outputs/35b-dryrun \
    > /workspace/35b_dryrun.log 2>&1 &
tail -f /workspace/35b_dryrun.log

# 5. Full training (~10-14 hrs)
nohup python /workspace/train_35b_a3b.py \
    --data /workspace/v1.1_35b_sft.jsonl \
    --save-merged \
    --output-dir /workspace/outputs/deltacoder-35b-a3b-v1.1 \
    > /workspace/35b_train.log 2>&1 &
tail -f /workspace/35b_train.log

# 6. GGUF export (~30 min)
python /workspace/merge_and_export.py \
    --skip-merge \
    --merged-dir /workspace/outputs/deltacoder-35b-a3b-v1.1-merged \
    --gguf-dir /workspace/outputs/deltacoder-35b-a3b-v1.1-gguf \
    --llama-cpp-dir /workspace/llama.cpp \
    --upload --hf-token $HF_TOKEN
```

**Total estimated cost: ~$30-40** (10-14 hrs training + ~1 hr setup/export at ~$2/hr, 250GB disk adds ~$0.10/hr)

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| OOM on H100 80GB at 8K context | Low-Medium | Jackrong proved 8K works, but VRAM is very tight (~78-80GB). Fallback: reduce context to 4096. |
| Disk space insufficient | High | 80GB is NOT enough. Must use 250GB+ disk for merge+GGUF pipeline. |
| GDN modules conflict with MoE auto-detection | Medium | GDN targets are opt-in (`--include-gdn-modules`). Default matches validated Unsloth notebook. |
| Triton kernel shared memory failure | Low | Only affects some GPUs. H100 SXM should be fine (Jackrong confirmed). |
| llama.cpp convert fails for MoE | Low | llama.cpp supports Qwen3.5 MoE (verified by community GGUF uploads). Fallback: ik_llama.cpp. |
| Training loss doesn't converge | Medium | 5K rows may be too few for our coding mix. If loss stagnates, increase to 3 epochs or add data. |
| Merged model config issues for vLLM | High | Same wrapping issues as 9B. Document and handle in export script. |
| `load_in_16bit` param not recognized | Low | Don't pass it — let Unsloth auto-detect bf16 when `load_in_4bit=False`. |
