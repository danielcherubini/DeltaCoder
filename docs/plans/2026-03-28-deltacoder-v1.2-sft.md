# DeltaCoder v1.2 — Broader SFT Plan

**Goal:** Retrain DeltaCoder from `Qwen/Qwen3.5-9B` base with a broader, higher-quality SFT dataset covering code instruction following, tool-calling, and chain-of-thought reasoning — then apply the existing DPO pipeline on top.

**Architecture:** Preprocess 7 datasets → merge/shuffle → single SFT run (Unsloth) → merge SFT LoRA → serve via vLLM → generate DPO pairs → DPO training (plain HF+PEFT+TRL) → GGUF export.

**Tech Stack:** Unsloth (SFT only), HuggingFace + PEFT + TRL (DPO), llama.cpp (GGUF export), Vast.ai H100

---

## Decisions (Final)

1. **Base model**: `Qwen/Qwen3.5-9B` — clean base, not Jackrong distill
2. **Sequence length**: 4096 for first run. If OOM during training, reduce batch size before reducing seq length.
3. **DPO pairs**: Regenerate on-policy from the v1.2 SFT model (better signal than reusing v1.1 pairs)
4. **Dataset merging**: Natural counts (no upsampling). The 60/20/20 split is approximate.
5. **DPO framework**: Plain HuggingFace + PEFT + TRL. **DO NOT use Unsloth DPOTrainer** — it has a known bug where it incorrectly detects Qwen3.5 as a vision model and crashes.

---

## Dataset Mix

| Dataset | HF ID | Rows (capped) | Category |
|---------|-------|--------------|----------|
| Magicoder | `ise-uiuc/Magicoder-OSS-Instruct-75K` | 50K | Code instruction |
| CoderForge | `togethercomputer/CoderForge-Preview` | 50K | Code + tool-calling |
| Code-Feedback | `m-a-p/Code-Feedback` | 50K | Code revision |
| Hermes | `NousResearch/hermes-function-calling-v1` | ~11.6K (all) | Tool-calling |
| Glaive | `glaiveai/glaive-function-calling-v2` | 50K | Tool-calling |
| Opus Reasoning | `nohurry/Opus-4.6-Reasoning-3000x-filtered` | ~3K (all) | Reasoning |
| Qwen3.5 Reasoning | `Jackrong/Qwen3.5-reasoning-700x` | ~700 (all) | Reasoning |

**Total: ~215K rows**

### Output Schema for All Preprocessed Files

Every preprocessing script must produce a JSONL file where each line is:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "source": "dataset_name", "id": "unique_id"}
```

For tool-calling examples that include function definitions:
```json
{"messages": [...], "tools": [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}], "source": "dataset_name", "id": "unique_id"}
```

The training script applies `tokenizer.apply_chat_template(messages, tools=tools)` at training time.

---

## Training Config

### SFT (Stage 1)

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3.5-9B` |
| Method | LoRA (r=64, alpha=32) |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Sequence length | 4096 |
| Batch size | 2, gradient_accumulation_steps=8 (effective batch=16) |
| Learning rate | 1e-4 (cosine schedule) |
| Warmup ratio | 0.05 |
| Epochs | 1 |
| Precision | BF16 (no QLoRA) |
| dataset_num_proc | 1 (Qwen3.5 tokenizer crashes with multiprocessing) |
| Framework | Unsloth |
| Hardware | Vast.ai H100 80GB |
| Output dir | `./outputs/deltacoder-9b-v1.2` |

### DPO (Stage 2)

| Parameter | Value |
|-----------|-------|
| Base model | Merged v1.2 SFT model (Qwen3.5-9B + v1.2 LoRA, merged to full weights) |
| Pair generation | vLLM serving merged model, 10K AceCode problems, 8 samples each |
| Beta | 0.1 |
| Loss type | sigmoid |
| Learning rate | 5e-6 (cosine) |
| Effective batch | 16 |
| Framework | Plain HuggingFace + PEFT + TRL (scripts/train_dpo.py) |
| Output dir | `./outputs/deltacoder-9b-v1.2-dpo` |

---

## Tasks

### Task 1: Preprocess Magicoder-OSS-Instruct-75K

**Context:**
Magicoder is a high-quality synthetic coding instruction dataset. It must be converted to the standard JSONL schema (`messages` + `source` + `id`) before merging. The dataset columns are `problem` (user instruction) and `solution` (assistant response) — NOT `instruction`/`response` as is common in other datasets.

**Files:**
- Create: `scripts/preprocess_magicoder.py`
- Output: `data/magicoder_converted.jsonl`

**What to implement:**
1. Load `ise-uiuc/Magicoder-OSS-Instruct-75K` split `train` from HuggingFace
2. **First**: print column names and 2 sample rows to verify schema before writing conversion
3. Convert each row: `problem` → user message, `solution` → assistant message
4. Filter rows where `solution` is empty or whitespace
5. Shuffle with seed=42, take first 50K rows
6. Write each row as: `{"messages": [{"role": "user", "content": problem}, {"role": "assistant", "content": solution}], "source": "magicoder", "id": f"magicoder-{i}"}`
7. Print final row count

**Steps:**
- [ ] Create `scripts/preprocess_magicoder.py`
- [ ] Run `python scripts/preprocess_magicoder.py` and verify output schema
- [ ] Confirm `data/magicoder_converted.jsonl` has ~50K rows
- [ ] Commit: `"add preprocess_magicoder.py"`

**Acceptance criteria:**
- [ ] Output file exists at `data/magicoder_converted.jsonl`
- [ ] Each line has `messages`, `source`, `id` keys
- [ ] `messages[0]["role"] == "user"`, `messages[1]["role"] == "assistant"`
- [ ] Row count is ~50K

---

### Task 2: Preprocess Code-Feedback

**Context:**
Code-Feedback (`m-a-p/Code-Feedback`) is a multi-turn code revision dataset where the model gives feedback and iteratively improves code. It is already in conversation format but needs to be verified and normalized to the standard schema. It teaches self-correction behavior which is valuable for DeltaCoder.

**Files:**
- Create: `scripts/preprocess_code_feedback.py`
- Output: `data/code_feedback_converted.jsonl`

**What to implement:**
1. Load `m-a-p/Code-Feedback` split `train` from HuggingFace
2. **First**: print column names and 2 sample rows to verify schema
3. The dataset likely has a `messages` or `conversations` column with role/content dicts — inspect and normalize to `{"role": "...", "content": "..."}` format (roles should be `user`/`assistant`, not `human`/`gpt`)
4. Filter rows where the conversation has fewer than 2 turns or any empty content
5. Shuffle with seed=42, take first 50K rows
6. Write each row as: `{"messages": [...normalized messages...], "source": "code_feedback", "id": f"code_feedback-{i}"}`

**Steps:**
- [ ] Create `scripts/preprocess_code_feedback.py`
- [ ] Run it and print 2 sample output rows to verify format
- [ ] Confirm output has ~50K rows
- [ ] Commit: `"add preprocess_code_feedback.py"`

**Acceptance criteria:**
- [ ] Output file exists at `data/code_feedback_converted.jsonl`
- [ ] All messages use `user`/`assistant` roles (not `human`/`gpt`)
- [ ] No empty content fields
- [ ] Row count is ~50K

---

### Task 3: Preprocess Hermes Function-Calling v1

**Context:**
`NousResearch/hermes-function-calling-v1` has 5 subsets totaling ~11.6K rows. Conversations use `{"from": "...", "value": "..."}` format (not `role`/`content`). The `tools` column contains a JSON string of function definitions. All subsets should be used.

Subsets: `func_calling_singleturn` (~1.89K), `func_calling` (~1.89K), `glaive_func_calling` (~5.21K), `json_mode_agentic` (~1.34K), `json_mode_singleturn` (~1.24K)

**Files:**
- Create: `scripts/preprocess_hermes.py`
- Output: `data/hermes_converted.jsonl`

**What to implement:**
1. Load all 5 subsets from `NousResearch/hermes-function-calling-v1` and concatenate
2. **First**: print column names and 2 sample rows from each subset to verify schema
3. For each row:
   - Parse `tools` column (JSON string) → list of tool dicts in OpenAI format
   - Convert `conversations` column: map `{"from": "system", "value": ...}` → `{"role": "system", "content": ...}`, `{"from": "human", ...}` → `{"role": "user", ...}`, `{"from": "gpt", ...}` → `{"role": "assistant", ...}`
   - Skip rows with empty conversations
4. Write each row as: `{"messages": [...], "tools": [...], "source": "hermes", "id": f"hermes-{i}"}`
5. Print total row count (expect ~11.6K)

**Steps:**
- [ ] Create `scripts/preprocess_hermes.py`
- [ ] Run and verify output
- [ ] Commit: `"add preprocess_hermes.py"`

**Acceptance criteria:**
- [ ] All 5 subsets loaded and concatenated
- [ ] Messages use `user`/`assistant`/`system` roles
- [ ] `tools` field is a list of dicts (not a JSON string)
- [ ] Row count ~11.6K

---

### Task 4: Preprocess Glaive Function-Calling v2

**Context:**
`glaiveai/glaive-function-calling-v2` is stored as raw text in two columns: `system` (containing tool definitions as text) and `chat` (containing the full conversation as delimited text). This requires substantial text parsing.

Format details:
- `system`: Raw text like `"SYSTEM: You are a helpful assistant with access to the following functions: ..."` containing JSON tool definitions inline
- `chat`: Flat text with `USER: ...`, `ASSISTANT: ...`, `FUNCTION RESPONSE: ...` sections, tool calls formatted as `<functioncall> {"name": "...", "arguments": {...}}`, turns separated by `<|endoftext|>`

**Files:**
- Create: `scripts/preprocess_glaive.py`
- Output: `data/glaive_converted.jsonl`

**What to implement:**
1. Load `glaiveai/glaive-function-calling-v2` split `train` from HuggingFace
2. **First**: print column names and 1 full raw sample row to understand exact format
3. Parse `system` text to extract tool definitions (find JSON arrays/objects embedded in the text)
4. Parse `chat` text by splitting on `USER:`, `ASSISTANT:`, `FUNCTION RESPONSE:` markers
5. Convert `<functioncall> {...}` blocks to OpenAI `tool_calls` format in assistant messages
6. Convert `FUNCTION RESPONSE:` sections to `{"role": "tool", "content": "..."}` messages
7. Skip rows that fail to parse cleanly
8. Shuffle with seed=42, take first 50K rows
9. Write each row as: `{"messages": [...], "tools": [...], "source": "glaive", "id": f"glaive-{i}"}`
10. Print parsed count vs skipped count

**Steps:**
- [ ] Create `scripts/preprocess_glaive.py`
- [ ] Run on 100 rows first to verify parsing logic before full run
- [ ] Run full and confirm output
- [ ] Commit: `"add preprocess_glaive.py"`

**Acceptance criteria:**
- [ ] Output has ~50K rows (less if many parse failures — log skip rate)
- [ ] Tool calls are in OpenAI format (not raw `<functioncall>` text)
- [ ] No raw `USER:`/`ASSISTANT:` markers in output

---

### Task 5: Preprocess CoderForge-Preview

**Context:**
`togethercomputer/CoderForge-Preview` was used for DeltaCoder v1 and has an existing preprocessing script (`scripts/preprocess_coderforge.py`). Reuse it but verify the output format matches the new standard schema and update if needed. Use the `filtered_reward1` subset.

**Files:**
- Modify: `scripts/preprocess_coderforge.py` (if output format doesn't match standard schema)
- Output: `data/coderforge_converted.jsonl`

**What to implement:**
1. Run existing `scripts/preprocess_coderforge.py` and inspect 2 output rows
2. Verify output has `messages`, `source`, `id` keys
3. If `source` or `id` keys are missing, add them
4. If role names differ (`human`/`gpt` instead of `user`/`assistant`), normalize them
5. Cap at 50K rows (shuffle with seed=42 if more)
6. **Important**: The existing script uses `streaming=True` which prevents shuffle+cap. Switch to `streaming=False`, then use `.shuffle(seed=42).select(range(50000))` before writing.

**Steps:**
- [ ] Run existing script, inspect output
- [ ] Fix format if needed
- [ ] Confirm `data/coderforge_converted.jsonl` exists with ~50K rows
- [ ] Commit: `"normalize coderforge preprocess output schema"`

**Acceptance criteria:**
- [ ] Output has `messages`, `source`, `id` keys
- [ ] Messages use `user`/`assistant` roles
- [ ] Row count ~50K

---

### Task 6: Preprocess Reasoning Datasets

**Context:**
Two reasoning datasets need to be included to maintain Qwen3.5's chain-of-thought capability: `nohurry/Opus-4.6-Reasoning-3000x-filtered` (~3K rows) and `Jackrong/Qwen3.5-reasoning-700x` (~700 rows). Both are small so use all rows. Schema must be inspected before writing conversion — these datasets have not been preprocessed before.

**Files:**
- Create: `scripts/preprocess_reasoning.py`
- Output: `data/opus_reasoning_converted.jsonl`, `data/qwen35_reasoning_converted.jsonl`

**What to implement:**
1. For each dataset: load, **print column names and 2 sample rows**, then convert
2. `nohurry/Opus-4.6-Reasoning-3000x-filtered`: likely has `instruction`/`output` or conversation columns — inspect and convert
3. `Jackrong/Qwen3.5-reasoning-700x`: inspect schema and convert
4. Both should use the standard `{"messages": [...], "source": "...", "id": "..."}` schema
5. Write to separate output files (merged in Task 7)

**Steps:**
- [ ] Create `scripts/preprocess_reasoning.py`
- [ ] Run and inspect output for both datasets
- [ ] Commit: `"add preprocess_reasoning.py for opus and qwen35 reasoning datasets"`

**Acceptance criteria:**
- [ ] `data/opus_reasoning_converted.jsonl` exists, ~3K rows
- [ ] `data/qwen35_reasoning_converted.jsonl` exists, ~700 rows
- [ ] Standard schema with `messages`, `source`, `id`

---

### Task 7: Merge + Shuffle All Datasets

**Context:**
All 7 preprocessed datasets must be merged, shuffled, and written to a single training file. The existing `scripts/merge_datasets.py` has hardcoded v1 input files and must be updated. Do NOT upsample smaller datasets — use natural counts. The 60/20/20 code/tool/reasoning split is approximate based on capped row counts.

**Files:**
- Modify: `scripts/merge_datasets.py`
- Output: `data/v1.2_sft_train.jsonl`

**What to implement:**
Replace the `INPUT_FILES` list with:
```python
INPUT_FILES = [
    "data/magicoder_converted.jsonl",
    "data/coderforge_converted.jsonl",
    "data/code_feedback_converted.jsonl",
    "data/hermes_converted.jsonl",
    "data/glaive_converted.jsonl",
    "data/opus_reasoning_converted.jsonl",
    "data/qwen35_reasoning_converted.jsonl",
]
OUTPUT_FILE = "data/v1.2_sft_train.jsonl"
```

After merging and shuffling (seed=42):
- Print total row count
- Print per-source row count distribution to verify proportions

**Steps:**
- [ ] Update `scripts/merge_datasets.py`
- [ ] Run and confirm output at `data/v1.2_sft_train.jsonl`
- [ ] Verify ~215K rows total and distribution looks correct
- [ ] Commit: `"update merge_datasets.py for v1.2 dataset mix"`

**Acceptance criteria:**
- [ ] Output exists at `data/v1.2_sft_train.jsonl`
- [ ] Total rows ~215K
- [ ] Source distribution printed and matches expected counts

---

### Task 8: Update SFT Training Script for v1.2

**Context:**
The existing `scripts/train_unsloth.py` is hardcoded for v1 (CoderForge, Jackrong base model, 50K rows). It must be updated for v1.2. Key changes: new base model, load from local JSONL instead of HuggingFace, larger dataset, different output dir, `dataset_num_proc=1` (mandatory — Qwen3.5 tokenizer crashes with multiprocessing).

**Files:**
- Modify: `scripts/train_unsloth.py`
- Create: `configs/deltacoder-9b-lora-v1.2.yaml` (documentation only, not used by training script)

**What to implement in `scripts/train_unsloth.py`:**

Change these constants:
```python
# OLD                                          # NEW
MODEL_NAME = "Jackrong/..."                    MODEL_NAME = "Qwen/Qwen3.5-9B"
SUBSET_SIZE = 50_000                           # Remove — use full dataset
OUTPUT_DIR = "./outputs/deltacoder-9b"         OUTPUT_DIR = "./outputs/deltacoder-9b-v1.2"
```

Change dataset loading (find the `load_dataset` call and replace):
```python
# OLD: loads from HuggingFace with subset
# NEW: loads from local JSONL
from datasets import load_dataset
dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
```

Add CLI arguments:
```python
parser.add_argument("--dataset", type=str, default="data/v1.2_sft_train.jsonl")
parser.add_argument("--max_steps", type=int, default=-1,
                    help="Override epochs with fixed step count (e.g. 5 for dry run)")
```

Pass `--max_steps` to SFTConfig:
```python
sft_config_kwargs = dict(...)
if args.max_steps > 0:
    sft_config_kwargs["max_steps"] = args.max_steps
trainer = SFTTrainer(..., args=SFTConfig(**sft_config_kwargs))
```

Change `dataset_num_proc` in SFTConfig from `16` to `1` (critical — crashes otherwise).

Update `format_messages()` function to handle the new JSONL schema:
- The new JSONL has `messages` as a **Python list** (already parsed), NOT a JSON string. Remove any `json.loads()` call on the `messages` field.
- Pass `tools=example.get("tools")` to `tokenizer.apply_chat_template()` so tool definitions are included in the formatted text.
- Example:
```python
def format_messages(example):
    messages = example["messages"]  # already a list, no json.loads needed
    tools = example.get("tools")    # None for non-tool-calling examples
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}
```

**`configs/deltacoder-9b-lora-v1.2.yaml`** is documentation only (like `deltacoder-9b-dpo.yaml`). Include the training parameters as YAML comments for reference.

**Steps:**
- [ ] Update `scripts/train_unsloth.py` with above changes
- [ ] Create `configs/deltacoder-9b-lora-v1.2.yaml` with config documentation
- [ ] Do a dry run: `python scripts/train_unsloth.py --max_steps 5` to verify it loads without errors
- [ ] Commit: `"update train_unsloth.py for v1.2: new base model, JSONL dataset, dataset_num_proc=1"`

**Acceptance criteria:**
- [ ] Script loads `Qwen/Qwen3.5-9B` as base model
- [ ] Script loads from `data/v1.2_sft_train.jsonl` by default
- [ ] `dataset_num_proc=1` in SFTConfig
- [ ] Dry run with `--max_steps 5` completes without error

---

### Task 9: SFT Training (Cloud — Vast.ai H100)

**Context:**
Run the full SFT training on Vast.ai. Use the Unsloth Studio template. Upload the merged dataset and run training. Expected ~12-15 hours on H100 80GB.

**What to do:**
1. Rent Vast.ai H100 80GB with Unsloth Studio template (≥100GB disk)
2. Clone repo: `git clone https://github.com/danielcherubini/DeltaCoder.git /workspace/DeltaCoder`
3. Upload dataset: `rsync -avP data/v1.2_sft_train.jsonl root@<instance>:/workspace/DeltaCoder/data/`
4. Install deps: `pip install "transformers==5.3.0" -q`
5. Run training (nohup):
   ```bash
   nohup python scripts/train_unsloth.py \
     --dataset data/v1.2_sft_train.jsonl \
     > logs/train_v1.2.log 2>&1 &
   ```
6. Monitor: `tail -f /workspace/DeltaCoder/logs/train_v1.2.log`
7. When done: rsync LoRA adapter back: `rsync -avP root@<instance>:/workspace/DeltaCoder/outputs/deltacoder-9b-v1.2/lora_adapter/ ~/Coding/AI/DeltaCoder/outputs/deltacoder-9b-v1.2/lora_adapter/`

**Acceptance criteria:**
- [ ] Training completes 1 epoch without OOM
- [ ] Final loss is lower than v1 (~0.94)
- [ ] LoRA adapter downloaded locally

---

### Task 10: DPO Pair Generation + Training (Cloud)

**Context:**
Generate on-policy DPO pairs from the v1.2 SFT model and train DPO. Use the same pipeline as v1.1 but with the v1.2 merged model as the generator. 

⚠️ **DO NOT use Unsloth DPOTrainer** — it has a known bug where it detects Qwen3.5 as a vision model and crashes. Use `scripts/train_dpo.py` which uses plain HuggingFace + PEFT + TRL.

**What to do:**

**Step A: Merge SFT LoRA**
On a Vast.ai instance (vLLM template, H100):
```bash
python3 -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3.5-9B', torch_dtype=torch.bfloat16, device_map='cpu')
model = PeftModel.from_pretrained(base, '/workspace/DeltaCoder/outputs/deltacoder-9b-v1.2/lora_adapter')
model = model.merge_and_unload()
model.save_pretrained('/workspace/merged_v1.2')
AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B').save_pretrained('/workspace/merged_v1.2')
"
```

**Step B: Serve with vLLM**
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model /workspace/merged_v1.2 \
  --served-model-name deltacoder \
  --port 18000 --host 0.0.0.0 \
  --dtype bfloat16 --max-model-len 4096 \
  --enforce-eager \
  > /workspace/logs/vllm.log 2>&1 &
```

**Step C: Generate pairs**
```bash
nohup env HF_TOKEN=$HF_TOKEN python scripts/generate_dpo_pairs.py \
  --n-problems 10000 --n-samples 8 \
  --api-base http://localhost:18000/v1 \
  --model deltacoder \
  --output data/dpo_pairs_v1.2.jsonl \
  > logs/generate_dpo_v1.2.log 2>&1 &
```

**Step D: DPO training** (update `scripts/train_dpo.py` `--output-dir`):
```bash
python scripts/train_dpo.py \
  --data data/dpo_pairs_v1.2.jsonl \
  --output-dir ./outputs/deltacoder-9b-v1.2-dpo
```

**Step E: Download adapter**
```bash
rsync -avP root@<instance>:/workspace/DeltaCoder/outputs/deltacoder-9b-v1.2-dpo/lora_adapter/ \
  ~/Coding/AI/DeltaCoder/outputs/deltacoder-9b-v1.2-dpo/lora_adapter/
```

**Acceptance criteria:**
- [ ] DPO pairs generated (~4K+ pairs expected at 45% keep rate)
- [ ] DPO training completes, final loss < 0.55
- [ ] rewards/margins positive and growing during training
- [ ] DPO adapter downloaded locally

---

### Task 11: GGUF Export + Upload

**Context:**
Two-stage merge (v1.2 SFT LoRA + v1.2 DPO LoRA) and export all quants. Use `scripts/merge_and_export_dpo.py` with updated adapter paths. Upload to HuggingFace.

**What to do:**
Update `scripts/merge_and_export_dpo.py` default args:
```python
--sft-adapter  →  outputs/deltacoder-9b-v1.2/lora_adapter  (or HF repo if pushed)
--dpo-adapter  →  outputs/deltacoder-9b-v1.2-dpo/lora_adapter
--merged-dir   →  outputs/deltacoder-9b-v1.2-merged
--gguf-dir     →  outputs/deltacoder-9b-v1.2-gguf
```

Also update the GGUF filename pattern in `merge_and_export_dpo.py` from `DeltaCoder-9B-v1.1-DPO` to `DeltaCoder-9B-v1.2-DPO` in all output path strings (f16_gguf, BF16 copy, and the quant loop).

Run on Vast.ai H100 (needs ≥100GB disk):
```bash
python scripts/merge_and_export_dpo.py --upload --hf-token $HF_TOKEN --skip-sanity
```

**Acceptance criteria:**
- [ ] All 13 quants uploaded to HuggingFace GGUF repo
- [ ] DPO LoRA adapter uploaded to HF main repo
- [ ] Old v1.1 GGUFs deleted from HF repo

---

## Estimated Timeline & Cost

| Step | Time | Cost |
|------|------|------|
| Dataset preprocessing (local) | 1-2hrs | $0 |
| SFT training (H100 80GB, Vast.ai) | ~12-15hrs | ~$20-25 |
| DPO pair generation (H100, vLLM) | ~4-6hrs | ~$7-10 |
| DPO training (H100 80GB) | ~4hrs | ~$6-8 |
| GGUF export + upload | ~1hr | ~$1.50 |
| **Total** | **~22-28hrs** | **~$35-45** |

---

## Expected Improvements over v1.1

- **HumanEval**: 50.6% → 55-60%
- **Tool-calling**: Maintained or improved
- **Terminal-Bench**: 2/4 → 3-4/4 (self-correction from Code-Feedback)
- **Reasoning**: Maintained (explicit reasoning data in SFT mix)
