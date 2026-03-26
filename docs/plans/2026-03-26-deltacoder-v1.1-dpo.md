# DeltaCoder v1.1-DPO Plan

**Goal:** Add a DPO alignment phase on top of DeltaCoder v1 to improve code correctness and reduce self-verification failures, producing a publishable v1.1 model.

**Architecture:** Generate on-policy chosen/rejected preference pairs by running DeltaCoder v1 on TIGER-Lab/AceCode-V2-122K problems and executing against their test cases locally; train a second LoRA adapter on top of the merged v1 model using Unsloth's DPOTrainer; merge, export to GGUF, and publish.

**Tech Stack:** Python, HuggingFace datasets, ik_llama.cpp (local inference on RTX 3080 10GB), Unsloth + TRL DPOTrainer (cloud H200 on Vast.ai), llama.cpp GGUF export.

**Motivation from Terminal-Bench results:**
- `overfull-hbox`: model used a word not in synonyms.txt — hallucination under constraint
- `cobol-modernization`: bytes vs int type error — tested wrong code path, missed the bug
- Both failures are code correctness / self-verification issues that DPO on passing/failing pairs directly targets

---

## Key Constraints

- Base model for DPO: `danielcherubini/Qwen3.5-DeltaCoder-9B` (merged v1, NOT the Jackrong base)
- RTX 3080 10GB is inference-only — no training locally
- DPO training runs on rented Vast.ai H200 (~$3-5 estimated, ~1-2hrs)
- AceCode-V2-122K columns: `id`, `question`, `tests` (List[str]), `source` — no pre-made pairs
- TRL DPOTrainer (≥0.13) requires: `prompt`, `chosen`, `rejected` as **conversational format** (lists of dicts)
- Must call `PatchDPOTrainer()` from Unsloth before importing DPOTrainer
- LoRA target_modules must include GDN projections (proven in v1): `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- DPO learning rate: 5e-6 (10-20x lower than SFT)
- `ref_model=None` — Unsloth uses implicit reference (frozen initial weights) to save VRAM
- Disable Qwen3.5 thinking mode during pair generation (simpler code extraction, avoids `<think>` blocks in pairs)

---

## Task 1: Pair Generation Script

**Files:**
- Create: `scripts/generate_dpo_pairs.py`

**Prerequisites (before running):**
```bash
# Start ik_llama-server with DeltaCoder v1 GGUF
./ik_llama-server -m /path/to/DeltaCoder-9B-Q4_K_M.gguf \
  -ngl 999 -c 4096 -fa 1 --jinja --port 8080

# Required Python packages
pip install openai datasets tqdm
```

**What it does:**
1. Downloads a configurable subset of AceCode-V2-122K (default: `--n-problems 10000`)
2. For each problem, calls DeltaCoder via ik_llama-server's OpenAI-compatible API at `--api-base` (default: `http://localhost:8080/v1`)
3. Samples N=8 completions at temp=0.8, `enable_thinking=False` (Qwen3.5 flag), strips any markdown code fences from responses
4. Executes each completion + test cases in a subprocess with: `env={"PATH": "/usr/bin:/usr/local/bin"}`, `cwd=tmpdir`, `timeout=10`, `stdin=subprocess.DEVNULL` — labels pass/fail
5. Keeps only problems where ≥1 completion passes AND ≥1 fails
6. For each kept problem: picks the shortest-by-token-count passing completion as `chosen`, picks a random failing completion as `rejected`
7. Saves as JSONL with **conversational format**:
   ```json
   {
     "prompt": [{"role": "user", "content": "...question..."}],
     "chosen": [{"role": "assistant", "content": "...passing code..."}],
     "rejected": [{"role": "assistant", "content": "...failing code..."}]
   }
   ```
8. Writes to `data/dpo_pairs.jsonl`, creates `data/` if needed

**Code extraction:** Strip ` ```python ... ``` ` fences, strip leading/trailing whitespace. If model outputs explanation text, extract the first code block only.

**Test execution:** Combine extracted code + `"\n".join(tests)` into a single Python script, run in a fresh tempdir, treat any exception or non-zero exit as fail.

**Expected yield:** With 10K problems × 8 samples at ~50.6% pass rate, expect ~25-35% of problems to have mixed outcomes → ~2500-3500 valid pairs. This is sufficient for DPO.

**Time estimate:** 10K problems × 8 samples × ~150 tokens avg × (1/150 tok/s) ≈ 8-10 hours on romulus. Run overnight.

**Argparse options:**
- `--n-problems` (default: 10000)
- `--n-samples` (default: 8)
- `--api-base` (default: `http://localhost:8080/v1`)
- `--model` (default: `deltacoder`)
- `--output` (default: `data/dpo_pairs.jsonl`)
- `--seed` (default: 42)

**Steps:**
- [ ] Write `scripts/generate_dpo_pairs.py`
- [ ] Quick sanity test: `python scripts/generate_dpo_pairs.py --n-problems 20 --n-samples 2` — verify JSONL output has correct conversational format, code extraction works, test execution works
- [ ] Check token length distribution of generated pairs (print p50/p90/p99) — inform `max_length` choice for training
- [ ] Run full generation overnight: `python scripts/generate_dpo_pairs.py --n-problems 10000 --n-samples 8`
- [ ] Verify yield: print stats (problems tried, pairs kept, keep rate)
- [ ] Commit

---

## Task 2: DPO Training Script

**Files:**
- Create: `scripts/train_dpo.py`
- Create: `configs/deltacoder-9b-dpo.yaml` (reference/documentation only — actual hyperparams in script)

**Cloud setup on Vast.ai H200:**
```bash
# 1. Rent H200 140GB with Unsloth Docker template (same as v1)
# 2. SSH in and run:
pip install flash-linear-attention==0.4.1
pip uninstall causal-conv1d -y
huggingface-cli login   # to download danielcherubini/Qwen3.5-DeltaCoder-9B
# 3. Upload data/dpo_pairs.jsonl
scp data/dpo_pairs.jsonl user@vastai-host:~/DeltaCoder/data/
# 4. Upload scripts/train_dpo.py
```

**What the script does:**
1. Calls `PatchDPOTrainer()` from Unsloth (MUST be before DPOTrainer import)
2. Loads `danielcherubini/Qwen3.5-DeltaCoder-9B` in bf16 via `FastLanguageModel.from_pretrained`
3. Applies LoRA via `FastLanguageModel.get_peft_model` with:
   - `r=32` (configurable — lower than SFT's r=64, but not as risky as r=16)
   - Same `target_modules` as v1 (full attention + GDN + MLP)
   - `lora_alpha=32`, `lora_dropout=0`, `bias="none"`
   - `use_gradient_checkpointing="unsloth"`
4. Loads `data/dpo_pairs.jsonl` — already in conversational format, TRL applies chat template
5. Runs `DPOTrainer` with `DPOConfig` (all DPO params in DPOConfig, NOT as separate kwargs):

```python
DPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    max_length=4096,          # informed by Task 1 token distribution check
    max_prompt_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,   # effective batch=16
    num_train_epochs=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    optim="adamw_torch",
    logging_steps=5,
    save_strategy="steps",
    save_steps=0.5,           # save mid-run checkpoint
    save_total_limit=2,
    output_dir="./outputs/deltacoder-9b-dpo",
    seed=3407,
    report_to="none",
)
```

6. Uses `processing_class=tokenizer` (TRL ≥0.19 API, not `tokenizer=tokenizer`)
7. `ref_model=None` (Unsloth implicit reference)
8. Saves LoRA to `outputs/deltacoder-9b-dpo/lora_adapter`

**Test run (50 steps, verify loss decreases before full run):**
```bash
python scripts/train_dpo.py --max-steps 50
```

**Steps:**
- [ ] Write `scripts/train_dpo.py` with argparse `--max-steps` override
- [ ] Write `configs/deltacoder-9b-dpo.yaml` (documentation of hyperparams)
- [ ] Upload to cloud, run 50-step test — verify DPO loss (`rewards/chosen`, `rewards/rejected`, `loss`) are moving in the right direction (chosen reward increasing, rejected decreasing)
- [ ] Run full training
- [ ] Commit

---

## Task 3: Export, Evaluate, and Publish

**Files:**
- Create: `scripts/merge_and_export_dpo.py` (Python, NOT a shell script variant — avoids Axolotl merge bug)

**What it does:**
1. Loads base model `danielcherubini/Qwen3.5-DeltaCoder-9B` + DPO LoRA adapter
2. Calls `model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")`
3. **Sanity check:** Load merged model, run a quick inference ("write a Python function to reverse a list") — verify it produces sensible output before GGUF conversion
4. Converts merged model to GGUF via llama.cpp `convert_hf_to_gguf.py`
5. Quantizes: Q4_K_M, Q5_K_M, Q6_K, Q8_0 (dropping Q4_K_S — v1.1 can slim the quant list)

**Evaluation (local, after downloading GGUFs):**
```bash
# HumanEval via EvalPlus — compare v1 vs v1.1
python -m evalplus.evaluate --model /path/to/DeltaCoder-v1.1-Q4_K_M.gguf ...

# Terminal-Bench easy tasks — compare v1 vs v1.1
OPENAI_API_KEY=sk-none harbor run \
  --task-name fix-git --task-name cobol-modernization \
  --task-name overfull-hbox --task-name prove-plus-comm \
  --ae GIT_PAGER=cat --ae GIT_EDITOR=true --ae GIT_SEQUENCE_EDITOR=true \
  --ek GIT_PAGER=cat --ek GIT_EDITOR=true --ek GIT_SEQUENCE_EDITOR=true \
  ...
```

**Publish:**
- Push new GGUFs to `danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF` (overwrites v1 files)
- Push new LoRA adapter to `danielcherubini/Qwen3.5-DeltaCoder-9B` (overwrites v1 adapter)
- Update `README.md`: add v1.1-DPO section with benchmark comparison table

**Steps:**
- [ ] Write `scripts/merge_and_export_dpo.py`
- [ ] Run on Vast.ai after training, including inference sanity check
- [ ] Download GGUFs locally
- [ ] Run HumanEval on v1.1 — record score vs v1 (50.6% / 43.9% greedy)
- [ ] Run Terminal-Bench easy 4 tasks on v1.1 — record score vs v1 (50%)
- [ ] Push GGUFs and LoRA to HuggingFace
- [ ] Update `README.md` with v1.1 benchmarks
- [ ] Commit

---

## Estimated Cost & Time

| Phase | Where | Time | Cost |
|-------|-------|------|------|
| Pair generation (10K problems, 8 samples) | romulus (RTX 3080) | ~8-10 hrs overnight | $0 |
| DPO training (1 epoch, ~2.5K-3.5K pairs) | Vast.ai H200 | ~1-2 hrs | ~$3-5 |
| GGUF export + quantization | Vast.ai H200 | ~30 min | ~$1 |
| Evaluation + publish | romulus | ~1-2 hrs | $0 |
| **Total** | | **~11-15 hrs elapsed** | **~$4-6** |

---

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Unsloth DPO + Qwen3.5 GDN architecture untested | Medium | Run 50-step test early; fall back to SFT-only if DPO crashes |
| `save_pretrained_merged` weight corruption (unsloth #3508) | Medium | Inference sanity check before GGUF conversion |
| Pair yield too low (<1000 pairs) | Low-Medium | Increase `--n-problems` to 15K; increase `--n-samples` to 8 |
| Code execution security risk | Low-Medium | Restricted subprocess env, tempdir, timeout=10, `DEVNULL` stdin |
| HumanEval regression after DPO | Low | DPO at lr=5e-6 is conservative; if regression occurs, reduce `num_train_epochs` to 0.5 |
| TRL version API mismatch on cloud | Low | Pin exact versions; use `DPOConfig` not `TrainingArguments` for all DPO params |
