# Repository Restructure: Version Folders → Model-Based Folders

**Goal:** Reorganize the repo from `v1.1/v1.2/v1.3` flat versioning to a model-based hierarchy: `qwen3.5/` (with `v1.0`, `v1.1`, `35b-a3b` subdirs) and `qwen3.6/` (with `v1.0`). Renumber versions and update all internal references.

**Architecture:** Use `git mv` to move folders (preserves blame/history). Then update all hardcoded paths, default CLI args, data file prefixes, GitHub raw URLs, docstrings, .gitignore patterns, AGENTS.md, and plan docs. Historical plan docs get a deprecation header instead of full rewrites.

**Tech Stack:** git, sed/editor for bulk string replacements, grep for verification.

---

## Mapping Table

| Old path | New path | Version renumber |
|----------|----------|-----------------|
| `v1.1/` | `qwen3.5/v1.0/` | v1.1 → v1.0 (original 9B) |
| `v1.2/` | `qwen3.5/v1.1/` | v1.2 → v1.1 (revised 9B, Jackrong-inspired) |
| `v1.3/` | `qwen3.6/v1.0/` | v1.3 → v1.0 (Qwen3.6, blocked) |
| (new) | `qwen3.5/35b-a3b/` | New MoE fine-tune |

## Data File Prefix Renaming

| Old prefix | New prefix | Used in |
|------------|-----------|---------|
| `v1.2_sft_train` | `v1.1_sft_train` | build_training_mix.py, train_unsloth.py, pretokenize_for_sft.py |
| `v1.2_pretokenized` | `v1.1_pretokenized` | train_unsloth.py, pretokenize_for_sft.py |
| `v1.2_pruned` | `v1.1_pruned` | filter_for_v12_pruned.py, build_training_mix.py |
| `v1.3_sft_train` | `v1.0_sft_train` | build_training_mix.py, train_unsloth.py |
| `v1.3_pretokenized` | `v1.0_pretokenized` | train_unsloth.py, pretokenize_for_sft.py |
| `v1.3_pruned` | `v1.0_pruned` | filter_for_v12_pruned.py, build_training_mix.py |

## What Does NOT Change

- **HuggingFace repo names** — `danielcherubini/Qwen3.5-DeltaCoder-9B`, `danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF`, etc. These are published artifacts.
- **Model names** — `Qwen/Qwen3.5-9B`, `Qwen/Qwen3.6-9B`, etc.
- **Output artifact names** — `DeltaCoder-9B-v1.2-DPO-*.gguf` filenames in HF repos.
- **Training output dirs** — `./outputs/deltacoder-9b-*` are local ephemeral dirs on Vast.ai, not repo paths. Update the version suffix in the constant but the structure is fine.

---

### Task 1: Move all folders with `git mv`

**Context:**
This is the foundation of the restructure. We move all three version directories into the new model-based hierarchy using `git mv`, which preserves git blame and history. No file contents are changed in this commit — it's purely structural. This must be done first because all subsequent tasks reference the new paths.

**Files:**
- Move: `v1.1/` → `qwen3.5/v1.0/`
- Move: `v1.2/` → `qwen3.5/v1.1/`
- Move: `v1.3/` → `qwen3.6/v1.0/`

**What to implement:**

```bash
# Create parent directories
mkdir -p qwen3.5 qwen3.6

# Move with git mv (preserves history)
git mv v1.1 qwen3.5/v1.0
git mv v1.2 qwen3.5/v1.1
git mv v1.3 qwen3.6/v1.0
```

Do NOT change any file contents in this commit. Only move files.

**Steps:**
- [ ] Run `mkdir -p qwen3.5 qwen3.6`
- [ ] Run `git mv v1.1 qwen3.5/v1.0`
- [ ] Run `git mv v1.2 qwen3.5/v1.1`
- [ ] Run `git mv v1.3 qwen3.6/v1.0`
- [ ] Run `git status` to verify all 74 files show as "renamed"
- [ ] Verify no files remain under `v1.1/`, `v1.2/`, `v1.3/`
- [ ] Commit with message: `Restructure repo: v1.1→qwen3.5/v1.0, v1.2→qwen3.5/v1.1, v1.3→qwen3.6/v1.0`

**Acceptance criteria:**
- [ ] `ls qwen3.5/` shows `v1.0/` and `v1.1/`
- [ ] `ls qwen3.6/` shows `v1.0/`
- [ ] `v1.1/`, `v1.2/`, `v1.3/` directories no longer exist
- [ ] `git status` shows only renames, no deletions or additions
- [ ] `git log --follow qwen3.5/v1.1/scripts/train_unsloth.py` shows history from before the move

---

### Task 2: Update `.gitignore` for new paths

**Context:**
The `.gitignore` has 14 version-specific patterns that exclude large artifacts (data, adapters, merged models, logs). These must be updated to match the new folder structure. This is a prerequisite for all other tasks since changed files in ignored directories won't show up in `git status`.

**Files:**
- Modify: `.gitignore`

**What to implement:**

Replace the entire version-specific section of `.gitignore`. The current content (lines 7-25):

```
# v1.1 large artifacts
v1.1/data/
v1.1/outputs/
v1.1/logs/

# v1.2 large artifacts (adapter weights, merged model, training data)
v1.2/lora_adapter/
v1.2/dpo_adapter/
v1.2/dpo_checkpoints/
v1.2/merged/
v1.2/data/
v1.2/logs/
v1.2/*.log

# v1.3 large artifacts
v1.3/lora_adapter/
v1.3/merged/
v1.3/data/
v1.3/*.log
```

Replace with:

```
# qwen3.5/v1.0 (was v1.1) large artifacts
qwen3.5/v1.0/data/
qwen3.5/v1.0/outputs/
qwen3.5/v1.0/logs/

# qwen3.5/v1.1 (was v1.2) large artifacts
qwen3.5/v1.1/lora_adapter/
qwen3.5/v1.1/dpo_adapter/
qwen3.5/v1.1/dpo_checkpoints/
qwen3.5/v1.1/merged/
qwen3.5/v1.1/data/
qwen3.5/v1.1/logs/
qwen3.5/v1.1/*.log

# qwen3.5/35b-a3b large artifacts
qwen3.5/35b-a3b/data/
qwen3.5/35b-a3b/lora_adapter/
qwen3.5/35b-a3b/merged/
qwen3.5/35b-a3b/*.log

# qwen3.6/v1.0 (was v1.3) large artifacts
qwen3.6/v1.0/lora_adapter/
qwen3.6/v1.0/merged/
qwen3.6/v1.0/data/
qwen3.6/v1.0/*.log
```

**Steps:**
- [ ] Read `.gitignore`
- [ ] Replace lines 7-25 with the new patterns above
- [ ] Commit with message: `Update .gitignore for new folder structure`

**Acceptance criteria:**
- [ ] No references to `v1.1/`, `v1.2/`, or `v1.3/` remain in `.gitignore`
- [ ] `qwen3.5/35b-a3b/` patterns are included (for future use)
- [ ] `qwen3.6/v1.0/` patterns match what `v1.3/` had

---

### Task 3: Update all `qwen3.5/v1.1/` scripts (was `v1.2/`)

**Context:**
This is the largest task. The 38 scripts under `qwen3.5/v1.1/scripts/` (formerly `v1.2/scripts/`) contain hardcoded default paths, version strings in docstrings/comments, data file prefixes, GitHub raw URLs, and output directory constants. All references to `v1.2` must become `v1.1`, and all path prefixes `v1.2/` must become `qwen3.5/v1.1/`. Data file prefixes like `v1.2_sft_train` become `v1.1_sft_train`.

HuggingFace repo names (`danielcherubini/Qwen3.5-DeltaCoder-9B`, etc.) do NOT change — they are published artifacts.

**Files:**
- Modify: All 38 files under `qwen3.5/v1.1/scripts/`
- Modify: `qwen3.5/v1.1/chat_template.jinja` (if it contains version references — check first)

**What to implement:**

For EVERY `.py`, `.sh`, and `.bat` file under `qwen3.5/v1.1/scripts/`, apply these replacements:

**Path replacements (order matters — do longer strings first):**
| Find | Replace |
|------|---------|
| `v1.2/data/v1.2_pruned` | `qwen3.5/v1.1/data/v1.1_pruned` |
| `v1.2/data/v1.2_sft_train_pruned.jsonl` | `qwen3.5/v1.1/data/v1.1_sft_train_pruned.jsonl` |
| `v1.2/data/v1.2_sft_train.jsonl` | `qwen3.5/v1.1/data/v1.1_sft_train.jsonl` |
| `v1.2/data` | `qwen3.5/v1.1/data` |
| `v1.2/scripts/` | `qwen3.5/v1.1/scripts/` |
| `v1.2/gguf` | `qwen3.5/v1.1/gguf` |
| `v1.2/merged` | `qwen3.5/v1.1/merged` |

**Data file prefix replacements (in CLI defaults, docstrings, comments):**
| Find | Replace |
|------|---------|
| `v1.2_sft_train` | `v1.1_sft_train` |
| `v1.2_pretokenized` | `v1.1_pretokenized` |
| `v1.2_pruned` | `v1.1_pruned` |
| `data/v1.2_sft_train` | `data/v1.1_sft_train` |

**Version string replacements in docstrings/comments/print statements:**
| Find | Replace |
|------|---------|
| `DeltaCoder v1.2` | `DeltaCoder Qwen3.5 v1.1` |
| `deltacoder-9b-v1.2` | `deltacoder-9b-v1.1` |
| `DeltaCoder-9B-v1.2` | `DeltaCoder-9B-v1.1` |
| `v1.2 SFT` | `v1.1 SFT` |
| `v1.2 DPO` | `v1.1 DPO` |
| `pruned v1.2 mix` | `pruned v1.1 mix` |
| `pruned v1.2` | `pruned v1.1` |
| `for v1.2` | `for Qwen3.5 v1.1` |

**GitHub raw URL replacements (in provision.sh only):**
| Find | Replace |
|------|---------|
| `DeltaCoder/main/v1.2/scripts/` | `DeltaCoder/main/qwen3.5/v1.1/scripts/` |

**Output directory constants:**
| Find (in Python) | Replace |
|-------------------|---------|
| `OUTPUT_DIR = "./outputs/deltacoder-9b-v1.2"` | `OUTPUT_DIR = "./outputs/deltacoder-9b-v1.1"` |

**Files that need special attention:**

1. `provision.sh` — Has 3 GitHub raw URLs pointing to `v1.2/scripts/*`. Update to `qwen3.5/v1.1/scripts/*`.
2. `build_training_mix.py` — Has multiple default paths and version strings in the docstring.
3. `filter_for_v12_pruned.py` — Filename mentions "v12" but we keep the filename as-is (it's a filter script, the name is fine). Update ONLY the internal paths and strings.
4. `train_unsloth.py` — Has docstring examples with `/workspace/v1.2_*` paths. Update prefix to `v1.1_*`.
5. `pretokenize_for_sft.py` — Has docstring examples with `v1.2_*` paths.
6. `upload_all.py` — Has Windows paths `D:/AI/DeltaCoder/v1.2/*`. Update to `D:/AI/DeltaCoder/qwen3.5/v1.1/*`.
7. `quantize_all.bat` — Has Windows paths `D:\AI\DeltaCoder\v1.2\*`. Update to `D:\AI\DeltaCoder\qwen3.5\v1.1\*`.
8. `restore_v11_adapter.py` — References `v1.1/outputs/`. Update to `qwen3.5/v1.0/outputs/`.
9. `merge_and_export_dpo.py` — Has default output paths with `v1.2` and `v1.3` in examples. Update both.
10. `delete_v12_ggufs.py` / `delete_v12_safetensors.py` — These filter HF files by `"v1.2"` string. Do NOT change the HF filter — the files on HuggingFace still contain "v1.2". Only update docstrings.
11. `merge_datasets.py` — References `data/v1.2_sft_train.jsonl`.

**Files with NO version references (verify, don't modify if clean):**
- `patch_vlm_packing.py`, `patch_causal_conv1d.py`, `fix_merge.py` — likely have no version paths
- `preprocess_*.py` (the old ones like `preprocess_magicoder.py`, `preprocess_xlam.py`, etc.) — most say "for DeltaCoder v1.2 SFT" in docstring
- `preprocess_competitive_programming.py`, `preprocess_qwen3_coder_distill.py` — have `v1.2/data` default paths

**Steps:**
- [ ] Read every `.py`, `.sh`, `.bat` file under `qwen3.5/v1.1/scripts/` and `qwen3.5/v1.1/chat_template.jinja`
- [ ] Apply ALL replacements listed above, file by file
- [ ] Run `grep -r 'v1\.2' qwen3.5/v1.1/` to verify NO references to `v1.2` remain (except in `delete_v12_ggufs.py` and `delete_v12_safetensors.py` where `"v1.2"` is an HF filename filter — those are correct)
- [ ] Commit with message: `Update qwen3.5/v1.1 scripts: v1.2 → v1.1 paths, prefixes, and version strings`

**Acceptance criteria:**
- [ ] `grep -r 'v1\.2/' qwen3.5/v1.1/` returns zero results (no folder path references)
- [ ] `grep -r 'v1\.2_' qwen3.5/v1.1/` returns zero results (no data file prefix references) — except `delete_v12_*.py` HF filters
- [ ] `grep -r 'deltacoder-9b-v1.2' qwen3.5/v1.1/` returns zero results
- [ ] `provision.sh` GitHub raw URLs point to `qwen3.5/v1.1/scripts/`
- [ ] `build_training_mix.py` defaults use `qwen3.5/v1.1/data/`

---

### Task 4: Update all `qwen3.6/v1.0/` scripts (was `v1.3/`)

**Context:**
The 29 scripts under `qwen3.6/v1.0/scripts/` (formerly `v1.3/scripts/`) need the same treatment as Task 3 but for the v1.3 → v1.0 renumber. Note that some v1.3 scripts have stale references to "v1.2" in their docstrings (they were copied from v1.2 and some docstrings weren't updated). Fix those to reference "Qwen3.6 v1.0" instead.

**Files:**
- Modify: All 29 files under `qwen3.6/v1.0/scripts/`
- Modify: `qwen3.6/v1.0/configs/.gitkeep` (no changes expected, but verify)

**What to implement:**

For EVERY file under `qwen3.6/v1.0/scripts/`, apply these replacements:

**Path replacements:**
| Find | Replace |
|------|---------|
| `v1.3/data/v1.3_pruned` | `qwen3.6/v1.0/data/v1.0_pruned` |
| `v1.3/data/v1.3_sft_train_pruned.jsonl` | `qwen3.6/v1.0/data/v1.0_sft_train_pruned.jsonl` |
| `v1.3/data/v1.3_sft_train.jsonl` | `qwen3.6/v1.0/data/v1.0_sft_train.jsonl` |
| `v1.3/data` | `qwen3.6/v1.0/data` |
| `v1.3/scripts/` | `qwen3.6/v1.0/scripts/` |
| `v1.3/gguf` | `qwen3.6/v1.0/gguf` |
| `v1.3/merged` | `qwen3.6/v1.0/merged` |

**Data file prefix replacements:**
| Find | Replace |
|------|---------|
| `v1.3_sft_train` | `v1.0_sft_train` |
| `v1.3_pretokenized` | `v1.0_pretokenized` |
| `v1.3_pruned` | `v1.0_pruned` |

**Version string replacements:**
| Find | Replace |
|------|---------|
| `DeltaCoder v1.3` | `DeltaCoder Qwen3.6 v1.0` |
| `deltacoder-9b-v1.3` | `deltacoder-9b-qwen3.6-v1.0` |
| `DeltaCoder-9B-v1.3` | `DeltaCoder-9B-Qwen3.6-v1.0` |
| `for v1.3` | `for Qwen3.6 v1.0` |
| `pruned v1.3` | `pruned Qwen3.6 v1.0` |

**ALSO fix stale v1.2 references in v1.3 scripts.** Several preprocess scripts were copied from v1.2 and still say "for DeltaCoder v1.2 SFT" in their docstrings. These must become "for DeltaCoder Qwen3.6 v1.0 SFT". Specifically check:
- `preprocess_magicoder.py`, `preprocess_xlam.py`, `preprocess_reasoning.py`, `preprocess_opencoder_reasoning.py`, `preprocess_hermes.py`, `preprocess_glaive.py`, `preprocess_code_feedback.py` — all say "v1.2 SFT" in docstring
- `merge_datasets.py` — references `data/v1.2_sft_train.jsonl`
- `quantize_all.bat` — references `D:\AI\DeltaCoder\v1.2\gguf\*`

Replace any remaining `v1.2` references in these files with the correct Qwen3.6 v1.0 equivalents.

**GitHub raw URL replacements (in provision.sh):**
| Find | Replace |
|------|---------|
| `DeltaCoder/main/v1.3/scripts/` | `DeltaCoder/main/qwen3.6/v1.0/scripts/` |

**Steps:**
- [ ] Read every file under `qwen3.6/v1.0/scripts/`
- [ ] Apply ALL replacements listed above
- [ ] Run `grep -r 'v1\.3' qwen3.6/v1.0/` to verify no references remain
- [ ] Run `grep -r 'v1\.2' qwen3.6/v1.0/` to verify no stale v1.2 references remain
- [ ] Commit with message: `Update qwen3.6/v1.0 scripts: v1.3 → v1.0 paths, fix stale v1.2 references`

**Acceptance criteria:**
- [ ] `grep -r 'v1\.3' qwen3.6/v1.0/` returns zero results
- [ ] `grep -r 'v1\.2' qwen3.6/v1.0/` returns zero results
- [ ] `grep -r 'v1\.1/' qwen3.6/v1.0/` returns zero results (no cross-contamination)
- [ ] `provision.sh` GitHub raw URLs point to `qwen3.6/v1.0/scripts/`

---

### Task 5: Update `qwen3.5/v1.0/` scripts (was `v1.1/`)

**Context:**
The 7 files under `qwen3.5/v1.0/` (formerly `v1.1/`) are the original messy v1.1 scripts. Confusingly, they ALREADY reference "v1.2" in many places (they were written during what was called v1.2 at the time). These need to be updated to "v1.0" since this is now qwen3.5/v1.0.

**Files:**
- Modify: `qwen3.5/v1.0/scripts/train_unsloth.py`
- Modify: `qwen3.5/v1.0/scripts/train_unsloth1.py`
- Check (likely no changes): `qwen3.5/v1.0/scripts/dry_run.py`, `generate_dpo_pairs-single.py`, `merge_and_export.sh`
- Check: `qwen3.5/v1.0/configs/deltacoder-9b-dpo.yaml`, `deltacoder-9b-lora.yaml`

**What to implement:**

In `train_unsloth.py`:
| Find | Replace |
|------|---------|
| `v1.2 — Unsloth LoRA SFT` | `Qwen3.5 v1.0 — Unsloth LoRA SFT` |
| `OUTPUT_DIR = "./outputs/deltacoder-9b-v1.2"` | `OUTPUT_DIR = "./outputs/deltacoder-9b-v1.0"` |
| `DeltaCoder v1.2 SFT training` | `DeltaCoder Qwen3.5 v1.0 SFT training` |
| `data/v1.2_sft_train.jsonl` | `data/v1.0_sft_train.jsonl` |

In `train_unsloth1.py` — check for any version references and update similarly.

Read all other files and update any `v1.1` or `v1.2` references to `v1.0`.

**Steps:**
- [ ] Read all 7 files under `qwen3.5/v1.0/`
- [ ] Apply replacements
- [ ] Run `grep -r 'v1\.1\|v1\.2' qwen3.5/v1.0/` to verify no old references remain
- [ ] Commit with message: `Update qwen3.5/v1.0 scripts: fix version references to v1.0`

**Acceptance criteria:**
- [ ] `grep -r 'v1\.2' qwen3.5/v1.0/` returns zero results
- [ ] `grep -r 'v1\.1/' qwen3.5/v1.0/` returns zero results (path references)
- [ ] OUTPUT_DIR references `v1.0`

---

### Task 6: Rewrite AGENTS.md for new structure

**Context:**
AGENTS.md is the single source of truth for the entire project. It has ~70 references to old paths scattered across 12 sections. This needs a complete update — not just find-and-replace, but restructuring sections to reflect the new hierarchy. The section numbering and content structure should reflect that Qwen3.5 and Qwen3.6 are separate model families.

**Files:**
- Modify: `AGENTS.md`

**What to implement:**

**Section 1 (Project Overview):** Update bullet points:
- `v1.2` → `qwen3.5/v1.1`
- `v1.3` → `qwen3.6/v1.0`

**Section 2 (Repository Structure):** Replace the entire directory tree with:
```
DeltaCoder/
├── qwen3.5/
│   ├── v1.0/              # Original 9B (SFT + DPO, Axolotl configs)
│   │   ├── configs/
│   │   └── scripts/
│   ├── v1.1/              # Revised 9B (Jackrong-inspired, Unsloth + packing)
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── data/          # Training data + preprocessed datasets (gitignored)
│   │   ├── lora_adapter/  # SFT LoRA adapter (gitignored)
│   │   └── merged/        # Merged SFT model (gitignored)
│   └── 35b-a3b/           # MoE fine-tune (NEW)
│       └── scripts/
├── qwen3.6/
│   └── v1.0/              # Qwen3.6 (BLOCKED — waiting for open weights)
│       ├── configs/
│       ├── scripts/
│       └── data/          # Training data (gitignored)
├── docs/
├── AGENTS.md
└── README.md
```

**Section 3 (Vast.ai Infrastructure):** Update all paths:
- `v1.2/scripts/provision.sh` → `qwen3.5/v1.1/scripts/provision.sh`
- GitHub raw URL: `main/v1.2/scripts/` → `main/qwen3.5/v1.1/scripts/`
- `v1.2/data/v1.2_pretokenized` → `qwen3.5/v1.1/data/v1.1_pretokenized`
- `/workspace/merged_v1.2` → `/workspace/merged_v1.1`

**Section 6 (Key Scripts):** Update entire table — change every `v1.2/scripts/` to `qwen3.5/v1.1/scripts/` and every `v1.3/scripts/` to `qwen3.6/v1.0/scripts/`.

**Section 8 (HuggingFace Repos):** Keep repo names but update descriptions:
- `v1.1/v1.2 DPO adapter` → `Qwen3.5 v1.0/v1.1 DPO adapter`
- Add placeholder for 35B-A3B repos

**Section 9 (v1.3 Structure):** Rename to "Qwen3.6 v1.0 Structure", update all paths from `v1.3/` to `qwen3.6/v1.0/`.

**Section 10 (Quick Commands):** Update all `v1.2/scripts/` and `v1.3/scripts/` paths. Update data file prefixes (`v1.2_*` → `v1.1_*`, `v1.3_*` → `v1.0_*`). Update `/workspace/merged_v1.2` → `/workspace/merged_v1.1`.

**Section 11 (Training Dataset & Strategy):** Mostly text, minimal path references. Update any `v1.2` → `v1.1` references.

**Section 12 (Key Discoveries):** Update `/workspace/merged_v1.2` → `/workspace/merged_v1.1` in the vLLM config wrapping example. Update any script paths.

**Steps:**
- [ ] Read the full AGENTS.md
- [ ] Apply all replacements section by section as described above
- [ ] Run `grep -n 'v1\.1/' AGENTS.md` — should only appear in the context of `qwen3.5/v1.0` descriptions or historical notes
- [ ] Run `grep -n 'v1\.2/' AGENTS.md` — should return ZERO results
- [ ] Run `grep -n 'v1\.3/' AGENTS.md` — should return ZERO results
- [ ] Verify the directory tree in Section 2 matches the actual repo structure
- [ ] Commit with message: `Rewrite AGENTS.md for model-based folder structure`

**Acceptance criteria:**
- [ ] Zero references to `v1.2/` or `v1.3/` as path prefixes
- [ ] Directory tree in Section 2 shows `qwen3.5/` and `qwen3.6/` hierarchy
- [ ] All script paths in Key Scripts table use new paths
- [ ] Quick Commands section uses new paths and data prefixes

---

### Task 7: Update plan docs

**Context:**
There are 11 plan docs under `docs/plans/`. Three are "active" (still referenced, may be executed) and eight are "historical" (completed or superseded). Active plans get full path updates. Historical plans get a deprecation header noting the paths are pre-restructure.

**Files:**
- Modify (full update): `docs/plans/2026-04-05-deltacoder-35b-a3b.md`
- Modify (full update): `docs/plans/2026-04-01-v1.2-redo.md`
- Modify (full update): `docs/plans/2026-04-02-v1.3-qwen3.6-plan.md`
- Modify (header only): `docs/plans/2026-04-02-v1.2-dataset-pruning.md`
- Modify (header only): `docs/plans/2026-03-28-v1.2-phase1-preprocessing.md`
- Modify (header only): `docs/plans/2026-03-28-v1.2-phase2-sft-training.md`
- Modify (header only): `docs/plans/2026-03-28-v1.2-phase3-dpo-and-export.md`
- Modify (header only): `docs/plans/2026-03-30-v1.3-phase1-config-and-pretokenize.md`
- Modify (header only): `docs/plans/2026-03-30-v1.3-phase2-sft-training.md`
- Modify (header only): `docs/plans/2026-03-30-v1.3-phase3-dpo-and-export.md`
- Modify (header only): `docs/plans/2026-03-26-deltacoder-v1.1-dpo.md`

**What to implement:**

**For the 3 active plan docs**, apply the same path/version replacements as in Tasks 3-4:

In `2026-04-05-deltacoder-35b-a3b.md`:
- `v1.2/scripts/` → `qwen3.5/v1.1/scripts/` (BUT the 35B scripts themselves should go under `qwen3.5/35b-a3b/scripts/`)
- `v1.2/data/` → `qwen3.5/v1.1/data/`
- `v1.2_pruned` → `v1.1_pruned`
- `v1.2_35b_sft.jsonl` → `v1.1_35b_sft.jsonl` (BUT this file should actually be under `qwen3.5/35b-a3b/data/`)
- All task file paths that say `v1.2/scripts/build_35b_subset.py` → `qwen3.5/35b-a3b/scripts/build_35b_subset.py`
- All task file paths that say `v1.2/scripts/train_35b_a3b.py` → `qwen3.5/35b-a3b/scripts/train_35b_a3b.py`
- All task file paths that say `v1.2/scripts/provision_35b.sh` → `qwen3.5/35b-a3b/scripts/provision.sh` (rename to just `provision.sh` since it's in its own dir)
- All task file paths that say `v1.2/scripts/merge_and_export_35b.py` → `qwen3.5/35b-a3b/scripts/merge_and_export.py` (simpler name in own dir)

In `2026-04-01-v1.2-redo.md`:
- `v1.2/` → `qwen3.5/v1.1/` throughout
- `v1.2_*` data prefixes → `v1.1_*`
- `/workspace/merged_v1.2` → `/workspace/merged_v1.1`

In `2026-04-02-v1.3-qwen3.6-plan.md`:
- `v1.3/` → `qwen3.6/v1.0/` throughout
- `v1.3_*` data prefixes → `v1.0_*`

**For the 8 historical plan docs**, add this header after the title line:

```markdown
> **Note:** This plan uses pre-restructure paths (`v1.1/`, `v1.2/`, `v1.3/`).
> The repo was restructured on 2026-04-06: `v1.1/` → `qwen3.5/v1.0/`, `v1.2/` → `qwen3.5/v1.1/`, `v1.3/` → `qwen3.6/v1.0/`.
```

Do NOT rewrite the historical plan contents — they are historical records.

**Steps:**
- [ ] Read all 11 plan files
- [ ] Apply full updates to the 3 active plans
- [ ] Add deprecation header to the 8 historical plans
- [ ] Run `grep -r 'v1\.2/' docs/plans/2026-04-05*` — should return zero
- [ ] Run `grep -r 'v1\.2/' docs/plans/2026-04-01*` — should return zero
- [ ] Run `grep -r 'v1\.3/' docs/plans/2026-04-02-v1.3*` — should return zero
- [ ] Commit with message: `Update plan docs for repo restructure, add deprecation headers to historical plans`

**Acceptance criteria:**
- [ ] 3 active plans have zero references to old path prefixes (`v1.2/`, `v1.3/`)
- [ ] 8 historical plans have deprecation header
- [ ] 35B-A3B plan references `qwen3.5/35b-a3b/scripts/` (not `v1.2/scripts/`)

---

## Verification (after all tasks complete)

Run these checks to confirm the restructure is complete:

```bash
# No old path prefixes anywhere in tracked files (except historical plan docs with deprecation headers)
git ls-files | xargs grep -l 'v1\.1/' | grep -v 'docs/plans/2026-03'
# Should return ZERO files (or only historical plan docs)

git ls-files | xargs grep -l 'v1\.2/' | grep -v 'docs/plans/2026-03' | grep -v 'delete_v12'
# Should return ZERO files

git ls-files | xargs grep -l 'v1\.3/' | grep -v 'docs/plans/2026-03'
# Should return ZERO files

# New structure exists
ls qwen3.5/v1.0/ qwen3.5/v1.1/ qwen3.6/v1.0/

# Old structure is gone
ls v1.1/ v1.2/ v1.3/  # Should all fail with "No such file or directory"
```
