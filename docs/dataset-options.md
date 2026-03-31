# DeltaCoder — Dataset Options for v1.3+

**Created:** 2026-03-31
**Context:** Researching datasets to expand beyond v1.2's 230K rows. OmniCoder (Tesslate) trained on 425K agentic trajectories and hit 23.6% on Terminal-Bench 2. Our v1.1-DPO sits at ~12% on the full benchmark.

---

## Current v1.2 Dataset Mix (230K rows)

| Dataset | Rows | Focus |
|---|---|---|
| nvidia/OpenCodeReasoning (OCR) | 65K | Reasoning traces (avg 8,477 tokens) |
| togethercomputer/CoderForge | 50K | Multi-turn agentic coding trajectories |
| Code-Feedback | 50K | Instruction following / code feedback |
| Magicoder | 50K | Code generation |
| Salesforce/xlam | 15K | Tool/function calling |
| **Total** | **230K** | |

---

## Available Datasets on HuggingFace

### Top Priority — Direct Terminal/Agent Trajectories

| Dataset | Rows | Description | License | Relevance |
|---|---|---|---|---|
| **nvidia/Nemotron-Terminal-Corpus** | 366K | NVIDIA terminal trajectories | Open | Directly trains terminal agent behavior |
| **nvidia/Nemotron-Cascade-SFT-SWE** | 141K | NVIDIA SWE agent trajectories | Open | Software engineering agent traces |
| **nebius/SWE-agent-trajectories** | 80K | SWE-agent on real GitHub issues. 13K resolved (`target=true`), 67K failed. | CC-BY-4.0 | Filter to target=true for 13K high-quality solved trajectories |
| **nebius/SWE-rebench-openhands-trajectories** | 67K | Qwen3-Coder OpenHands trajectories | Open | Agent coding traces |
| **m-a-p/TerminalTraj** | 20K | Terminal agent trajectories | Open | Directly relevant to Terminal-Bench |
| **Aznaur/terminal-agent-sft-data-v2** | 2.4K | Terminal-Bench tasks, pre-tokenized for Qwen3.5! | Open | Literally Terminal-Bench training data |

### High Value — Agentic/Tool-Use

| Dataset | Rows | Description | License |
|---|---|---|---|
| **nvidia/Nemotron-SFT-Agentic-v2** | 992K | Tool-calling (707K), customer service (279K), search (7K) | CC-BY-4.0 |
| **Agent-Ark/Toucan-1.5M** | 1.65M | Massive tool-use trajectories | Open |
| **LogicStar/SWE-Star** | 244K | SWE-Star agent traces (7B-32B models trained on this hit 50.2% SWE-bench verified) | Open |
| **Nanbeige/ToolMind** | 369K | Tool-use training data | Open |
| **allenai/Dolci-Instruct-SFT-Tool-Use** | 228K | Allen AI tool-use SFT | Open |
| **open-thoughts/OpenThoughts-Agent-v1-SFT** | 15K | Agent reasoning + SFT | Open |
| **ricdomolm/mini-coder-trajs-400k** | 397K | Massive coding agent traces | Open |

### Worth Exploring

| Dataset | Rows | Description |
|---|---|---|
| **SWE-bench/SWE-smith-trajectories** | 76K | 3 formats (tool/xml/ticks), used to train Qwen 2.5 Coder |
| **Kwai-Klear/SWE-smith-mini** | 66K | SWE-agent+ trajectories |
| **miromind-ai/MiroVerse-v0.1** | 228K | Agent trajectories |
| **zai-org/SWE-Dev-train** | 20K | SWE development training |
| **jupyter-agent/jupyter-agent-dataset** | 96K | Jupyter notebook agent traces |
| **nvidia/Nemotron-SFT-OpenCode-v1** | 639 | Small but NVIDIA quality, OpenCode traces |
| **SWE-Lego/SWE-Lego-Real-Data** | 18K | Real SWE data |
| **SWE-Lego/SWE-Lego-Synthetic-Data** | 11.5K | Synthetic SWE data |
| **allenai/Sera-4.6-Lite-T2** | 25.4K | Allen AI agent traces |
| **GAIR/daVinci-Dev** | 4.94M | Massive development dataset |
| **Bingguang/HardGen** | 17K | Hard generation tasks |
| **AweAI-Team/Scale-SWE-Distilled** | 71.5K | Distilled SWE trajectories |

### Gated/Commercial — Contact Required

| Dataset | Rows | Description | Access |
|---|---|---|---|
| **zenlm/zen-agentic-dataset** | 3.5M (10.5B tokens) | Real Claude Code sessions, git history, debug workflows | Commercial — contact z@hanzo.ai |

### Reference: casey-martin's Agent Trajectories Collection
Full curated list at: https://huggingface.co/collections/casey-martin/agent-trajectories

---

## Proposed Dataset Mixes

### Conservative Mix: ~430K rows (match OmniCoder scale)

| Source | Take | Running Total |
|---|---|---|
| Current v1.2 mix | 230K | 230K |
| nvidia/Nemotron-Terminal-Corpus (sample) | 50K | 280K |
| nebius/SWE-agent-trajectories (target=true) | 13K | 293K |
| m-a-p/TerminalTraj | 20K | 313K |
| Aznaur/terminal-agent-sft-data-v2 | 2.4K | 315K |
| LogicStar/SWE-Star (sample) | 50K | 365K |
| open-thoughts/OpenThoughts-Agent-v1-SFT | 15K | 380K |
| nvidia/Nemotron-SFT-Agentic-v2 tool_calling (sample) | 50K | **430K** |

### Aggressive Mix: ~600K rows

Add on top of conservative:
| Source | Take | Running Total |
|---|---|---|
| Conservative mix | 430K | 430K |
| ricdomolm/mini-coder-trajs-400k (sample) | 100K | 530K |
| Agent-Ark/Toucan-1.5M (sample) | 70K | **600K** |

---

## Training Time Estimates (seq_len=32768)

### Token Budget

| Mix | Rows | Est total tokens |
|---|---|---|
| v1.2 (current, same data) | 230K | ~760M |
| Conservative | 430K | ~1,908M |
| Aggressive | 600K | ~2,758M |

### Training Time by Hardware

Per-step token throughput: `micro_batch(1) x ga x seq_len(32768)` = 131,072 tokens per optimizer step (effective batch = 4).

Step time at 32768: estimated ~25-30s (GDN is linear, 8/32 attention layers are quadratic, gradient checkpointing on).

| Mix | Steps | 1x H200 | 2x H200 DDP | 4x H200 DDP | Est Cost |
|---|---|---|---|---|---|
| v1.3 same data (230K) | ~5,800 | ~50-55hrs | ~25-28hrs | ~13-15hrs | ~$100-120 |
| Conservative (430K) | ~14,560 | ~101hrs (4.2d) | ~51hrs (2.1d) | ~25hrs (1d) | ~$200 |
| Aggressive (600K) | ~21,040 | ~146hrs (6d) | ~73hrs (3d) | ~36hrs (1.5d) | ~$290 |

**Multi-GPU is mandatory for the expanded datasets.** 4x H200 DDP makes the conservative mix a 1-day run.

---

## Important Notes

- **GGUF size is NOT affected by dataset size.** The model has 9B parameters regardless of training data. Q4_K_M stays ~5.5GB whether trained on 230K or 6M rows.
- **Aznaur/terminal-agent-sft-data-v2** is pre-tokenized for Qwen3.5 — may need de-tokenizing to convert to our messages format, or use directly with custom pretokenize script.
- **nebius/SWE-agent-trajectories** has a `target` boolean — filter to `target=true` for only successfully resolved trajectories (13K out of 80K).
- **nvidia/Nemotron-SFT-Agentic-v2** has 3 splits: tool_calling (707K), search (7K), customer_service (279K). The tool_calling split is most relevant.
- **OmniCoder reference:** 425K rows, 4x H200 DDP, Axolotl, LoRA r=64 alpha=32, lr=2e-4, SFT-only → 23.6% Terminal-Bench 2.

---

## Preprocessing Requirements

Each dataset needs a conversion script to normalize to our `{"messages": [...]}` JSONL format:
- Most SWE/agent datasets use different role names or formats
- Tool calls need normalizing (same as xlam preprocessing)
- Some datasets are pre-tokenized (Aznaur) — need de-tokenizing or separate handling
- Filter for quality: only successful trajectories, no loops, no truncated conversations
