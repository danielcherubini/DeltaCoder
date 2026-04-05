"""
Preprocess reasoning datasets for DeltaCoder Qwen3.6 v1.0 SFT.

Input:
  - nohurry/Opus-4.6-Reasoning-3000x-filtered
  - Jackrong/Qwen3.5-reasoning-700x
Output:
  - data/opus_reasoning_converted.jsonl
  - data/qwen35_reasoning_converted.jsonl
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def process_opus(output: str):
    """Process Opus-4.6-Reasoning-3000x-filtered."""
    print("Loading Opus-4.6-Reasoning-3000x-filtered...")
    ds = load_dataset("nohurry/Opus-4.6-Reasoning-3000x-filtered", split="train")

    print(f"Columns: {ds.column_names}")
    print(f"Sample: problem={ds[0]['problem'][:100]}...")
    print(f"Rows: {len(ds)}")

    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            problem = (row.get("problem") or "").strip()
            thinking = (row.get("thinking") or "").strip()
            solution = (row.get("solution") or "").strip()

            if not problem or not solution:
                continue

            # Format: thinking + solution as assistant response
            if thinking:
                response = f"<think>\n{thinking}\n</think>\n\n{solution}"
            else:
                response = solution

            out = {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": response},
                ],
                "source": "opus_reasoning",
                "id": row.get("id", f"opus-{i}"),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Done: {count} rows → {output}")


def process_qwen35(output: str):
    """Process Qwen3.5-reasoning-700x."""
    print("Loading Qwen3.5-reasoning-700x...")
    ds = load_dataset("Jackrong/Qwen3.5-reasoning-700x", split="train")

    print(f"Columns: {ds.column_names}")
    print(f"Sample conversation[0]: {ds[0]['conversation'][0]}")
    print(f"Rows: {len(ds)}")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            conversation = row.get("conversation", [])
            if not conversation or len(conversation) < 2:
                skipped += 1
                continue

            # Convert from/value → role/content
            messages = []
            valid = True
            for turn in conversation:
                from_role = turn.get("from", "")
                value = turn.get("value", "")
                role = ROLE_MAP.get(from_role, from_role)
                if role not in ("user", "assistant", "system"):
                    print(f"  Unknown role '{from_role}' in row {i}")
                    valid = False
                    break
                messages.append({"role": role, "content": value})

            if not valid or len(messages) < 2:
                skipped += 1
                continue

            out = {
                "messages": messages,
                "source": "qwen35_reasoning",
                "id": row.get("id", f"qwen35-{i}"),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Done: {count} rows, {skipped} skipped → {output}")


def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    process_opus("data/opus_reasoning_converted.jsonl")
    process_qwen35("data/qwen35_reasoning_converted.jsonl")


if __name__ == "__main__":
    main()
