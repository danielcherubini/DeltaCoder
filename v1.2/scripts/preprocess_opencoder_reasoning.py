"""
Preprocess nvidia/OpenCodeReasoning for DeltaCoder v1.2 SFT.

Input: nvidia/OpenCodeReasoning split_0 (567K rows, competitive coding with R1 reasoning)
Output: data/opencoder_reasoning_converted.jsonl

Schema: {"messages": [...], "source": "opencoder_reasoning", "id": "ocr-N"}

The `output` field contains R1's full response with <think> reasoning traces + code.
The `input` field contains the competitive programming question.
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

MAX_ROWS = 65_000
SEED = 42


def main():
    output = (
        sys.argv[1] if len(sys.argv) > 1 else "data/opencoder_reasoning_converted.jsonl"
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading nvidia/OpenCodeReasoning split_0...")
    ds = load_dataset("nvidia/OpenCodeReasoning", "split_0", split="split_0")

    # Verify schema
    print(f"Columns: {ds.column_names}")
    print(f"Total rows: {len(ds)}")
    sample = ds[0]
    print(f"Sample input (first 200 chars): {sample['input'][:200]}")
    print(f"Sample output (first 200 chars): {sample['output'][:200]}")

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_ROWS, len(ds))))
    print(f"Using {len(ds)} rows after shuffle+cap")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            question = (row.get("input") or "").strip()
            response = (row.get("output") or "").strip()

            if not question or not response or question == "-":
                skipped += 1
                continue

            # R1's output already contains <think>...</think> + code
            out = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ],
                "source": "opencoder_reasoning",
                "id": row.get("id", f"ocr-{i}"),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Done: {count} rows written, {skipped} skipped -> {output}")


if __name__ == "__main__":
    main()
