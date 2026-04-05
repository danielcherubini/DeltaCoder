"""
Preprocess Magicoder-OSS-Instruct-75K for DeltaCoder Qwen3.5 v1.1 SFT.

Input: ise-uiuc/Magicoder-OSS-Instruct-75K (HuggingFace)
Output: data/magicoder_converted.jsonl

Schema: {"messages": [...], "source": "magicoder", "id": "magicoder-N"}
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

MAX_ROWS = 50_000
SEED = 42


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "data/magicoder_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading Magicoder-OSS-Instruct-75K...")
    ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")

    # Verify schema
    print(f"Columns: {ds.column_names}")
    print(f"Sample row: {ds[0]}")
    assert "problem" in ds.column_names, (
        f"Expected 'problem' column, got {ds.column_names}"
    )
    assert "solution" in ds.column_names, (
        f"Expected 'solution' column, got {ds.column_names}"
    )

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_ROWS, len(ds))))
    print(f"Using {len(ds)} rows")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            problem = (row.get("problem") or "").strip()
            solution = (row.get("solution") or "").strip()

            if not problem or not solution:
                skipped += 1
                continue

            out = {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ],
                "source": "magicoder",
                "id": f"magicoder-{i}",
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Done: {count} rows written, {skipped} skipped → {output}")


if __name__ == "__main__":
    main()
