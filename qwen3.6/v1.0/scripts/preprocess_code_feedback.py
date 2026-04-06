"""
Preprocess Code-Feedback for DeltaCoder Qwen3.6 v1.0 SFT.

Input: m-a-p/Code-Feedback (HuggingFace)
Output: data/code_feedback_converted.jsonl

Dataset is already in conversation format with role/content dicts.
We verify schema, add source/id fields, and cap at 50K.
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

MAX_ROWS = 50_000
SEED = 42


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "data/code_feedback_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading Code-Feedback...")
    ds = load_dataset("m-a-p/Code-Feedback", split="train")

    # Verify schema
    print(f"Columns: {ds.column_names}")
    sample = ds[0]
    print(f"Sample messages[0]: {sample['messages'][0]}")
    assert "messages" in ds.column_names, (
        f"Expected 'messages' column, got {ds.column_names}"
    )

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_ROWS, len(ds))))
    print(f"Using {len(ds)} rows")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            messages = row.get("messages", [])

            # Skip if fewer than 2 turns
            if not messages or len(messages) < 2:
                skipped += 1
                continue

            # Verify all messages have role/content
            valid = True
            for msg in messages:
                if "role" not in msg or "content" not in msg:
                    valid = False
                    break
                if not msg["content"] or not msg["content"].strip():
                    valid = False
                    break
            if not valid:
                skipped += 1
                continue

            out = {
                "messages": messages,
                "source": "code_feedback",
                "id": row.get("id", f"code_feedback-{i}"),
            }
            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Done: {count} rows written, {skipped} skipped → {output}")


if __name__ == "__main__":
    main()
