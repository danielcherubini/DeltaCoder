"""
Merge all preprocessed datasets into a single training JSONL.

Combines:
  - data/coderforge_converted.jsonl    (~155K rows)
  - data/nemotron_swe_converted.jsonl  (~51K rows)
  - data/nemotron_agentic_converted.jsonl (~19K rows)
  - data/sweagent_converted.jsonl      (~13K rows)

Shuffles the result and writes to data/train.jsonl.
"""

import json
import random
import sys
from pathlib import Path

INPUT_FILES = [
    "data/magicoder_converted.jsonl",
    "data/coderforge_converted.jsonl",
    "data/code_feedback_converted.jsonl",
    "data/hermes_converted.jsonl",
    "data/glaive_converted.jsonl",
    "data/opus_reasoning_converted.jsonl",
    "data/qwen35_reasoning_converted.jsonl",
]


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "data/v1.2_sft_train.jsonl"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for path in INPUT_FILES:
        if not Path(path).exists():
            print(f"  SKIP {path} (not found)")
            continue
        rows = load_jsonl(path)
        print(f"  Loaded {len(rows):>7,} rows from {path}")
        all_rows.extend(rows)

    print(f"\nTotal: {len(all_rows):,} rows")

    # Shuffle
    random.seed(seed)
    random.shuffle(all_rows)

    # Write
    with open(output, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Written to {output}")

    # Print source distribution
    sources = {}
    for row in all_rows:
        src = row.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\nSource distribution:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        pct = cnt / len(all_rows) * 100
        print(f"  {src:>20}: {cnt:>7,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
