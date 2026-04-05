"""
Download and preprocess Jackrong/Competitive-Programming-python-blend for DeltaCoder v1.2.

Dataset is already in messages format with <think> blocks — minimal processing needed.
Just adds source tag and applies an 8K token length filter.

Dataset card: ~28K rows, apache-2.0/cc-by-4.0 licensed.
87.5% from Nemotron-SFT-Competitive-Programming-v2 (Python only).
Already contains <think>...</think> in assistant responses.

Usage:
    python preprocess_competitive_programming.py [--output-dir v1.2/data]
"""

import argparse
import json
import os
import sys

from datasets import load_dataset


# Token limit for Tier 1 (coding sources)
TOKEN_LIMIT = 8_192
CHARS_PER_TOKEN = 3.5

DATASET_NAME = "Jackrong/Competitive-Programming-python-blend"
OUTPUT_FILENAME = "competitive_programming_converted.jsonl"
TARGET_ROWS = 28_000


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count from total character length."""
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return int(total_chars / CHARS_PER_TOKEN)


def convert_row(row: dict) -> dict | None:
    """
    Convert a dataset row to DeltaCoder messages format.

    Dataset is already in messages format:
      {"id": "...", "messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    Just validates and adds source tag.
    """
    messages = row.get("messages", [])
    if not messages:
        return None

    # Validate: must have at least one user and one assistant message
    roles = {m.get("role") for m in messages}
    if "user" not in roles or "assistant" not in roles:
        return None

    # Apply 8K token filter
    if estimate_tokens(messages) > TOKEN_LIMIT:
        return None

    return {
        "messages": messages,
        "source": "competitive_programming",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Competitive-Programming-python-blend for v1.2"
    )
    parser.add_argument(
        "--output-dir", default="v1.2/data", help="Output directory for JSONL"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=TARGET_ROWS,
        help=f"Maximum rows to output (default: {TARGET_ROWS:,})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and count without writing output",
    )
    args = parser.parse_args()

    print(f"Downloading {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(ds)
    print(f"Loaded {total:,} rows")

    converted = []
    skipped_format = 0
    skipped_tokens = 0

    for row in ds:
        messages = row.get("messages", [])
        if not messages:
            skipped_format += 1
            continue

        roles = {m.get("role") for m in messages}
        if "user" not in roles or "assistant" not in roles:
            skipped_format += 1
            continue

        if estimate_tokens(messages) > TOKEN_LIMIT:
            skipped_tokens += 1
            continue

        converted.append(
            {
                "messages": messages,
                "source": "competitive_programming",
            }
        )

        if len(converted) >= args.max_rows:
            break

    print(f"\nResults:")
    print(f"  Total loaded:        {total:,}")
    print(f"  Skipped (format):    {skipped_format:,}")
    print(f"  Skipped (>{TOKEN_LIMIT:,} tokens): {skipped_tokens:,}")
    print(f"  Converted:           {len(converted):,}")

    if args.dry_run:
        print("\nDry run — not writing output.")
        return

    output_path = os.path.join(args.output_dir, OUTPUT_FILENAME)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row) + "\n")

    size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"\nWrote {len(converted):,} rows to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
