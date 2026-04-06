"""
Download and preprocess Jackrong/qwen3-coder-480b-distill-mini for DeltaCoder Qwen3.5 v1.1.

Converts from flat Input/code_output format to DeltaCoder messages format.
Applies an 8K token length filter.

Dataset: 9,543 rows, apache-2.0 licensed.
Distilled from Qwen3-Coder-480B-A35B-Instruct via rStar-Coder seeds.
Format: {"Input": "...", "code_output": "...", "generator": "...", "category": "code"}

Usage:
    python preprocess_qwen3_coder_distill.py [--output-dir qwen3.5/v1.1/data]
"""

import argparse
import json
import os
import sys

from datasets import load_dataset


# Token limit for Tier 1 (coding sources)
TOKEN_LIMIT = 8_192
CHARS_PER_TOKEN = 3.5

DATASET_NAME = "Jackrong/qwen3-coder-480b-distill-mini"
OUTPUT_FILENAME = "qwen3_coder_distill_converted.jsonl"
TARGET_ROWS = 9_500


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count from total character length."""
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return int(total_chars / CHARS_PER_TOKEN)


def convert_row(row: dict) -> dict | None:
    """
    Convert a dataset row to DeltaCoder messages format.

    Input format:
      {"Input": "...", "code_output": "...", "generator": "...", "category": "code"}

    Output format:
      {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
       "source": "qwen3_coder_distill"}
    """
    user_content = (row.get("Input") or "").strip()
    assistant_content = (row.get("code_output") or "").strip()

    if not user_content or not assistant_content:
        return None

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    if estimate_tokens(messages) > TOKEN_LIMIT:
        return None

    return {
        "messages": messages,
        "source": "qwen3_coder_distill",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess qwen3-coder-480b-distill-mini for Qwen3.5 v1.1"
    )
    parser.add_argument(
        "--output-dir", default="qwen3.5/v1.1/data", help="Output directory for JSONL"
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
    skipped_empty = 0
    skipped_tokens = 0

    for row in ds:
        user_content = (row.get("Input") or "").strip()
        assistant_content = (row.get("code_output") or "").strip()

        if not user_content or not assistant_content:
            skipped_empty += 1
            continue

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        if estimate_tokens(messages) > TOKEN_LIMIT:
            skipped_tokens += 1
            continue

        converted.append(
            {
                "messages": messages,
                "source": "qwen3_coder_distill",
            }
        )

        if len(converted) >= args.max_rows:
            break

    print(f"\nResults:")
    print(f"  Total loaded:        {total:,}")
    print(f"  Skipped (empty):     {skipped_empty:,}")
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
