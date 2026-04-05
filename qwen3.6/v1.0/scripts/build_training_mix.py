"""
Build the DeltaCoder v1.3 training mix from preprocessed dataset files.

Combines all preprocessed JSONL files into a single shuffled training file.
Each source is capped to its target row count.

Jackrong-inspired mix (~157K rows, ~700M tokens, 2026-04-05):
  Philosophy: quality > quantity, tiered token limits (8K/16K)

  Tier 1 — ≤8K tokens (Coding + Tool Calling):
    nemotron_tool_calling:          ~40K  (top by tool call count)
    competitive_programming:        ~28K  (NEW — Jackrong blend, 87.5% Nemotron Python competitive coding)
    nemotron_agentic:               ~19K  (all kept — 99.1% naturally ≤8K)
    xlam:                           ~15K  (all kept)
    code_feedback:                  ~15K  (multi-turn ≥4 msgs)
    qwen3_coder_distill:            ~9.5K (NEW — distilled from Qwen3-Coder-480B via rStar-Coder)
    magicoder:                       ~5K  (top 5K by length)

  Tier 2 — ≤16K tokens (Agentic/SWE):
    opencoder_reasoning:            ~16K  (64.1% of 25K survive 16K filter)
    swesmith:                       ~10K  (48.9% of 20K survive 16K filter)

  Dropped entirely:
    nemotron_swe — 100% of rows exceed 16K tokens (median 43K), zero useful signal.

Previous mix (262K, unfiltered):
  Use --use-unfiltered to build from the original *_converted.jsonl files.
"""

import json
import random
import sys
from pathlib import Path

# Source configs: (filename, max_rows, source_label)

# Tier 1: ≤8K tokens (coding + tool calling) — token filtering done by filter_for_v12_pruned.py
SOURCES_TIER1 = [
    ("nemotron_tool_calling_filtered.jsonl", 40_000, "nemotron_tool_calling"),
    ("competitive_programming_converted.jsonl", 28_000, "competitive_programming"),
    ("nemotron_agentic_filtered.jsonl", 19_028, "nemotron_agentic"),
    ("xlam_filtered.jsonl", 15_000, "xlam"),
    ("code_feedback_filtered.jsonl", 15_000, "code_feedback"),
    ("qwen3_coder_distill_converted.jsonl", 9_500, "qwen3_coder_distill"),
    ("magicoder_filtered.jsonl", 5_000, "magicoder"),
]

# Tier 2: ≤16K tokens (agentic/SWE) — token filtering done by filter_for_v12_pruned.py
SOURCES_TIER2 = [
    ("opencoder_reasoning_filtered.jsonl", 16_025, "opencoder_reasoning"),
    ("swesmith_filtered.jsonl", 9_780, "swesmith"),
]

# Combined pruned sources
SOURCES_PRUNED = SOURCES_TIER1 + SOURCES_TIER2

# Unfiltered mix: original *_converted.jsonl files (old 262K mix, kept for reference)
SOURCES_UNFILTERED = [
    ("nemotron_tool_calling_converted.jsonl", 80_000, "nemotron_tool_calling"),
    ("opencoder_reasoning_converted.jsonl", 50_000, "opencoder_reasoning"),
    ("code_feedback_converted.jsonl", 40_000, "code_feedback"),
    ("magicoder_converted.jsonl", 30_000, "magicoder"),
    ("swesmith_converted.jsonl", 30_000, "swesmith"),
    ("nemotron_agentic_converted.jsonl", 17_000, "nemotron_agentic"),
    ("xlam_converted.jsonl", 15_000, "xlam"),
]

SEED = 42


def load_jsonl(path: Path, max_rows: int, label: str) -> list[dict]:
    """Load up to max_rows from a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_rows and len(rows) >= max_rows:
                break
            row = json.loads(line)
            # Ensure source tag
            if "source" not in row:
                row["source"] = label
            rows.append(row)
    return rows


def validate_row(row: dict) -> bool:
    """Basic validation: must have messages with at least one assistant turn."""
    messages = row.get("messages", [])
    if len(messages) < 2:
        return False
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    return has_assistant


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build DeltaCoder v1.3 training mix")
    parser.add_argument(
        "--data-dir",
        default="v1.3/data/v1.3_pruned",
        help="Directory containing preprocessed JSONL files",
    )
    parser.add_argument(
        "--output",
        default="v1.3/data/v1.3_sft_train_pruned.jsonl",
        help="Output training JSONL",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--use-unfiltered",
        action="store_true",
        help="Use original unfiltered *_converted.jsonl files (old 262K mix)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check which files exist, don't build",
    )
    args = parser.parse_args()

    # Select source list and data directory based on mode
    if args.use_unfiltered:
        sources = SOURCES_UNFILTERED
        data_dir = Path("v1.3/data")
        output_path = Path("v1.3/data/v1.3_sft_train.jsonl")
        print("MODE: Unfiltered (original 262K mix)")
    else:
        sources = SOURCES_PRUNED
        data_dir = Path(args.data_dir)
        output_path = Path(args.output)
        print("MODE: Jackrong-inspired (~157K rows, tiered 8K/16K token filter)")
        print(
            "  Tier 1 (≤8K): nemotron_tool_calling, competitive_programming, nemotron_agentic,"
        )
        print("                 xlam, code_feedback, qwen3_coder_distill, magicoder")
        print("  Tier 2 (≤16K): opencoder_reasoning, swesmith")

    print(f"Data directory: {data_dir}")
    print(f"Output: {output_path}")
    print()

    # Check which sources are available
    all_rows = []
    stats = {}

    for filename, max_rows, label in sources:
        path = data_dir / filename
        if not path.exists():
            print(f"  MISSING: {filename} (expected {max_rows:,} rows)")
            stats[label] = {"expected": max_rows, "found": 0, "status": "MISSING"}
            continue

        if args.dry_run:
            # Just count lines
            count = sum(1 for _ in open(path, "r", encoding="utf-8"))
            actual = min(count, max_rows)
            print(f"  OK: {filename} — {count:,} available, will use {actual:,}")
            stats[label] = {"expected": max_rows, "found": count, "status": "OK"}
            continue

        print(f"  Loading {filename} (max {max_rows:,})...")
        rows = load_jsonl(path, max_rows, label)

        # Validate
        valid = [r for r in rows if validate_row(r)]
        invalid = len(rows) - len(valid)
        if invalid > 0:
            print(f"    Dropped {invalid} invalid rows")

        all_rows.extend(valid)
        stats[label] = {
            "expected": max_rows,
            "found": len(valid),
            "status": "OK",
        }
        print(f"    Loaded {len(valid):,} rows")

    print()

    # Summary
    total_expected = sum(s["expected"] for s in stats.values())
    total_found = sum(s["found"] for s in stats.values())
    missing = [k for k, v in stats.items() if v["status"] == "MISSING"]

    print("=" * 60)
    print(f"{'Source':<30} {'Expected':>10} {'Found':>10}")
    print("-" * 60)
    for label, s in stats.items():
        status = " ✗" if s["status"] == "MISSING" else ""
        print(f"{label:<30} {s['expected']:>10,} {s['found']:>10,}{status}")
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_expected:>10,} {total_found:>10,}")
    print("=" * 60)

    if missing:
        print(f"\nWARNING: {len(missing)} sources missing: {', '.join(missing)}")

    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to build.")
        return

    if not all_rows:
        print("ERROR: No rows loaded!")
        sys.exit(1)

    # Shuffle
    print(f"\nShuffling {len(all_rows):,} rows (seed={args.seed})...")
    random.seed(args.seed)
    random.shuffle(all_rows)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    file_size = output_path.stat().st_size / (1024**3)
    print(f"\nDone! {len(all_rows):,} rows → {output_path}")
    print(f"File size: {file_size:.2f} GB")

    # Category breakdown
    categories = {}
    for row in all_rows:
        src = row.get("source", "unknown")
        categories[src] = categories.get(src, 0) + 1

    print("\nCategory breakdown:")
    for src, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(all_rows) * 100
        print(f"  {src:<30} {count:>8,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
