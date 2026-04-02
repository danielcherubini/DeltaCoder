"""
Filter existing preprocessed datasets for the pruned v1.2 training mix.

Reads the existing *_converted.jsonl files and applies quality filtering:
  - nemotron_tool_calling: Keep top 40K by conversation length (most tool calls)
  - opencoder_reasoning: Keep top 25K by assistant response length (longest reasoning)
  - code_feedback: Keep rows with ≥4 messages (multi-turn feedback loops)
  - swesmith: Keep rows with ≥3 tool calls (longer agentic trajectories)
  - magicoder: Keep top 5K by total text length (hardest/longest problems)
  - nemotron_agentic: Keep all (only 19K, all high quality)
  - xlam: Keep all (15K, clean verified function calling)

Outputs filtered files to data/v1.2_pruned/ directory.

Usage:
    python filter_for_v12_pruned.py [--data-dir v1.2/data] [--output-dir v1.2/data/v1.2_pruned]
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


SEED = 42


def count_tool_calls(messages: list[dict]) -> int:
    """Count total tool_calls across all assistant messages."""
    total = 0
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            total += len(m["tool_calls"])
    return total


def total_text_length(messages: list[dict]) -> int:
    """Sum of all content lengths across messages."""
    return sum(len(m.get("content", "") or "") for m in messages)


def assistant_text_length(messages: list[dict]) -> int:
    """Sum of assistant message content lengths."""
    return sum(
        len(m.get("content", "") or "")
        for m in messages
        if m.get("role") == "assistant"
    )


def load_jsonl(path: str) -> list[dict]:
    """Load all rows from a JSONL file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: str):
    """Write rows to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def filter_by_top_n(rows: list[dict], key_fn, n: int, desc: str) -> list[dict]:
    """Sort rows by key_fn descending, keep top N."""
    scored = [(key_fn(r), i, r) for i, r in enumerate(rows)]
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [r for _, _, r in scored[:n]]
    if scored:
        kept_min = scored[min(n - 1, len(scored) - 1)][0]
        kept_max = scored[0][0]
        dropped_max = scored[n][0] if n < len(scored) else 0
        print(
            f"  {desc}: kept range [{kept_min}, {kept_max}], dropped max {dropped_max}"
        )
    return result


def filter_nemotron_tool_calling(data_dir: str, output_dir: str):
    """80K → 40K: Keep conversations with most tool calls."""
    path = os.path.join(data_dir, "nemotron_tool_calling_converted.jsonl")
    print(f"\n=== nemotron_tool_calling ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Sort by number of tool calls, then by message count as tiebreaker
    def sort_key(r):
        msgs = r.get("messages", [])
        return (count_tool_calls(msgs), len(msgs))

    result = filter_by_top_n(rows, sort_key, 40_000, "tool_calls + msg_count")
    print(f"  Output: {len(result):,}")
    write_jsonl(
        result, os.path.join(output_dir, "nemotron_tool_calling_filtered.jsonl")
    )
    return len(result)


def filter_opencoder_reasoning(data_dir: str, output_dir: str):
    """65K → 25K: Keep rows with longest assistant responses (deepest reasoning)."""
    path = os.path.join(data_dir, "opencoder_reasoning_converted.jsonl")
    print(f"\n=== opencoder_reasoning ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    def sort_key(r):
        return assistant_text_length(r.get("messages", []))

    result = filter_by_top_n(rows, sort_key, 25_000, "assistant_text_length")
    print(f"  Output: {len(result):,}")
    write_jsonl(result, os.path.join(output_dir, "opencoder_reasoning_filtered.jsonl"))
    return len(result)


def filter_code_feedback(data_dir: str, output_dir: str):
    """50K → 15K: Keep only multi-turn conversations (≥4 messages)."""
    path = os.path.join(data_dir, "code_feedback_converted.jsonl")
    print(f"\n=== code_feedback ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Filter for ≥4 messages
    multi_turn = [r for r in rows if len(r.get("messages", [])) >= 4]
    print(f"  After ≥4 messages filter: {len(multi_turn):,}")

    # If we have more than 15K, take the longest ones
    if len(multi_turn) > 15_000:

        def sort_key(r):
            return total_text_length(r.get("messages", []))

        multi_turn = filter_by_top_n(multi_turn, sort_key, 15_000, "total_text_length")

    # If we have fewer than 15K multi-turn, that's fine — quality > quantity
    print(f"  Output: {len(multi_turn):,}")
    write_jsonl(multi_turn, os.path.join(output_dir, "code_feedback_filtered.jsonl"))
    return len(multi_turn)


def filter_swesmith(data_dir: str, output_dir: str):
    """30K → 20K: Keep rows with ≥3 tool calls (longer agentic trajectories)."""
    path = os.path.join(data_dir, "swesmith_converted.jsonl")
    print(f"\n=== swesmith ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Filter for ≥3 tool calls
    multi_tool = [r for r in rows if count_tool_calls(r.get("messages", [])) >= 3]
    print(f"  After ≥3 tool_calls filter: {len(multi_tool):,}")

    # If we have more than 20K, take the ones with most tool calls
    if len(multi_tool) > 20_000:

        def sort_key(r):
            return count_tool_calls(r.get("messages", []))

        multi_tool = filter_by_top_n(multi_tool, sort_key, 20_000, "tool_call_count")

    print(f"  Output: {len(multi_tool):,}")
    write_jsonl(multi_tool, os.path.join(output_dir, "swesmith_filtered.jsonl"))
    return len(multi_tool)


def filter_magicoder(data_dir: str, output_dir: str):
    """50K → 5K: Keep only the longest/hardest problems."""
    path = os.path.join(data_dir, "magicoder_converted.jsonl")
    print(f"\n=== magicoder ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    def sort_key(r):
        return total_text_length(r.get("messages", []))

    result = filter_by_top_n(rows, sort_key, 5_000, "total_text_length")
    print(f"  Output: {len(result):,}")
    write_jsonl(result, os.path.join(output_dir, "magicoder_filtered.jsonl"))
    return len(result)


def copy_nemotron_agentic(data_dir: str, output_dir: str):
    """19K → 17K: Keep all (capped by build_training_mix)."""
    path = os.path.join(data_dir, "nemotron_agentic_converted.jsonl")
    print(f"\n=== nemotron_agentic ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,} (keeping all)")
    write_jsonl(rows, os.path.join(output_dir, "nemotron_agentic_filtered.jsonl"))
    return len(rows)


def copy_xlam(data_dir: str, output_dir: str):
    """15K → 15K: Keep all."""
    path = os.path.join(data_dir, "xlam_converted.jsonl")
    print(f"\n=== xlam ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,} (keeping all)")
    write_jsonl(rows, os.path.join(output_dir, "xlam_filtered.jsonl"))
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Filter datasets for pruned v1.2 mix")
    parser.add_argument("--data-dir", default="v1.2/data", help="Input data directory")
    parser.add_argument(
        "--output-dir", default="v1.2/data/v1.2_pruned", help="Output directory"
    )
    args = parser.parse_args()

    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    totals = {}
    totals["nemotron_tool_calling"] = filter_nemotron_tool_calling(
        args.data_dir, args.output_dir
    )
    totals["opencoder_reasoning"] = filter_opencoder_reasoning(
        args.data_dir, args.output_dir
    )
    totals["code_feedback"] = filter_code_feedback(args.data_dir, args.output_dir)
    totals["swesmith"] = filter_swesmith(args.data_dir, args.output_dir)
    totals["magicoder"] = filter_magicoder(args.data_dir, args.output_dir)
    totals["nemotron_agentic"] = copy_nemotron_agentic(args.data_dir, args.output_dir)
    totals["xlam"] = copy_xlam(args.data_dir, args.output_dir)

    print("\n" + "=" * 60)
    print(f"{'Source':<30} {'Rows':>10}")
    print("-" * 60)
    grand_total = 0
    for src, count in totals.items():
        print(f"{src:<30} {count:>10,}")
        grand_total += count
    print("-" * 60)
    print(f"{'TOTAL':<30} {grand_total:>10,}")
    print("=" * 60)
    print(f"\nFiltered files in: {args.output_dir}/")


if __name__ == "__main__":
    main()
