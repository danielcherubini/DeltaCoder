"""
Filter existing preprocessed datasets for the pruned v1.3 training mix.

Reads the existing *_converted.jsonl files and applies quality + token-length filtering.

Tiered token limits (Jackrong-inspired, 2026-04-05):
  Tier 1 — ≤8K tokens (coding + tool calling):
    nemotron_tool_calling: Top 40K by tool call count, then 8K token filter
    code_feedback:         ≥4 messages (multi-turn), then 8K token filter
    swesmith:              ≤16K token filter (agentic — Tier 2), top by tool call count
    magicoder:             Top 5K by total length, then 8K token filter
    nemotron_agentic:      All kept (99.1% naturally ≤8K)
    xlam:                  All kept

  Tier 2 — ≤16K tokens (agentic/SWE):
    opencoder_reasoning:   ≤16K token filter, top by assistant response length
    swesmith:              ≤16K token filter, top by tool call count

  Dropped entirely:
    nemotron_swe — 100% of rows exceed 16K tokens (median 43K). Not worth filtering.

Token estimation: chars / 3.5 (conservative for English code/text mix).

Outputs filtered files to data/v1.3_pruned/ directory.

Usage:
    python filter_for_v12_pruned.py [--data-dir v1.3/data] [--output-dir v1.3/data/v1.3_pruned]
"""

import argparse
import json
import os
import sys
from pathlib import Path


SEED = 42

# Token limits
TOKEN_LIMIT_TIER1 = 8_192  # ≤8K tokens for coding/tool-calling sources
TOKEN_LIMIT_TIER2 = 16_384  # ≤16K tokens for agentic/SWE sources
CHARS_PER_TOKEN = 3.5  # Conservative estimate for English code/text


def estimate_tokens(messages: list[dict]) -> int:
    """Estimate token count from total character length."""
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    return int(total_chars / CHARS_PER_TOKEN)


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


def apply_token_filter(rows: list[dict], token_limit: int, label: str) -> list[dict]:
    """Filter rows to those with estimated token count <= token_limit."""
    before = len(rows)
    filtered = [
        r for r in rows if estimate_tokens(r.get("messages", [])) <= token_limit
    ]
    dropped = before - len(filtered)
    if dropped > 0:
        print(
            f"  Token filter (≤{token_limit:,}): dropped {dropped:,} rows ({dropped / before * 100:.1f}%)"
        )
    return filtered


def filter_by_top_n(rows: list[dict], key_fn, n: int, desc: str) -> list[dict]:
    """Sort rows by key_fn descending, keep top N."""
    scored = [(key_fn(r), i, r) for i, r in enumerate(rows)]
    scored.sort(key=lambda x: x[0], reverse=True)
    result = [r for _, _, r in scored[:n]]
    if scored and len(scored) > 1:
        kept_min = scored[min(n - 1, len(scored) - 1)][0]
        kept_max = scored[0][0]
        print(f"  {desc}: kept range [{kept_min}, {kept_max}]")
    return result


def filter_nemotron_tool_calling(data_dir: str, output_dir: str):
    """80K → ~40K: Keep conversations with most tool calls, then 8K token filter."""
    path = os.path.join(data_dir, "nemotron_tool_calling_converted.jsonl")
    print(f"\n=== nemotron_tool_calling (Tier 1, ≤8K) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # First apply 8K token filter
    rows = apply_token_filter(rows, TOKEN_LIMIT_TIER1, "nemotron_tool_calling")

    # Sort by number of tool calls + message count as tiebreaker
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
    """65K → ~16K: Apply 16K token filter, then keep by longest assistant responses."""
    path = os.path.join(data_dir, "opencoder_reasoning_converted.jsonl")
    print(f"\n=== opencoder_reasoning (Tier 2, ≤16K) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Apply 16K token filter first
    rows = apply_token_filter(rows, TOKEN_LIMIT_TIER2, "opencoder_reasoning")

    # From survivors, keep top by assistant response length
    TARGET = 16_025
    if len(rows) > TARGET:

        def sort_key(r):
            return assistant_text_length(r.get("messages", []))

        rows = filter_by_top_n(rows, sort_key, TARGET, "assistant_text_length")

    print(f"  Output: {len(rows):,}")
    write_jsonl(rows, os.path.join(output_dir, "opencoder_reasoning_filtered.jsonl"))
    return len(rows)


def filter_code_feedback(data_dir: str, output_dir: str):
    """50K → ~15K: Multi-turn ≥4 messages, then 8K token filter."""
    path = os.path.join(data_dir, "code_feedback_converted.jsonl")
    print(f"\n=== code_feedback (Tier 1, ≤8K) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Filter for ≥4 messages (multi-turn feedback loops)
    multi_turn = [r for r in rows if len(r.get("messages", [])) >= 4]
    print(f"  After ≥4 messages filter: {len(multi_turn):,}")

    # Apply 8K token filter
    multi_turn = apply_token_filter(multi_turn, TOKEN_LIMIT_TIER1, "code_feedback")

    # If more than 15K, take the longest ones
    if len(multi_turn) > 15_000:

        def sort_key(r):
            return total_text_length(r.get("messages", []))

        multi_turn = filter_by_top_n(multi_turn, sort_key, 15_000, "total_text_length")

    print(f"  Output: {len(multi_turn):,}")
    write_jsonl(multi_turn, os.path.join(output_dir, "code_feedback_filtered.jsonl"))
    return len(multi_turn)


def filter_swesmith(data_dir: str, output_dir: str):
    """30K → ~10K: Apply 16K token filter, then keep top by tool call count."""
    path = os.path.join(data_dir, "swesmith_converted.jsonl")
    print(f"\n=== swesmith (Tier 2, ≤16K) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Apply 16K token filter first
    rows = apply_token_filter(rows, TOKEN_LIMIT_TIER2, "swesmith")

    # From survivors, keep top by tool call count
    TARGET = 9_780
    if len(rows) > TARGET:

        def sort_key(r):
            return count_tool_calls(r.get("messages", []))

        rows = filter_by_top_n(rows, sort_key, TARGET, "tool_call_count")

    print(f"  Output: {len(rows):,}")
    write_jsonl(rows, os.path.join(output_dir, "swesmith_filtered.jsonl"))
    return len(rows)


def filter_magicoder(data_dir: str, output_dir: str):
    """50K → ~5K: Keep longest/hardest problems, then 8K token filter."""
    path = os.path.join(data_dir, "magicoder_converted.jsonl")
    print(f"\n=== magicoder (Tier 1, ≤8K) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Sort by total length first (hardest problems)
    def sort_key(r):
        return total_text_length(r.get("messages", []))

    rows = filter_by_top_n(
        rows, sort_key, min(len(rows), 10_000), "total_text_length (pre-filter)"
    )

    # Apply 8K token filter
    rows = apply_token_filter(rows, TOKEN_LIMIT_TIER1, "magicoder")

    # Keep top 5K
    if len(rows) > 5_000:
        rows = rows[:5_000]

    print(f"  Output: {len(rows):,}")
    write_jsonl(rows, os.path.join(output_dir, "magicoder_filtered.jsonl"))
    return len(rows)


def copy_nemotron_agentic(data_dir: str, output_dir: str):
    """~19K → all: Keep all (99.1% naturally ≤8K tokens)."""
    path = os.path.join(data_dir, "nemotron_agentic_converted.jsonl")
    print(f"\n=== nemotron_agentic (Tier 1, ≤8K, all kept) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,}")

    # Apply 8K filter just to be safe (99.1% should pass)
    rows = apply_token_filter(rows, TOKEN_LIMIT_TIER1, "nemotron_agentic")

    print(f"  Output: {len(rows):,} (keeping all that fit in 8K)")
    write_jsonl(rows, os.path.join(output_dir, "nemotron_agentic_filtered.jsonl"))
    return len(rows)


def copy_xlam(data_dir: str, output_dir: str):
    """~15K → all: Keep all (median ~529 tokens, well within 8K)."""
    path = os.path.join(data_dir, "xlam_converted.jsonl")
    print(f"\n=== xlam (Tier 1, ≤8K, all kept) ===")
    rows = load_jsonl(path)
    print(f"  Loaded: {len(rows):,} (keeping all)")
    write_jsonl(rows, os.path.join(output_dir, "xlam_filtered.jsonl"))
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Filter datasets for pruned v1.3 mix")
    parser.add_argument("--data-dir", default="v1.3/data", help="Input data directory")
    parser.add_argument(
        "--output-dir", default="v1.3/data/v1.3_pruned", help="Output directory"
    )
    args = parser.parse_args()

    print(f"Input: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(
        f"Token limits: Tier 1 ≤{TOKEN_LIMIT_TIER1:,} tokens, Tier 2 ≤{TOKEN_LIMIT_TIER2:,} tokens"
    )
    print(f"NOTE: nemotron_swe skipped — 100% of rows exceed 16K tokens (median 43K)")

    os.makedirs(args.output_dir, exist_ok=True)

    totals = {}
    # Tier 1: ≤8K tokens
    totals["nemotron_tool_calling"] = filter_nemotron_tool_calling(
        args.data_dir, args.output_dir
    )
    totals["code_feedback"] = filter_code_feedback(args.data_dir, args.output_dir)
    totals["magicoder"] = filter_magicoder(args.data_dir, args.output_dir)
    totals["nemotron_agentic"] = copy_nemotron_agentic(args.data_dir, args.output_dir)
    totals["xlam"] = copy_xlam(args.data_dir, args.output_dir)

    # Tier 2: ≤16K tokens
    totals["opencoder_reasoning"] = filter_opencoder_reasoning(
        args.data_dir, args.output_dir
    )
    totals["swesmith"] = filter_swesmith(args.data_dir, args.output_dir)

    print("\n" + "=" * 60)
    print(f"{'Source':<30} {'Rows':>10} {'Tier':>8}")
    print("-" * 60)
    tier1_sources = {
        "nemotron_tool_calling",
        "code_feedback",
        "magicoder",
        "nemotron_agentic",
        "xlam",
    }
    grand_total = 0
    for src, count in totals.items():
        tier = "1 (≤8K)" if src in tier1_sources else "2 (≤16K)"
        print(f"{src:<30} {count:>10,} {tier:>8}")
        grand_total += count
    print("-" * 60)
    print(f"{'TOTAL':<30} {grand_total:>10,}")
    print("=" * 60)
    print(f"\nFiltered files in: {args.output_dir}/")
    print(f"Note: competitive_programming and qwen3_coder_distill are pre-cleaned")
    print(
        f"      by preprocess_competitive_programming.py and preprocess_qwen3_coder_distill.py"
    )


if __name__ == "__main__":
    main()
