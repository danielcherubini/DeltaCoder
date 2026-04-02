"""
Preprocess Nemotron-Agentic-v1 tool_calling split: Strip reasoning_content, normalize schema.

Input: nvidia/Nemotron-Agentic-v1 data/tool_calling.jsonl (316K rows)
Output: JSONL with standardized OpenAI tool_calls format

We sample a subset (default 80K) since the full 316K would dominate the mix.
"""

import json
import random
import sys
from pathlib import Path


def clean_message(msg: dict) -> dict:
    """Normalize a message, stripping reasoning_content and null fields."""
    cleaned = {"role": msg["role"], "content": msg.get("content", "") or ""}

    # Preserve tool_calls on assistant messages
    if msg.get("tool_calls"):
        cleaned["tool_calls"] = msg["tool_calls"]

    # Preserve tool_call_id on tool messages
    if msg["role"] == "tool" and msg.get("tool_call_id"):
        cleaned["tool_call_id"] = msg["tool_call_id"]

    # Deliberately drop reasoning_content — we don't want to train on it
    # since local inference won't produce it

    return cleaned


def process_dataset(output_path: str, max_rows: int = 80_000, seed: int = 42):
    """Load Nemotron-Agentic-v1 tool_calling split, normalize, sample, and write JSONL."""
    from huggingface_hub import hf_hub_download

    print("Downloading Nemotron-Agentic-v1 tool_calling split...")
    jsonl_path = hf_hub_download(
        repo_id="nvidia/Nemotron-Agentic-v1",
        filename="data/tool_calling.jsonl",
        repo_type="dataset",
    )

    # First pass: load all rows
    print("Loading all rows...")
    all_rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            all_rows.append(json.loads(line))

    print(f"Total rows: {len(all_rows)}")

    # Sample if needed
    if max_rows and len(all_rows) > max_rows:
        print(f"Sampling {max_rows} rows (seed={seed})...")
        random.seed(seed)
        all_rows = random.sample(all_rows, max_rows)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for row in all_rows:
            messages = [clean_message(m) for m in row["messages"]]
            tools = row.get("tools", [])

            out = {
                "messages": messages,
                "tools": tools,
                "source": "nemotron_tool_calling",
                "id": row.get("uuid", f"nemotron_tc_{count}"),
            }

            f_out.write(json.dumps(out) + "\n")
            count += 1

            if count % 10000 == 0:
                print(f"  Written {count} rows")

    print(f"Done: {count} rows written to {output_path}")


if __name__ == "__main__":
    output = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/nemotron_tool_calling_converted.jsonl"
    )
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 80_000
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    process_dataset(output, max_rows=max_rows)
