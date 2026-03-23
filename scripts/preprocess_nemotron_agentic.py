"""
Preprocess Nemotron-Agentic-v1: Strip reasoning_content, normalize schema.

Input: nvidia/Nemotron-Agentic-v1 (HuggingFace)
Output: JSONL with standardized OpenAI tool_calls format

Only uses the `interactive_agent` split (19K rows) — the `tool_calling` split
is general-purpose, not coding-focused.
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset


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


def process_dataset(output_path: str, max_rows: int = None):
    """Load Nemotron-Agentic-v1 interactive_agent split, normalize, and write JSONL."""
    print("Loading Nemotron-Agentic-v1 (interactive_agent split)...")
    ds = load_dataset(
        "nvidia/Nemotron-Agentic-v1",
        split="interactive_agent",
        streaming=True
    )

    count = 0

    with open(output_path, "w") as f:
        for row in ds:
            if max_rows and count >= max_rows:
                break

            messages = [clean_message(m) for m in row["messages"]]
            tools = row.get("tools", [])

            out = {
                "messages": messages,
                "tools": tools,
                "source": "nemotron_agentic",
                "id": row.get("uuid", f"nemotron_agentic_{count}")
            }

            f.write(json.dumps(out) + "\n")
            count += 1

            if count % 5000 == 0:
                print(f"  Processed {count} rows")

    print(f"Done: {count} rows written to {output_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/nemotron_agentic_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    process_dataset(output)
