"""
Preprocess Nemotron-SWE-v1: Already in OpenAI JSON format — minimal conversion.

Input: nvidia/Nemotron-SWE-v1 (HuggingFace)
Output: JSONL with standardized schema

The dataset is already in the correct format. We just normalize the schema
and strip any null fields.
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset


def clean_message(msg: dict) -> dict:
    """Normalize a message, stripping null fields."""
    cleaned = {"role": msg["role"], "content": msg.get("content", "") or ""}

    # Preserve tool_calls on assistant messages
    if msg.get("tool_calls"):
        cleaned["tool_calls"] = msg["tool_calls"]

    # Preserve tool_call_id on tool messages
    if msg["role"] == "tool" and msg.get("id"):
        cleaned["tool_call_id"] = msg["id"]

    return cleaned


def process_dataset(output_path: str, max_rows: int = None):
    """Load Nemotron-SWE-v1, normalize, and write JSONL."""
    print("Loading Nemotron-SWE-v1...")
    ds = load_dataset("nvidia/Nemotron-SWE-v1", split="r2e_gym", streaming=True)

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
                "source": "nemotron_swe",
                "id": row.get("uuid", f"nemotron_swe_{count}")
            }

            f.write(json.dumps(out) + "\n")
            count += 1

            if count % 5000 == 0:
                print(f"  Processed {count} rows")

    print(f"Done: {count} rows written to {output_path}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/nemotron_swe_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    process_dataset(output)
