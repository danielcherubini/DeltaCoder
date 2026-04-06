"""
Preprocess SWE-smith mini_swe_agent_plus trajectories: text-based bash → structured tool_calls.

Input: Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k (HuggingFace)
Output: JSONL with OpenAI-style messages + tool_calls

SWE-smith uses a simple format:
  - system: instructs agent to interact with bash, include THOUGHT section
  - user: provides issue description or bash output
  - assistant: THOUGHT + bash code block

We convert:
  - Extract bash command from code block → tool_call for execute_bash
  - Next user message becomes the tool result
  - Keep the THOUGHT text as assistant content (reasoning)
"""

import json
import re
import sys
import uuid
import random
from pathlib import Path

from datasets import load_dataset


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def extract_bash_command(text: str) -> tuple[str, str | None]:
    """Extract bash command from assistant message. Returns (thought, command)."""
    # Look for ```bash or ``` code blocks
    match = re.search(r"```(?:bash)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        command = match.group(1).strip()
        thought = text[: match.start()].strip()
        # Clean up THOUGHT prefix
        thought = re.sub(
            r"^THOUGHT\s*:?\s*\n?", "", thought, flags=re.IGNORECASE
        ).strip()
        return thought, command

    return text.strip(), None


def convert_trajectory(messages: list[dict]) -> list[dict]:
    """Convert SWE-smith trajectory to OpenAI format with structured tool_calls."""
    converted = []
    pending_tool_call = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            converted.append({"role": "system", "content": content})

        elif role == "assistant":
            thought, command = extract_bash_command(content)

            if command:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                converted.append(
                    {
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": "execute_bash",
                                    "arguments": json.dumps({"command": command}),
                                },
                            }
                        ],
                    }
                )
                pending_tool_call = call_id
            else:
                converted.append({"role": "assistant", "content": content})

        elif role == "user":
            if pending_tool_call:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": pending_tool_call,
                        "content": content,
                    }
                )
                pending_tool_call = None
            else:
                converted.append({"role": "user", "content": content})

    return converted


def process_dataset(output_path: str, max_rows: int = 30_000, seed: int = 42):
    """Load SWE-smith trajectories, convert, sample, and write JSONL."""
    print("Loading SWE-smith-mini_swe_agent_plus-trajectories-66k...")
    ds = load_dataset(
        "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k", split="train"
    )
    print(f"Total rows: {len(ds)}")

    # Sample if needed
    if max_rows and len(ds) > max_rows:
        print(f"Sampling {max_rows} rows (seed={seed})...")
        random.seed(seed)
        indices = random.sample(range(len(ds)), max_rows)
        ds = ds.select(indices)

    count = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            messages = row.get("messages", [])
            if not messages or len(messages) < 3:
                skipped += 1
                continue

            converted = convert_trajectory(messages)

            # Verify we have at least one tool call
            has_tool_call = any(
                "tool_calls" in m for m in converted if m.get("role") == "assistant"
            )
            if not has_tool_call:
                skipped += 1
                continue

            out = {
                "messages": converted,
                "tools": TOOL_DEFINITIONS,
                "source": "swesmith",
                "id": row.get("instance_id", f"swesmith_{count}"),
            }

            f.write(json.dumps(out) + "\n")
            count += 1

            if count % 5000 == 0:
                print(f"  Written {count} rows (skipped {skipped})")

    print(f"Done: {count} rows written to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/swesmith_converted.jsonl"
    max_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    process_dataset(output, max_rows=max_rows)
