"""
Preprocess SWE-agent-trajectories: Plain text commands → OpenAI JSON tool_calls.

Input: nebius/SWE-agent-trajectories (HuggingFace)
Output: JSONL with OpenAI-style messages + tool_calls

SWE-agent uses a custom format:
  - Roles: system, user, ai (not "assistant")
  - No structured tool calls — agent writes DISCUSSION then a command in a code block
  - Commands: open, edit, create, search_dir, search_file, submit, plus raw bash

We convert:
  - "ai" → "assistant"
  - Extract command from code block → tool_call for execute_bash
  - Next "user" message becomes the tool result
"""

import json
import re
import sys
import uuid
from pathlib import Path

from datasets import load_dataset


# SWE-agent built-in commands that we wrap as tool calls
SWEAGENT_COMMANDS = {
    "open", "edit", "create", "search_dir", "search_file", "find_file",
    "scroll_up", "scroll_down", "goto", "submit"
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command or SWE-agent command in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute."}
                },
                "required": ["command"]
            }
        }
    }
]


def extract_command(text: str) -> tuple[str, str | None]:
    """Extract the command from an AI message. Returns (discussion, command)."""
    # SWE-agent format: DISCUSSION text followed by a command in a code block
    # or just a bare command at the end

    # Try code block first
    code_match = re.search(r'```\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_match:
        command = code_match.group(1).strip()
        discussion = text[:code_match.start()].strip()
        # Clean up DISCUSSION prefix
        discussion = re.sub(r'^DISCUSSION\s*\n?', '', discussion).strip()
        return discussion, command

    # Try bare command after DISCUSSION
    parts = re.split(r'\nDISCUSSION\s*\n', text)
    if len(parts) == 1:
        parts = re.split(r'^DISCUSSION\s*\n', text)

    if len(parts) >= 2:
        discussion = parts[0].strip() if parts[0].strip() else parts[1].strip()
        # Last non-empty line is likely the command
        lines = [l for l in text.split('\n') if l.strip()]
        if lines:
            return discussion, lines[-1].strip()

    return text.strip(), None


def convert_trajectory(trajectory: list[dict]) -> list[dict]:
    """Convert SWE-agent trajectory to OpenAI format."""
    converted = []
    pending_tool_call = None

    for turn in trajectory:
        role = turn.get("role", "")
        text = turn.get("text", "")

        if role == "system":
            converted.append({"role": "system", "content": text})

        elif role == "ai":
            discussion, command = extract_command(text)

            if command:
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                converted.append({
                    "role": "assistant",
                    "content": discussion,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "execute_bash",
                            "arguments": json.dumps({"command": command})
                        }
                    }]
                })
                pending_tool_call = call_id
            else:
                converted.append({"role": "assistant", "content": text})

        elif role == "user":
            if pending_tool_call:
                converted.append({
                    "role": "tool",
                    "tool_call_id": pending_tool_call,
                    "content": text
                })
                pending_tool_call = None
            else:
                converted.append({"role": "user", "content": text})

    return converted


def process_dataset(output_path: str, max_rows: int = None):
    """Load SWE-agent-trajectories, filter to target=True, convert, write JSONL."""
    print("Loading SWE-agent-trajectories...")
    ds = load_dataset("nebius/SWE-agent-trajectories", split="train", streaming=True)

    count = 0
    skipped = 0

    with open(output_path, "w") as f:
        for row in ds:
            if max_rows and count >= max_rows:
                break

            # Filter to successful trajectories only
            if not row.get("target", False):
                skipped += 1
                continue

            trajectory = row.get("trajectory", [])
            if not trajectory:
                skipped += 1
                continue

            converted = convert_trajectory(trajectory)

            out = {
                "messages": converted,
                "tools": TOOL_DEFINITIONS,
                "source": "sweagent",
                "id": row.get("instance_id", f"sweagent_{count}")
            }

            f.write(json.dumps(out) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"  Processed {count} rows (skipped {skipped})")

    print(f"Done: {count} rows written to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/sweagent_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    process_dataset(output)
