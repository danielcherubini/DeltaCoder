"""
Preprocess CoderForge-Preview: XML tool calls → OpenAI JSON tool_calls format.

Input: togethercomputer/CoderForge-Preview (HuggingFace)
Output: JSONL with OpenAI-style messages + tool_calls

CoderForge uses inline XML in assistant content:
  <function_calls>
  <invoke name="execute_bash">
  <parameter name="command">ls -la</parameter>
  </invoke>
  </function_calls>

We convert these to:
  {"role": "assistant", "content": "...", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "execute_bash", "arguments": "{\"command\": \"ls -la\"}"}}]}
  {"role": "tool", "tool_call_id": "...", "content": "..."}
"""

import json
import re
import sys
import uuid
from pathlib import Path

from datasets import load_dataset


def parse_xml_tool_calls(content: str) -> tuple[str, list[dict]]:
    """Extract XML tool calls from content, return (clean_content, tool_calls)."""
    tool_calls = []

    # Match <function_calls>...</function_calls> blocks
    pattern = re.compile(
        r'<function_calls>\s*<invoke name="([^"]+)">(.*?)</invoke>\s*</function_calls>',
        re.DOTALL
    )

    clean_content = content

    for match in pattern.finditer(content):
        func_name = match.group(1)
        params_block = match.group(2)

        # Parse <parameter name="key">value</parameter>
        param_pattern = re.compile(
            r'<parameter name="([^"]+)">(.*?)</parameter>',
            re.DOTALL
        )
        arguments = {}
        for param_match in param_pattern.finditer(params_block):
            arguments[param_match.group(1)] = param_match.group(2).strip()

        call_id = f"call_{uuid.uuid4().hex[:24]}"
        tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments)
            }
        })

        # Remove the XML block from content
        clean_content = clean_content.replace(match.group(0), "").strip()

    return clean_content, tool_calls


def convert_messages(raw_messages: list[dict]) -> list[dict]:
    """Convert a CoderForge message list to OpenAI tool_calls format."""
    converted = []
    pending_tool_calls = []  # tool calls from previous assistant msg

    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            converted.append({"role": "system", "content": content})

        elif role == "assistant":
            clean_content, tool_calls = parse_xml_tool_calls(content)

            entry = {"role": "assistant", "content": clean_content}
            if tool_calls:
                entry["tool_calls"] = tool_calls
                pending_tool_calls = tool_calls

            converted.append(entry)

        elif role == "user":
            # If there are pending tool calls, this is a tool result
            if pending_tool_calls:
                for tc in pending_tool_calls:
                    converted.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": content
                    })
                pending_tool_calls = []
            else:
                converted.append({"role": "user", "content": content})

    return converted


# Standard tool definitions for OpenHands tools
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute a bash command in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "View, create, or edit files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "enum": ["view", "create", "str_replace"]},
                    "path": {"type": "string"},
                    "file_text": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"}
                },
                "required": ["command", "path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Log a reasoning step without taking action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"}
                },
                "required": ["thought"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                },
                "required": []
            }
        }
    }
]


def process_dataset(output_path: str, split: str = "SWE_Rebench", max_rows: int = None):
    """Load CoderForge, filter to reward=1.0, convert, and write JSONL."""
    print(f"Loading CoderForge-Preview split={split}...")
    ds = load_dataset(
        "togethercomputer/CoderForge-Preview",
        "Trajectories",
        split=split,
        streaming=True
    )

    count = 0
    skipped = 0

    with open(output_path, "w") as f:
        for row in ds:
            if max_rows and count >= max_rows:
                break

            # Filter to successful trajectories only
            if row.get("reward", 0) != 1.0:
                skipped += 1
                continue

            # Parse messages (stored as JSON string)
            raw_messages = row.get("messages", [])
            if isinstance(raw_messages, str):
                raw_messages = json.loads(raw_messages)

            if not raw_messages:
                skipped += 1
                continue

            converted = convert_messages(raw_messages)

            out = {
                "messages": converted,
                "tools": TOOL_DEFINITIONS,
                "source": "coderforge",
                "id": row.get("trajectory_id", f"coderforge_{count}")
            }

            f.write(json.dumps(out) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"  Processed {count} rows (skipped {skipped})")

    print(f"Done: {count} rows written to {output_path} (skipped {skipped})")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "data/coderforge_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Process the filtered_reward1 split (already filtered to passing only)
    process_dataset(output, split="filtered_reward1")
