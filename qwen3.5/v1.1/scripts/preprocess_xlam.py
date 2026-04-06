"""
Preprocess Salesforce/xlam-function-calling-60k for DeltaCoder Qwen3.5 v1.1 SFT.

Input: Salesforce/xlam-function-calling-60k (60K verified function-calling samples)
Output: data/xlam_converted.jsonl

Schema: {"messages": [...], "tools": [...], "source": "xlam", "id": "xlam-N"}

Each row has:
  - query (str): User question
  - tools (str): JSON string of tool definitions
  - answers (str): JSON string of tool calls
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

MAX_ROWS = 15_000
SEED = 42


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "data/xlam_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading Salesforce/xlam-function-calling-60k...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    # Verify schema
    print(f"Columns: {ds.column_names}")
    print(f"Total rows: {len(ds)}")

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_ROWS, len(ds))))
    print(f"Using {len(ds)} rows after shuffle+cap")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            try:
                query = (row.get("query") or "").strip()
                if not query:
                    skipped += 1
                    continue

                # Parse tools (stored as JSON string)
                raw_tools = row.get("tools", "[]")
                if isinstance(raw_tools, str):
                    tools_list = json.loads(raw_tools)
                else:
                    tools_list = raw_tools

                # Parse answers (stored as JSON string)
                raw_answers = row.get("answers", "[]")
                if isinstance(raw_answers, str):
                    answers_list = json.loads(raw_answers)
                else:
                    answers_list = raw_answers

                if not tools_list or not answers_list:
                    skipped += 1
                    continue

                # Convert tools to OpenAI function-calling format
                openai_tools = []
                for tool in tools_list:
                    # xlam tools have: name, description, parameters
                    params = tool.get("parameters", {})
                    # Convert xlam param format to JSON Schema
                    properties = {}
                    required = []
                    for pname, pinfo in params.items():
                        properties[pname] = {
                            "type": pinfo.get("type", "string"),
                            "description": pinfo.get("description", ""),
                        }
                        if pinfo.get("required", False):
                            required.append(pname)

                    openai_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool.get("description", ""),
                                "parameters": {
                                    "type": "object",
                                    "properties": properties,
                                    "required": required,
                                },
                            },
                        }
                    )

                # Build tool_calls from answers
                tool_calls = []
                for j, ans in enumerate(answers_list):
                    args = ans.get("arguments", {})
                    tool_calls.append(
                        {
                            "id": f"call_{j}",
                            "type": "function",
                            "function": {
                                "name": ans["name"],
                                "arguments": json.dumps(args)
                                if isinstance(args, dict)
                                else str(args),
                            },
                        }
                    )

                # Build system message listing available tools
                tool_names = [t["function"]["name"] for t in openai_tools]
                system_msg = (
                    "You are a helpful assistant with access to the following functions. "
                    "Use them if required to answer the user's question.\n\n"
                    + json.dumps(openai_tools, indent=2)
                )

                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    },
                ]

                out = {
                    "messages": messages,
                    "tools": openai_tools,
                    "source": "xlam",
                    "id": f"xlam-{row.get('id', i)}",
                }
                f.write(json.dumps(out) + "\n")
                count += 1

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  Skip row {i}: {e}")

    print(f"Done: {count} rows written, {skipped} skipped -> {output}")


if __name__ == "__main__":
    main()
