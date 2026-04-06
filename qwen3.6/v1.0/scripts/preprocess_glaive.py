"""
Preprocess Glaive Function-Calling v2 for DeltaCoder Qwen3.6 v1.0 SFT.

Input: glaiveai/glaive-function-calling-v2 (HuggingFace)
Output: data/glaive_converted.jsonl

Raw text parsing required — system has JSON tool defs, chat has USER/ASSISTANT delimiters.
"""

import json
import re
import sys
from pathlib import Path
from datasets import load_dataset

MAX_ROWS = 50_000
SEED = 42


def parse_system(system_text: str) -> tuple[str, list[dict]]:
    """Parse system prompt and extract tool definitions.

    Returns (system_prompt, tools_list).
    """
    if not system_text:
        return "", []

    # Remove "SYSTEM: " prefix
    text = system_text.strip()
    if text.startswith("SYSTEM:"):
        text = text[len("SYSTEM:") :].strip()

    # Find JSON objects in the text (tool definitions)
    tools = []
    # Match top-level JSON objects
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    if "name" in obj:  # looks like a tool definition
                        tools.append(
                            {
                                "type": "function",
                                "function": obj,
                            }
                        )
                except json.JSONDecodeError:
                    pass
                start = None

    # System prompt is the text before the first tool definition
    first_brace = text.find("{")
    if first_brace > 0:
        system_prompt = text[:first_brace].strip().rstrip("-").strip()
    else:
        system_prompt = text

    return system_prompt, tools


def parse_chat(chat_text: str) -> list[dict]:
    """Parse chat text into messages list.

    Handles USER:, ASSISTANT:, FUNCTION RESPONSE:, <functioncall> blocks.
    """
    if not chat_text:
        return []

    messages = []

    # Split on role markers
    # Pattern: captures USER: ASSISTANT: FUNCTION RESPONSE:
    parts = re.split(r"\n?(USER|ASSISTANT|FUNCTION RESPONSE)\s*:\s*", chat_text)

    # parts is: [preamble, role1, content1, role2, content2, ...]
    i = 1  # skip preamble
    while i < len(parts) - 1:
        role_text = parts[i].strip()
        content = parts[i + 1].strip()
        # Remove  markdown markers
        content = content.replace("```", "").strip()

        if not content:
            i += 2
            continue

        if role_text == "USER":
            messages.append({"role": "user", "content": content})

        elif role_text == "ASSISTANT":
            # Check for <functioncall>
            fc_match = re.match(r"<functioncall>\s*({.*})", content, re.DOTALL)
            if fc_match:
                try:
                    fc_data = json.loads(fc_match.group(1))
                    # Build tool_calls
                    args = fc_data.get("arguments", {})
                    if isinstance(args, str):
                        args_str = args
                    else:
                        args_str = json.dumps(args)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"call_{len(messages)}",
                                    "type": "function",
                                    "function": {
                                        "name": fc_data.get("name", "unknown"),
                                        "arguments": args_str,
                                    },
                                }
                            ],
                        }
                    )
                except json.JSONDecodeError:
                    # If we can't parse the function call, keep as plain text
                    messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})

        elif role_text == "FUNCTION RESPONSE":
            # This is a tool response — add as tool role
            messages.append({"role": "tool", "content": content})

        i += 2

    return messages


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "data/glaive_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading Glaive Function-Calling v2...")
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

    # Verify schema
    print(f"Columns: {ds.column_names}")
    print(f"Sample system (first 200 chars): {ds[0]['system'][:200]}")
    print(f"Sample chat (first 200 chars): {ds[0]['chat'][:200]}")

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED).select(range(min(MAX_ROWS, len(ds))))
    print(f"Using {len(ds)} rows")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            try:
                system_prompt, tools = parse_system(row.get("system", ""))
                messages = parse_chat(row.get("chat", ""))

                if not messages or len(messages) < 2:
                    skipped += 1
                    continue

                # Prepend system message if we have one
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})

                out = {
                    "messages": messages,
                    "source": "glaive",
                    "id": f"glaive-{i}",
                }
                if tools:
                    out["tools"] = tools

                f.write(json.dumps(out) + "\n")
                count += 1

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  Skip row {i}: {e}")

            if count % 5000 == 0 and count > 0:
                print(f"  Processed {count} rows (skipped {skipped})")

    skip_rate = skipped / (count + skipped) * 100 if (count + skipped) > 0 else 0
    print(
        f"Done: {count} rows written, {skipped} skipped ({skip_rate:.1f}%) → {output}"
    )


if __name__ == "__main__":
    main()
