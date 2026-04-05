"""
Preprocess Hermes Function-Calling v1 for DeltaCoder Qwen3.5 v1.1 SFT.

Input: NousResearch/hermes-function-calling-v1 (all 5 subsets)
Output: data/hermes_converted.jsonl

Conversations use {"from": ..., "value": ...} format.
Tools is a JSON string that needs parsing.
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

SUBSETS = [
    "func_calling_singleturn",
    "func_calling",
    "glaive_func_calling",
    "json_mode_agentic",
    "json_mode_singleturn",
]

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "data/hermes_converted.jsonl"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Load all subsets
    all_datasets = []
    for subset in SUBSETS:
        print(f"Loading hermes-function-calling-v1/{subset}...")
        ds = load_dataset(
            "NousResearch/hermes-function-calling-v1",
            subset,
            split="train",
        )
        print(f"  {len(ds)} rows")
        all_datasets.append(ds)

    ds = concatenate_datasets(all_datasets)
    print(f"Total: {len(ds)} rows")

    # Verify schema
    print(f"Columns: {ds.column_names}")
    sample = ds[0]
    print(f"Sample conversations[0]: {sample['conversations'][0]}")
    print(f"Tools type: {type(sample['tools'])}")

    count = 0
    skipped = 0
    with open(output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            conversations = row.get("conversations", [])
            if not conversations:
                skipped += 1
                continue

            # Convert from/value → role/content
            messages = []
            for turn in conversations:
                from_role = turn.get("from", "")
                value = turn.get("value", "")
                role = ROLE_MAP.get(from_role)
                if role is None:
                    print(f"  Unknown role '{from_role}' in row {i}, skipping row")
                    break
                messages.append({"role": role, "content": value})
            else:
                # Only reach here if loop completed without break
                pass

            if not messages or len(messages) < 2:
                skipped += 1
                continue

            # Parse tools (JSON string → list)
            tools = None
            raw_tools = row.get("tools")
            if raw_tools:
                if isinstance(raw_tools, str):
                    try:
                        tools = json.loads(raw_tools)
                    except json.JSONDecodeError:
                        tools = None
                elif isinstance(raw_tools, list):
                    tools = raw_tools

            out = {
                "messages": messages,
                "source": "hermes",
                "id": row.get("id", f"hermes-{i}"),
            }
            if tools:
                out["tools"] = tools

            f.write(json.dumps(out) + "\n")
            count += 1

    print(f"Done: {count} rows written, {skipped} skipped → {output}")


if __name__ == "__main__":
    main()
