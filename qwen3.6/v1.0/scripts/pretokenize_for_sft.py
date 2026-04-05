"""
Pre-tokenize v1.3 training data for SFTTrainer with packing.

Applies the chat template and tokenizes all rows, saving the result
as a parquet file on disk. The training script loads this directly, skipping
the slow tokenization step.

Output dataset has columns: input_ids, attention_mask, labels
SFTTrainer with packing=True detects these and skips its own tokenization.

Usage:
    python pretokenize_for_sft.py --data /workspace/v1.3_sft_train.jsonl --output /workspace/v1.3_pretokenized.parquet --max-seq-length 32768
"""

import argparse
import json
import os
import sys
import time

from transformers import AutoTokenizer


# TODO: Verify model name when Qwen3.6 open weights release
BASE_MODEL = "Qwen/Qwen3.6-9B"


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize for SFTTrainer")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument(
        "--output", required=True, help="Output parquet file or directory"
    )
    parser.add_argument("--max-seq-length", type=int, default=32768)
    return parser.parse_args()


def normalize_messages(messages):
    """Normalize a conversation's messages for apply_chat_template."""
    normalized = []
    for m in messages:
        content = m.get("content", "") or ""
        if not isinstance(content, str):
            content = json.dumps(content) if content else ""
        msg = {"role": m["role"], "content": content}

        if "tool_calls" in m and m["tool_calls"]:
            fixed_calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                if "function" in tc:
                    func = dict(tc["function"])
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            func["arguments"] = {"raw": args}
                    tc["function"] = func
                fixed_calls.append(tc)
            msg["tool_calls"] = fixed_calls

        if "tool_call_id" in m:
            msg["tool_call_id"] = m["tool_call_id"]
        if "name" in m and m["role"] == "tool":
            msg["name"] = m["name"]

        normalized.append(msg)
    return normalized


def main():
    args = parse_args()

    # Check for existing output (single file or directory with parquet)
    output_check = args.output
    if output_check.endswith(".parquet") and os.path.isfile(output_check):
        import pyarrow.parquet as pq

        print(f"Output already exists at {output_check} — delete it to re-tokenize")
        meta = pq.read_metadata(output_check)
        print(f"  {meta.num_rows:,} rows")
        return
    elif os.path.isdir(output_check) and any(
        f.endswith(".parquet") for f in os.listdir(output_check)
    ):
        import pyarrow.parquet as pq

        print(f"Output already exists at {output_check} — delete it to re-tokenize")
        total = sum(
            pq.read_metadata(os.path.join(output_check, f)).num_rows
            for f in sorted(os.listdir(output_check))
            if f.endswith(".parquet")
        )
        print(f"  {total:,} rows")
        return

    print(f"Loading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # Step 1: Apply chat template to get text
    print(f"Loading and applying chat template to {args.data}...")
    texts = []
    skipped = 0
    t0 = time.time()
    with open(args.data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            try:
                messages = normalize_messages(row["messages"])
                tools = row.get("tools", None)
                kwargs = {"tokenize": False, "add_generation_prompt": False}
                if tools:
                    kwargs["tools"] = tools
                text = tokenizer.apply_chat_template(messages, **kwargs)
                texts.append(text)
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Skip row {i}: {e}")
            if (i + 1) % 50000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  {i + 1:,} rows in {elapsed:.1f}s ({(i + 1) / elapsed:.0f} rows/s)"
                )

    print(
        f"  Applied template to {len(texts):,} rows ({skipped} skipped) in {time.time() - t0:.1f}s"
    )

    # Step 2: Tokenize all rows
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path

    print(f"Tokenizing {len(texts):,} rows (max_seq_length={args.max_seq_length})...")
    t1 = time.time()

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    total_tokens = 0

    for i, text in enumerate(texts):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        # For SFT, labels = input_ids (shifted internally by the model)
        labels = input_ids.copy()

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
        total_tokens += len(input_ids)

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t1
            print(
                f"  {i + 1:,}/{len(texts):,} tokenized in {elapsed:.1f}s "
                f"({(i + 1) / elapsed:.0f} rows/s)"
            )

    elapsed = time.time() - t1
    print(f"  Tokenized {len(all_input_ids):,} rows in {elapsed:.1f}s")

    # Step 3: Save as single parquet file
    output_path = args.output
    if not output_path.endswith(".parquet"):
        # If output is a directory path, make it a single file instead
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(output_path, "data.parquet")

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    print(f"Saving to {output_path}...")
    table = pa.table(
        {
            "input_ids": pa.array(all_input_ids, type=pa.list_(pa.int32())),
            "attention_mask": pa.array(all_attention_mask, type=pa.list_(pa.int32())),
            "labels": pa.array(all_labels, type=pa.list_(pa.int32())),
        }
    )
    pq.write_table(table, output_path)

    # Stats
    total_rows = len(all_input_ids)
    avg_len = total_tokens / total_rows if total_rows > 0 else 0
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg sequence length: {avg_len:.0f}")
    disk_size = Path(output_path).stat().st_size
    print(f"  Dataset size on disk: {disk_size / 1024**3:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
