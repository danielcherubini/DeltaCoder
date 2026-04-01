"""
Pre-tokenize v1.2 training data for SFTTrainer with packing.

Applies the chat template and tokenizes all 262K rows, saving the result
as a HF Dataset on disk. The training script loads this directly, skipping
the slow tokenization step (which takes ~4.6 hours single-threaded).

Output dataset has columns: input_ids, attention_mask, labels
SFTTrainer with packing=True detects these and skips its own tokenization.

Usage:
    python pretokenize_for_sft.py --data /workspace/v1.2_sft_train.jsonl --output /workspace/v1.2_pretokenized --max-seq-length 32768
"""

import argparse
import json
import os
import sys
import time

from transformers import AutoTokenizer


BASE_MODEL = "Qwen/Qwen3.5-9B"


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize for SFTTrainer")
    parser.add_argument("--data", required=True, help="Path to training JSONL")
    parser.add_argument(
        "--output", required=True, help="Output directory for HF Dataset"
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

    if os.path.isdir(args.output) and any(
        f.endswith(".parquet") for f in os.listdir(args.output)
    ):
        import pyarrow.parquet as pq

        print(f"Output already exists at {args.output} — delete it to re-tokenize")
        ds = pq.ParquetDataset(args.output)
        schema = ds.schema
        total = sum(
            pq.read_metadata(os.path.join(args.output, f)).num_rows
            for f in sorted(os.listdir(args.output))
            if f.endswith(".parquet")
        )
        print(f"  {total:,} rows, columns: {schema.names}")
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

    # Step 2: Tokenize and save in batches (avoids OOM on 64GB machines)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pathlib import Path

    BATCH_SIZE = 50000  # Write to disk every 50K rows to limit memory

    os.makedirs(args.output, exist_ok=True)
    print(f"Tokenizing {len(texts):,} rows (max_seq_length={args.max_seq_length})...")
    t1 = time.time()

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    total_tokens = 0
    total_rows = 0
    shard_idx = 0

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

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        total_tokens += len(input_ids)
        total_rows += 1

        # Write batch to disk as parquet shard
        if len(batch_input_ids) >= BATCH_SIZE:
            elapsed = time.time() - t1
            print(
                f"  {total_rows:,}/{len(texts):,} tokenized in {elapsed:.1f}s "
                f"({total_rows / elapsed:.0f} rows/s) — writing shard {shard_idx}..."
            )
            table = pa.table(
                {
                    "input_ids": pa.array(batch_input_ids, type=pa.list_(pa.int32())),
                    "attention_mask": pa.array(
                        batch_attention_mask, type=pa.list_(pa.int32())
                    ),
                    "labels": pa.array(batch_labels, type=pa.list_(pa.int32())),
                }
            )
            pq.write_table(
                table,
                os.path.join(args.output, f"shard_{shard_idx:04d}.parquet"),
            )
            batch_input_ids.clear()
            batch_attention_mask.clear()
            batch_labels.clear()
            shard_idx += 1

    # Write remaining rows
    if batch_input_ids:
        print(f"  Writing final shard {shard_idx} ({len(batch_input_ids):,} rows)...")
        table = pa.table(
            {
                "input_ids": pa.array(batch_input_ids, type=pa.list_(pa.int32())),
                "attention_mask": pa.array(
                    batch_attention_mask, type=pa.list_(pa.int32())
                ),
                "labels": pa.array(batch_labels, type=pa.list_(pa.int32())),
            }
        )
        pq.write_table(
            table,
            os.path.join(args.output, f"shard_{shard_idx:04d}.parquet"),
        )
        shard_idx += 1

    elapsed = time.time() - t1
    print(f"  Tokenized {total_rows:,} rows in {elapsed:.1f}s")

    # Step 3: Stats
    avg_len = total_tokens / total_rows if total_rows > 0 else 0
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg sequence length: {avg_len:.0f}")
    print(f"  Shards written: {shard_idx}")
    disk_size = sum(
        f.stat().st_size for f in Path(args.output).rglob("*") if f.is_file()
    )
    print(f"  Dataset size on disk: {disk_size / 1024**3:.2f} GB")
    print("Done!")


if __name__ == "__main__":
    main()
