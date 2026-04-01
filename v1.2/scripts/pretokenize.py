"""
Pre-tokenize DeltaCoder training data to bypass Axolotl's O(n²) chat_template bug.
See: https://github.com/axolotl-ai-cloud/axolotl/issues/2396

Instead of Axolotl calling apply_chat_template 2N times per conversation (to diff
turn boundaries), we call it ONCE and manually find assistant turn boundaries using
the <|im_start|>assistant marker tokens.

Output: JSONL with {input_ids, attention_mask, labels} — ready for Axolotl with `type:` empty.
Labels: -100 for system/user tokens (masked), token IDs for assistant tokens (trained on).
"""

import json
import sys
import os
import multiprocessing as mp
from functools import partial
from transformers import AutoTokenizer

SEQUENCE_LEN = 8192
MODEL_ID = "Qwen/Qwen3.5-9B"

# Qwen3.5 chat template format:
# <|im_start|>system\n...<|im_end|>\n
# <|im_start|>user\n...<|im_end|>\n
# <|im_start|>assistant\n...<|im_end|>\n
#
# We mask everything EXCEPT assistant content (between "assistant\n" and "<|im_end|>")


def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tok


def normalize_messages(messages):
    """
    Normalize messages for apply_chat_template compatibility.
    - Preserve tool_calls on assistant messages (let apply_chat_template handle formatting)
    - Preserve tool_call_id/name on tool messages
    - Ensure content is always a string
    """
    normalized = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "") or ""

        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)

        msg = {"role": role, "content": content}

        # Pass through tool_calls — apply_chat_template formats them natively
        # Parse arguments from JSON string to dict if needed (Jinja template expects dict)
        if "tool_calls" in m and m["tool_calls"]:
            fixed_calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)  # shallow copy
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

        # Pass through tool response metadata
        if "tool_call_id" in m:
            msg["tool_call_id"] = m["tool_call_id"]
        if "name" in m and role == "tool":
            msg["name"] = m["name"]

        normalized.append(msg)
    return normalized


def find_assistant_spans(input_ids, im_start_id, im_end_id, assistant_nl_ids):
    """
    Find token spans that are assistant content.
    Returns list of (start, end) tuples where start is inclusive, end is exclusive.

    Strategy: scan for <|im_start|> followed by "assistant\n" tokens,
    then mark everything from after that header until <|im_end|> as assistant content.
    We also include the <|im_end|> token so the model learns to stop.
    """
    spans = []
    i = 0
    n = len(input_ids)
    header_len = len(assistant_nl_ids)

    while i < n:
        # Look for <|im_start|>
        if input_ids[i] == im_start_id:
            # Check if next tokens match "assistant\n"
            header_start = i + 1
            header_end = header_start + header_len
            if (
                header_end <= n
                and input_ids[header_start:header_end] == assistant_nl_ids
            ):
                # Found assistant turn — content starts after header
                content_start = header_end
                # Find the closing <|im_end|>
                j = content_start
                while j < n and input_ids[j] != im_end_id:
                    j += 1
                # Include <|im_end|> token in the span (model must learn to emit it)
                content_end = j + 1 if j < n else j
                spans.append((content_start, content_end))
                i = content_end
                continue
        i += 1

    return spans


def tokenize_conversation(
    messages,
    tokenizer,
    sequence_len,
    im_start_id,
    im_end_id,
    assistant_nl_ids,
    tools=None,
):
    """Tokenize a single conversation with proper assistant masking."""
    # Single call to apply_chat_template — O(1) per conversation
    kwargs = dict(
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
    )
    if tools:
        kwargs["tools"] = tools
    out = tokenizer.apply_chat_template(
        messages,
        **kwargs,
    )
    input_ids = list(out["input_ids"])

    # Find assistant content spans
    spans = find_assistant_spans(input_ids, im_start_id, im_end_id, assistant_nl_ids)

    # Build labels: -100 everywhere, then fill in assistant spans
    labels = [-100] * len(input_ids)
    for start, end in spans:
        for k in range(start, min(end, len(input_ids))):
            labels[k] = input_ids[k]

    # Truncate to sequence_len
    input_ids = input_ids[:sequence_len]
    labels = labels[:sequence_len]
    attention_mask = [1] * len(input_ids)

    return input_ids, attention_mask, labels


def process_chunk(chunk, model_id, sequence_len):
    """Process a chunk of rows (for multiprocessing)."""
    tokenizer = load_tokenizer(model_id)

    # Get special token IDs for this tokenizer
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Tokenize "assistant\n" to get the header tokens after <|im_start|>
    assistant_nl_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    results = []
    for row in chunk:
        messages = normalize_messages(row["messages"])
        tools = row.get("tools", None)
        try:
            input_ids, attention_mask, labels = tokenize_conversation(
                messages,
                tokenizer,
                sequence_len,
                im_start_id,
                im_end_id,
                assistant_nl_ids,
                tools=tools,
            )
            # Skip empty sequences
            if len(input_ids) < 10:
                continue
            # Skip if no assistant content found
            n_trainable = sum(1 for l in labels if l != -100)
            if n_trainable == 0:
                continue
            results.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        except Exception as e:
            print(f"Skipping row: {e}", file=sys.stderr)
            continue
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-tokenize DeltaCoder training data"
    )
    parser.add_argument(
        "input_path", nargs="?", default="data/train.jsonl", help="Input JSONL file"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="data/train_tokenized.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "num_workers",
        nargs="?",
        type=int,
        default=min(mp.cpu_count(), 32),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows (for testing)",
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    num_workers = args.num_workers

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers}")
    print(f"Sequence length: {SEQUENCE_LEN}")
    print(f"Model: {MODEL_ID}")
    if args.limit:
        print(f"Limit: {args.limit} rows")

    # Load all rows into memory
    print("Loading dataset...", flush=True)
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"Loaded {len(rows)} rows", flush=True)

    # Test with first row
    print("Testing tokenizer...", flush=True)
    tokenizer = load_tokenizer(MODEL_ID)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_nl_ids = tokenizer.encode("assistant\n", add_special_tokens=False)
    print(f"  <|im_start|> = {im_start_id}, <|im_end|> = {im_end_id}")
    print(f"  'assistant\\n' tokens = {assistant_nl_ids}")

    # Test a few rows to verify masking works
    for test_idx in range(min(5, len(rows))):
        test_messages = normalize_messages(rows[test_idx]["messages"])
        test_tools = rows[test_idx].get("tools", None)
        test_ids, test_mask, test_labels = tokenize_conversation(
            test_messages,
            tokenizer,
            SEQUENCE_LEN,
            im_start_id,
            im_end_id,
            assistant_nl_ids,
            tools=test_tools,
        )
        n_masked = sum(1 for l in test_labels if l == -100)
        n_total = len(test_labels)
        n_trainable = n_total - n_masked
        has_tools = "tools" in rows[test_idx]
        has_tool_calls = any("tool_calls" in m for m in rows[test_idx]["messages"])
        tag = ""
        if has_tools:
            tag += " [tools]"
        if has_tool_calls:
            tag += " [tool_calls]"
        print(
            f"Row {test_idx}: {n_total} tokens, {n_trainable} assistant ({n_trainable / max(n_total, 1) * 100:.1f}%), {n_masked} masked{tag}"
        )
        if n_trainable > 0:
            break
    else:
        print(
            "WARNING: No assistant tokens found in first 5 rows — check chat template markers"
        )
    del tokenizer

    # Split into chunks for multiprocessing
    chunk_size = max(1, len(rows) // num_workers)
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} rows", flush=True)

    # Process in parallel
    print("Tokenizing...", flush=True)
    process_fn = partial(process_chunk, model_id=MODEL_ID, sequence_len=SEQUENCE_LEN)

    written = 0
    with mp.Pool(num_workers) as pool, open(output_path, "w", encoding="utf-8") as fout:
        for i, results in enumerate(pool.imap_unordered(process_fn, chunks)):
            for result in results:
                fout.write(json.dumps(result) + "\n")
                written += 1
            print(
                f"  Chunk {i + 1}/{len(chunks)} done ({written} rows written)",
                flush=True,
            )

    print(f"\nDone! {written}/{len(rows)} rows tokenized → {output_path}")
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"Output size: {file_size:.2f} GB")


if __name__ == "__main__":
    main()
