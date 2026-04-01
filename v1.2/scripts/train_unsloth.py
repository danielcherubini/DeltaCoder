"""
DeltaCoder v1.2 SFT training with Unsloth + Qwen3.5-9B.

Uses FastVisionModel to load the full VLM (preserving vision weights),
but only trains language layers via finetune_vision_layers=False.

Applies VLM packing unblock (from unslothai/unsloth#4160) to enable
sample packing at 32K context. The NaN gradient issue only affects
total tokens per batch >~64K; at batch_size=1 + 32K we're safe.

Supports two data input modes:
  1. Raw JSONL (--data /path/to/v1.2_sft_train.jsonl) — tokenizes on-the-fly
  2. Pre-tokenized parquet dir (--data /path/to/v1.2_pretokenized/) — skips tokenization

Pre-tokenized mode is ~100x faster to start since 262K rows are already tokenized.

Usage:
    python train_unsloth.py --data /workspace/v1.2_pretokenized
    python train_unsloth.py --data /workspace/v1.2_pretokenized --max-steps 20  # dry run
    python train_unsloth.py --data /workspace/v1.2_sft_train.jsonl  # raw JSONL fallback
"""

import argparse
import json
import os
import sys

# ---------- VLM Packing Unblock ----------
# Must be done BEFORE importing unsloth, since unsloth patches SFTTrainer at import time.
# This removes the VLM check that blocks sample packing for VLMs doing text-only training.
# Reference: https://github.com/unslothai/unsloth/issues/4160


def _apply_vlm_packing_unblock():
    """
    Monkey-patch unsloth's trainer module to remove the VLM packing block.

    Unsloth blocks sample packing when it detects a VLM (ForConditionalGeneration
    in architectures or vision_config in model config). For text-only SFT on a VLM,
    packing is safe and necessary for reasonable training speed.

    We also need to bypass the ProcessorMixin check, since FastVisionModel returns
    a processor, not a plain tokenizer. We'll pass the tokenizer directly to SFTTrainer
    instead of the processor, so this check won't fire.
    """
    pass  # We'll patch trainer.py directly on the instance instead


# CRITICAL: unsloth must be imported FIRST, before any other ML imports
import unsloth  # noqa: F401 — patches must be applied before transformers/trl/peft load

import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel


# ---------- Constants ----------
BASE_MODEL = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 32768
LORA_R = 64
LORA_ALPHA = 32
OUTPUT_DIR = "./outputs/deltacoder-9b-v1.2"

# Qwen3.5 GDN + Attention + MLP target modules
LORA_TARGET_MODULES = [
    # Full Attention (8 of 32 layers)
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # GDN / Gated Delta Net (24 of 32 layers)
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    # MLP (all 32 layers)
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaCoder v1.2 SFT with Unsloth")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training JSONL or pre-tokenized parquet directory",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max steps (-1 = full epoch)"
    )
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    return parser.parse_args()


def load_dataset_from_jsonl(path: str) -> list[dict]:
    """Load our JSONL format into a list of dicts."""
    print(f"Loading dataset from {path}...")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"Loaded {len(rows):,} rows")
    return rows


def normalize_messages(messages, tools=None):
    """Normalize a conversation's messages for apply_chat_template."""
    normalized = []
    for m in messages:
        content = m.get("content", "") or ""
        # Some tool messages have content as dict — stringify it
        if not isinstance(content, str):
            content = json.dumps(content) if content else ""
        msg = {"role": m["role"], "content": content}

        # Pass through tool_calls, parsing arguments from JSON string to dict
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

        # Pass through tool response metadata
        if "tool_call_id" in m:
            msg["tool_call_id"] = m["tool_call_id"]
        if "name" in m and m["role"] == "tool":
            msg["name"] = m["name"]

        normalized.append(msg)
    return normalized


def build_text_dataset(rows: list[dict], tokenizer) -> Dataset:
    """Convert raw rows to a Dataset with a 'text' column using the chat template."""
    print("Applying chat template to all rows...")
    texts = []
    skipped = 0
    for i, row in enumerate(rows):
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
            print(f"  Processed {i + 1:,}/{len(rows):,} rows...")

    print(f"  Built {len(texts):,} texts ({skipped} skipped)")
    return Dataset.from_dict({"text": texts})


def main():
    args = parse_args()

    print("=" * 60)
    print("DeltaCoder v1.2 SFT Training (FastVisionModel + Packing)")
    print(f"Model: {BASE_MODEL}")
    print(f"Max seq length: {args.max_seq_length}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Batch size: {args.batch_size}, Grad accum: {args.grad_accum}")
    print(f"Learning rate: {args.lr}")
    print(f"Max steps: {args.max_steps if args.max_steps > 0 else 'full epoch'}")
    print("=" * 60)

    # Load model with FastVisionModel (preserves full VLM including vision encoder)
    print("\nLoading model with FastVisionModel...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,  # bf16 LoRA, no QLoRA for Qwen3.5
        load_in_16bit=True,
        full_finetuning=False,
    )

    # Apply LoRA — ONLY language layers, freeze vision
    print("Applying LoRA adapters (language-only, vision frozen)...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # FREEZE vision encoder
        finetune_language_layers=True,  # Train language layers
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=args.max_seq_length,
    )

    # NOTE: FastVisionModel returns a processor (ProcessorMixin), not a plain tokenizer.
    # For text-only training, we need to extract the tokenizer from the processor
    # and use it directly with SFTTrainer (otherwise ProcessorMixin blocks packing).
    from transformers import ProcessorMixin

    if isinstance(tokenizer, ProcessorMixin):
        print(
            "Detected ProcessorMixin — extracting underlying tokenizer for text-only training"
        )
        actual_tokenizer = tokenizer.tokenizer
    else:
        actual_tokenizer = tokenizer

    # Load dataset — two modes:
    # 1. Pre-tokenized parquet dir (fast): has input_ids/attention_mask/labels columns
    # 2. Raw JSONL (slow): needs chat template + tokenization
    is_pretokenized = os.path.isdir(args.data) and any(
        f.endswith(".parquet") for f in os.listdir(args.data)
    )

    if is_pretokenized:
        print(f"\nLoading pre-tokenized dataset from {args.data}...")
        dataset = Dataset.from_parquet(
            sorted(
                os.path.join(args.data, f)
                for f in os.listdir(args.data)
                if f.endswith(".parquet")
            )
        )
        print(f"  Loaded {len(dataset):,} rows, columns: {dataset.column_names}")
        # Quick stats from first 1000 rows (full scan is too slow for 262K rows)
        sample_lens = [
            len(dataset[i]["input_ids"]) for i in range(min(1000, len(dataset)))
        ]
        avg_sample = sum(sample_lens) / len(sample_lens)
        print(f"  Sample avg seq length (first 1K): {avg_sample:.0f}")
        print(f"  Estimated total tokens: {avg_sample * len(dataset):,.0f}")
        use_text_field = False
    else:
        print(f"\nLoading raw JSONL dataset from {args.data}...")
        # Cache the tokenized dataset on disk so we don't re-tokenize 262K rows every run
        cache_path = args.data + ".templated_cache"
        if os.path.isdir(cache_path):
            print(f"Loading cached dataset from {cache_path}...")
            dataset = Dataset.load_from_disk(cache_path)
            print(f"  Loaded {len(dataset):,} rows from cache")
        else:
            rows = load_dataset_from_jsonl(args.data)
            dataset = build_text_dataset(rows, actual_tokenizer)
            del rows  # free memory
            print(f"Saving dataset cache to {cache_path}...")
            dataset.save_to_disk(cache_path)
            print(f"  Cached {len(dataset):,} rows")
        use_text_field = True

    # Enable training mode
    FastVisionModel.for_training(model)

    # Configure trainer — pass actual_tokenizer (not processor) to avoid ProcessorMixin block
    sft_config_kwargs = dict(
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=1,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        output_dir=args.output_dir,
        optim="adamw_8bit",
        bf16=True,
        tf32=True,
        seed=42,
        dataset_num_proc=1,  # CRITICAL: Qwen3.5 tokenizer crashes with multiprocessing
        packing=True,  # Enabled! Requires VLM packing unblock patch on trainer.py
        report_to="none",
    )

    # Only set dataset_text_field for raw JSONL mode (text column)
    # For pre-tokenized data, SFTTrainer detects input_ids/attention_mask/labels columns
    if use_text_field:
        sft_config_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=actual_tokenizer,
        args=SFTConfig(**sft_config_kwargs),
    )

    # Print trainable params
    model.print_trainable_parameters()

    # Train
    print("\nStarting training...")
    stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Final loss: {stats.training_loss:.4f}")

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    actual_tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
