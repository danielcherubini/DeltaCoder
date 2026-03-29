"""
Qwen3.5-DeltaCoder-9B v1.2 — Unsloth LoRA SFT training script.
Loads preprocessed JSONL from local disk, uses Unsloth's VLM SFT pipeline.

Qwen3.5 is a unified VLM — must use FastVisionModel + UnslothVisionDataCollator.
See: https://unsloth.ai/docs/models/qwen3.5/fine-tune

Usage:
    # Dry run (5 steps):
    python scripts/train_unsloth.py --max_steps 5

    # Full training:
    python scripts/train_unsloth.py
"""

import argparse
import json

from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

# ---------- Config ----------
MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./outputs/deltacoder-9b-v1.2"
LORA_R = 64
LORA_ALPHA = 32


def parse_args():
    parser = argparse.ArgumentParser(description="DeltaCoder v1.2 SFT training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/v1.2_sft_train.jsonl",
        help="Path to preprocessed training JSONL",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override epochs with fixed step count (e.g. 5 for dry run)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory for checkpoints and adapter",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- Load model ----------
    # Must use FastVisionModel for Qwen3.5 (it's a VLM architecture)
    print("Loading model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # No QLoRA for Qwen3.5
        load_in_16bit=True,  # bf16 LoRA
        trust_remote_code=True,
    )

    # ---------- LoRA ----------
    print("Applying LoRA...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Text-only SFT — skip vision encoder
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # ---------- Dataset ----------
    # UnslothVisionDataCollator expects a list of {"messages": [...]} dicts.
    # It applies the chat template internally via the tokenizer.
    # Pass messages directly — do NOT pre-format to text.
    print(f"Loading dataset from {args.dataset}...")

    dataset = []
    skipped = 0
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                messages = row["messages"]
                if isinstance(messages, str):
                    messages = json.loads(messages)

                # Validate messages are list of dicts with role/content
                if not isinstance(messages, list) or len(messages) < 2:
                    skipped += 1
                    continue

                # UnslothVisionDataCollator's Jinja template requires content to be
                # a list of typed blocks: [{"type": "text", "text": "..."}]
                # Convert plain string content to this format.
                converted = []
                for msg in messages:
                    if not isinstance(msg, dict) or "role" not in msg:
                        raise ValueError(f"Invalid message format: {msg}")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    converted.append({"role": msg["role"], "content": content})

                dataset.append({"messages": converted})
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  Skipping row: {e}")

    print(f"Prepared {len(dataset)} rows ({skipped} skipped)")

    # ---------- Trainer ----------
    print("Setting up trainer...")
    FastVisionModel.for_training(model)

    sft_config_kwargs = dict(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        warmup_ratio=0.05,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=0.25,
        save_total_limit=3,
        output_dir=args.output_dir,
        optim="adamw_torch",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        seed=3407,
        report_to="none",
        # Required for VLM SFT with UnslothVisionDataCollator:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
    )

    if args.max_steps > 0:
        sft_config_kwargs["max_steps"] = args.max_steps

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=SFTConfig(**sft_config_kwargs),
    )

    # ---------- Train ----------
    print("Starting training...")
    trainer.train()

    # ---------- Save ----------
    adapter_path = f"{args.output_dir}/lora_adapter"
    print(f"Saving model to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print("Done!")


if __name__ == "__main__":
    main()
