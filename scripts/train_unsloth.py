"""
Qwen3.5-DeltaCoder-9B v1.2 — Unsloth LoRA SFT training script.
Loads preprocessed JSONL from local disk, uses Unsloth's native SFT pipeline.

Usage:
    # Dry run (5 steps):
    python scripts/train_unsloth.py --max_steps 5

    # Full training:
    python scripts/train_unsloth.py
"""

import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
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
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,  # No QLoRA for Qwen3.5
        load_in_16bit=True,  # bf16 LoRA
        full_finetuning=False,
        trust_remote_code=True,
    )

    # ---------- LoRA ----------
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            # Full Attention (8/32 layers)
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # GDN (24/32 layers) — Qwen3.5-specific, MUST include these
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
            # MLP (all 32 layers)
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # ---------- Dataset ----------
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files={"train": args.dataset}, split="train")
    print(f"Loaded {len(dataset)} rows")

    # Format messages for chat template
    def format_messages(example):
        """Apply chat template to messages, including tools if present."""
        messages = example["messages"]
        tools = example.get("tools")

        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    print("Formatting dataset...")
    dataset = dataset.map(
        format_messages, num_proc=1, remove_columns=dataset.column_names
    )
    print(f"Formatted {len(dataset)} rows")

    # ---------- Trainer ----------
    print("Setting up trainer...")

    sft_config_kwargs = dict(
        max_seq_length=MAX_SEQ_LENGTH,
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
        dataset_text_field="text",
        packing=True,
        dataset_num_proc=1,  # CRITICAL: Qwen3.5 tokenizer crashes with multiprocessing
        report_to="none",
    )

    if args.max_steps > 0:
        sft_config_kwargs["max_steps"] = args.max_steps

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
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
