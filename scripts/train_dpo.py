"""
Qwen3.5-DeltaCoder-9B — DPO training script (v1.1).
Uses standard HuggingFace + PEFT + TRL (no Unsloth) to avoid vision model
detection issues with Qwen3.5.

Usage:
    # Quick test (50 steps, verify loss moves before full run):
    python scripts/train_dpo.py --max-steps 50

    # Full training:
    python scripts/train_dpo.py
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# ---------- Config ----------
MODEL_NAME = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH = 4096
LORA_R = 32
LORA_ALPHA = 32

# Proven LoRA target_modules from v1 — includes GDN projections.
# Omitting the GDN projections would leave 75% of attention layers untrained.
LORA_TARGET_MODULES = [
    # Full Attention
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # GDN (Gated DeltaNet) — Qwen3.5-specific, 24/32 layers
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    # MLP
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for DeltaCoder v1.1")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs with a fixed step count (e.g. 50 for a quick test run)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dpo_pairs.jsonl",
        help="Path to DPO pairs JSONL in conversational format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/deltacoder-9b-dpo",
        help="Output directory for checkpoints and final adapter",
    )
    return parser.parse_args()


def load_dpo_dataset(path: str) -> Dataset:
    """Load conversational-format JSONL into a HuggingFace Dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} preference pairs from {path}")
    return Dataset.from_list(records)


def main():
    args = parse_args()

    # ---------- Load model ----------
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Qwen3.5 GDN layers don't support SDPA/FA2
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Ensure pad token is set (needed for DPO batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- LoRA ----------
    print(f"Applying LoRA (r={LORA_R})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.enable_input_require_grads()  # Required for gradient checkpointing + LoRA
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------- Dataset ----------
    print("Loading DPO pairs...")
    dataset = load_dpo_dataset(args.data)

    # ---------- DPO Config ----------
    dpo_config_kwargs = dict(
        beta=0.1,  # DPO temperature
        loss_type="sigmoid",  # Standard DPO loss
        max_length=MAX_SEQ_LENGTH,  # Max total (prompt + response) length
        max_prompt_length=1024,  # Truncate prompts beyond this
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch size = 16
        num_train_epochs=1,
        learning_rate=5e-6,  # ~20x lower than SFT — DPO is LR-sensitive
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="steps",
        save_steps=0.5,  # Save mid-run checkpoint
        save_total_limit=2,
        output_dir=args.output_dir,
        seed=3407,
        report_to="none",
        gradient_checkpointing=True,
    )

    if args.max_steps > 0:
        dpo_config_kwargs["max_steps"] = args.max_steps

    dpo_config = DPOConfig(**dpo_config_kwargs)

    # ---------- Trainer ----------
    print("Setting up DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Uses implicit reference (PEFT base weights)
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ---------- Train ----------
    n_steps = args.max_steps if args.max_steps > 0 else "full epoch"
    print(f"Starting DPO training ({len(dataset)} pairs, {n_steps} steps)...")
    print(f"  LR: 5e-6  |  beta: 0.1  |  effective batch: 16  |  r: {LORA_R}")
    print("  Watch: rewards/chosen should increase, rewards/rejected should decrease")

    trainer_stats = trainer.train()

    # ---------- Save ----------
    adapter_path = f"{args.output_dir}/lora_adapter"
    print(f"Saving LoRA adapter to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\nDone! Training stats: {trainer_stats}")


if __name__ == "__main__":
    main()
