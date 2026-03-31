"""
Qwen3.5-DeltaCoder-9B — Unsloth LoRA SFT training script.
Loads CoderForge directly from HuggingFace, uses Unsloth's native SFT pipeline.
"""

from unsloth import FastLanguageModel
import json
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------- Config ----------
MODEL_NAME = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "./outputs/deltacoder-9b"
LORA_R = 64
LORA_ALPHA = 32
SUBSET_SIZE = 50_000

# ---------- Load model ----------
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
    trust_remote_code=True,
)

# ---------- LoRA ----------
print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        # Full Attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # GDN
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        # MLP
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
)

# ---------- Dataset ----------
print("Loading CoderForge from HuggingFace...")
dataset = load_dataset(
    "togethercomputer/CoderForge-Preview",
    name="trajectories",
    split="filtered_reward1",
    trust_remote_code=True,
)
print(f"Full dataset: {len(dataset)} rows")

# Subsample
dataset = dataset.shuffle(seed=3407).select(range(min(SUBSET_SIZE, len(dataset))))
print(f"Using {len(dataset)} rows")


# Parse messages JSON and format for chat template
def format_messages(example):
    """Parse the messages JSON string into chat format."""
    messages = json.loads(example["messages"])
    # Apply chat template to get formatted text
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


print("Formatting dataset...")
dataset = dataset.map(format_messages, num_proc=16, remove_columns=dataset.column_names)
print(f"Formatted {len(dataset)} rows")

# ---------- Trainer ----------
print("Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=0.25,
        save_total_limit=3,
        output_dir=OUTPUT_DIR,
        optim="adamw_torch",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        seed=3407,
        dataset_text_field="text",
        packing=True,
        dataset_num_proc=16,
        report_to="none",
    ),
)

# ---------- Train ----------
print("Starting training...")
trainer.train()

# ---------- Save ----------
print("Saving model...")
model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
print("Done!")
