"""
Fix DeltaCoder v1.2 merge: SFT LoRA keys have 'language_model.' prefix
that doesn't match AutoModelForCausalLM (text-only) module paths.

Remaps SFT adapter keys, then does proper two-stage merge:
  1. Base (text-only) + remapped SFT LoRA -> merged SFT
  2. merged SFT + DPO LoRA -> merged final

Run on Romulus:
  python fix_merge.py --base-model Qwen/Qwen3.5-9B ^
      --sft-adapter D:\AI\DeltaCoder\v1.2\sft_adapter ^
      --dpo-adapter D:\AI\DeltaCoder\v1.2\dpo_adapter ^
      --output-dir D:\AI\DeltaCoder\v1.2\merged_fixed
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix SFT LoRA key mismatch and re-merge"
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sft-adapter", type=str, required=True)
    parser.add_argument("--dpo-adapter", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip weight difference verification"
    )
    return parser.parse_args()


def remap_sft_adapter(sft_adapter_dir: str, temp_dir: str) -> str:
    """Copy SFT adapter to temp dir with remapped key names.

    Strips 'language_model.' from key paths so they match
    AutoModelForCausalLM module paths.

    VLM path:  base_model.model.model.language_model.layers.0.linear_attn.in_proj_a.lora_A.weight
    Text path: base_model.model.model.layers.0.linear_attn.in_proj_a.lora_A.weight
    """
    print("=== Remapping SFT LoRA keys ===")

    # Load original weights
    weights_path = os.path.join(sft_adapter_dir, "adapter_model.safetensors")
    weights = load_file(weights_path)

    # Remap keys
    remapped = {}
    changed = 0
    for key, tensor in weights.items():
        new_key = key.replace(".language_model.", ".")
        if new_key != key:
            changed += 1
        remapped[new_key] = tensor

    print(f"  Remapped {changed}/{len(weights)} keys (stripped 'language_model.')")

    # Save remapped weights to temp dir
    fixed_dir = os.path.join(temp_dir, "sft_adapter_fixed")
    os.makedirs(fixed_dir, exist_ok=True)

    save_file(remapped, os.path.join(fixed_dir, "adapter_model.safetensors"))

    # Copy config and other files
    for fname in os.listdir(sft_adapter_dir):
        if fname == "adapter_model.safetensors":
            continue
        src = os.path.join(sft_adapter_dir, fname)
        dst = os.path.join(fixed_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Verify a remapped key looks right
    sample_key = list(remapped.keys())[0]
    print(f"  Sample remapped key: {sample_key}")

    return fixed_dir


def verify_weights_differ(model, base_model_id: str, device="cpu"):
    """Verify merged weights differ from base by checking a few layers."""
    print("\n=== Verifying weights differ from base ===")

    # Load just the base model state dict for comparison (memory efficient: load one shard)
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    # Download just the first shard
    shard_path = hf_hub_download(
        base_model_id,
        "model.safetensors",
        cache_dir=None,
    )
    base_weights = safetensors.torch.load_file(shard_path)

    # Compare a few layers
    test_keys = [
        "model.layers.0.linear_attn.in_proj_a.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.15.linear_attn.out_proj.weight",
        "model.layers.31.mlp.down_proj.weight",
    ]

    model_state = model.state_dict()

    all_differ = True
    for key in test_keys:
        if key in base_weights and key in model_state:
            base_tensor = base_weights[key].to(device)
            merged_tensor = model_state[key].to(device)

            if torch.equal(base_tensor, merged_tensor):
                print(
                    f"  WARNING: {key} is IDENTICAL to base (LoRA may not have been applied)"
                )
                all_differ = False
            else:
                diff = (base_tensor.float() - merged_tensor.float()).abs().mean().item()
                print(f"  OK: {key} differs from base (mean abs diff: {diff:.6f})")
        else:
            print(f"  SKIP: {key} not found in one of the models")

    del base_weights

    if all_differ:
        print("  All checked weights differ from base - merge looks correct!")
    else:
        print("  WARNING: Some weights are identical to base!")

    return all_differ


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Remap SFT adapter keys
        fixed_sft_dir = remap_sft_adapter(args.sft_adapter, temp_dir)

        # Step 2: Load base model (text-only)
        print(f"\n=== Loading base model {args.base_model} (text-only) ===")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )

        # Show model module names for debugging
        sample_params = [n for n, _ in model.named_parameters()][:5]
        print(f"  Model param names (first 5): {sample_params}")

        # Step 3: Apply remapped SFT LoRA
        print(f"\n=== Applying SFT LoRA (remapped keys) ===")
        model = PeftModel.from_pretrained(model, fixed_sft_dir)
        model = model.merge_and_unload()
        print("  SFT LoRA merged successfully!")

        # Step 4: Apply DPO LoRA
        print(f"\n=== Applying DPO LoRA ({args.dpo_adapter}) ===")
        model = PeftModel.from_pretrained(model, args.dpo_adapter)
        model = model.merge_and_unload()
        print("  DPO LoRA merged successfully!")

    # Step 5: Verify weights differ from base
    if not args.skip_verify:
        verify_weights_differ(model, args.base_model)

    # Step 6: Save merged model
    print(f"\n=== Saving merged model -> {args.output_dir} ===")
    model.save_pretrained(args.output_dir, max_shard_size="99GB")
    tokenizer.save_pretrained(args.output_dir)

    # Check what we saved
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    print(f"  model_type: {config.get('model_type')}")
    print(f"  architectures: {config.get('architectures')}")

    # Report file sizes
    total_size = 0
    for fname in os.listdir(args.output_dir):
        fpath = os.path.join(args.output_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            if size > 1_000_000:
                print(f"  {fname}: {size / 1e9:.2f} GB")
    print(f"  Total: {total_size / 1e9:.2f} GB")

    print("\n=== Done! ===")
    print(f"Merged model at: {args.output_dir}")
    print(f"Next: Convert to GGUF with llama.cpp convert_hf_to_gguf.py")


if __name__ == "__main__":
    main()
