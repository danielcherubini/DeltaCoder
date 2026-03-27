"""
DeltaCoder v1.1-DPO — Merge LoRA adapters + export to GGUF.

Two-stage merge:
  1. Qwen/Qwen3.5-9B + v1 SFT LoRA (danielcherubini/Qwen3.5-DeltaCoder-9B)
     → merged_v1 (full bf16 weights)
  2. merged_v1 + DPO LoRA (outputs/deltacoder-9b-dpo/lora_adapter)
     → merged_v1.1 (full bf16 weights)
  3. Convert merged_v1.1 → GGUF (f16)
  4. Quantize f16 → all quants
  5. Upload to HuggingFace (optional)

Usage:
    python scripts/merge_and_export_dpo.py

    # With upload:
    python scripts/merge_and_export_dpo.py --upload --hf-token <TOKEN>
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SANITY_PROMPT = "Write a Python function that reverses a list."
QUANTS = [
    # Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_K_S already uploaded
    "Q4_K_M",  # 4-bit (recommended)
    "Q5_K_S",
    "Q5_0",
    "Q5_K_M",  # 5-bit
    "Q6_K",  # 6-bit
    "Q8_0",  # 8-bit
    "BF16",  # 16-bit (raw, for high-VRAM users)
]
HF_GGUF_REPO = "danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF"
HF_ADAPTER_REPO = "danielcherubini/Qwen3.5-DeltaCoder-9B"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge DPO LoRA and export to GGUF")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Base model HuggingFace ID",
    )
    parser.add_argument(
        "--sft-adapter",
        type=str,
        default="danielcherubini/Qwen3.5-DeltaCoder-9B",
        help="v1 SFT LoRA adapter (HF repo or local path)",
    )
    parser.add_argument(
        "--dpo-adapter",
        type=str,
        default="./outputs/deltacoder-9b-dpo/lora_adapter",
        help="DPO LoRA adapter (local path)",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="./outputs/deltacoder-9b-dpo-merged",
        help="Output directory for merged bf16 model",
    )
    parser.add_argument(
        "--gguf-dir",
        type=str,
        default="./outputs/deltacoder-9b-dpo-gguf",
        help="Output directory for GGUF files",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default="./llama.cpp",
        help="Path to llama.cpp directory (will be cloned if not present)",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip inference sanity check",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload GGUFs and adapter to HuggingFace after export",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace write token (defaults to HF_TOKEN env var)",
    )
    return parser.parse_args()


def run(cmd: str, check: bool = True) -> int:
    """Run a shell command, streaming output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(
            f"ERROR: command failed with exit code {result.returncode}", file=sys.stderr
        )
        sys.exit(result.returncode)
    return result.returncode


def sanity_check(merged_dir: str) -> bool:
    """Load merged model and run a quick inference to verify weights are valid."""
    print("\n=== Sanity Check ===")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_dir, trust_remote_code=True)

        messages = [{"role": "user", "content": SANITY_PROMPT}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=256, temperature=0.6, do_sample=True
            )
        response = tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )

        print(f"\nPrompt: {SANITY_PROMPT}")
        print(f"Response:\n{response}")

        if "def" in response and "return" in response:
            print("\n✓ Sanity check PASSED")
            return True
        else:
            print("\n✗ Sanity check FAILED — response does not look like Python code")
            return False

    except Exception as e:
        print(f"\n✗ Sanity check FAILED: {e}", file=sys.stderr)
        return False


def setup_llama_cpp(llama_cpp_dir: str):
    """Clone and build llama.cpp if not already present."""
    llama_path = Path(llama_cpp_dir)
    if not llama_path.exists():
        print("\n=== Cloning llama.cpp ===")
        run(
            f"git clone --depth 1 https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}"
        )
        run(f"cmake -B {llama_cpp_dir}/build -S {llama_cpp_dir} -DGGML_CUDA=ON")
        run(
            f"cmake --build {llama_cpp_dir}/build --config Release -j$(nproc) --target llama-quantize llama-gguf-split"
        )
    else:
        print(f"\nUsing existing llama.cpp at {llama_cpp_dir}")


def main():
    args = parse_args()

    merged_dir = args.merged_dir
    gguf_dir = args.gguf_dir
    Path(gguf_dir).mkdir(parents=True, exist_ok=True)
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Step 1: Load base model ----------
    print(f"\n=== Step 1: Loading base model {args.base_model} ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # ---------- Step 2: Merge v1 SFT LoRA ----------
    print(f"\n=== Step 2: Merging v1 SFT LoRA ({args.sft_adapter}) ===")
    model = PeftModel.from_pretrained(model, args.sft_adapter)
    model = model.merge_and_unload()
    print("v1 SFT LoRA merged.")

    # ---------- Step 3: Merge DPO LoRA ----------
    print(f"\n=== Step 3: Merging DPO LoRA ({args.dpo_adapter}) ===")
    model = PeftModel.from_pretrained(model, args.dpo_adapter)
    model = model.merge_and_unload()
    print("DPO LoRA merged.")

    # ---------- Step 4: Save merged model ----------
    print(f"\n=== Step 4: Saving merged model → {merged_dir} ===")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to: {merged_dir}")

    # Free memory before sanity check
    del model
    torch.cuda.empty_cache()

    # ---------- Step 5: Sanity check ----------
    if not args.skip_sanity:
        passed = sanity_check(merged_dir)
        if not passed:
            print("\nAborting GGUF conversion due to failed sanity check.")
            print("Re-run with --skip-sanity to force conversion anyway.")
            sys.exit(1)
    else:
        print("\nSkipping sanity check (--skip-sanity flag set).")

    # ---------- Step 6: Setup llama.cpp ----------
    setup_llama_cpp(args.llama_cpp_dir)

    # ---------- Step 7: Convert to GGUF (f16) ----------
    f16_gguf = f"{gguf_dir}/DeltaCoder-9B-v1.1-DPO-f16.gguf"
    print(f"\n=== Step 7: Converting to GGUF (f16) → {f16_gguf} ===")
    run(
        f"python {args.llama_cpp_dir}/convert_hf_to_gguf.py {merged_dir} "
        f"--outfile {f16_gguf} --outtype f16"
    )

    # Delete merged model immediately after f16 conversion — no longer needed
    print(f"  Removing merged model to free disk ({merged_dir})...")
    run(f"rm -rf {merged_dir}")

    # ---------- Step 8: Quantize + upload-and-delete each quant ----------
    print("\n=== Step 8: Generating quantized GGUFs ===")
    quantize_bin = f"{args.llama_cpp_dir}/build/bin/llama-quantize"

    token = None
    if args.upload:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print(
                "\nERROR: --upload requires --hf-token or HF_TOKEN env var",
                file=sys.stderr,
            )
            sys.exit(1)

    for quant in QUANTS:
        if quant == "BF16":
            out = f"{gguf_dir}/DeltaCoder-9B-v1.1-DPO-BF16.gguf"
            run(f"cp {f16_gguf} {out}")
        else:
            out = f"{gguf_dir}/DeltaCoder-9B-v1.1-DPO-{quant}.gguf"
            print(f"  Quantizing {quant}...")
            run(f"{quantize_bin} {f16_gguf} {out} {quant}")

        # Upload immediately and delete to free disk
        if args.upload and token:
            print(f"  Uploading {quant} to HuggingFace...")
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=out,
                path_in_repo=Path(out).name,
                repo_id=HF_GGUF_REPO,
                repo_type="model",
            )
            Path(out).unlink(missing_ok=True)
            print(f"  Deleted local {quant} GGUF to free disk.")

    # Delete f16 GGUF after all quants done
    print("  Removing intermediate f16 GGUF...")
    Path(f16_gguf).unlink(missing_ok=True)

    print("\n=== Done! ===")

    # ---------- Step 9: Upload adapter ----------
    if args.upload and token:
        print(f"\n=== Step 9: Uploading DPO adapter to {HF_ADAPTER_REPO} ===")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.upload_folder(
            folder_path=args.dpo_adapter,
            repo_id=HF_ADAPTER_REPO,
            repo_type="model",
        )
        print("\nAll uploads complete!")
    else:
        print(f"""
Next steps:
  1. Download GGUFs:
       rsync -avP <instance>:{gguf_dir}/*.gguf ./

  2. Test with ik_llama:
       ./ik_llama-server -m DeltaCoder-9B-v1.1-DPO-Q4_K_M.gguf -ngl 999 -c 4096 -fa 1 --jinja --port 8080

  3. Upload to HuggingFace:
       python scripts/merge_and_export_dpo.py --upload --hf-token <TOKEN>
""")


if __name__ == "__main__":
    main()
