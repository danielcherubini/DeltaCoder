"""
DeltaCoder v1.1-DPO — Merge LoRA adapter + export to GGUF.

Loads the merged v1 base model and the DPO LoRA adapter, merges them,
runs an inference sanity check, then exports to GGUF via llama.cpp.

Usage (run on Vast.ai after training completes):
    python scripts/merge_and_export_dpo.py

    # Custom paths:
    python scripts/merge_and_export_dpo.py \\
        --base-model danielcherubini/Qwen3.5-DeltaCoder-9B \\
        --adapter ./outputs/deltacoder-9b-dpo/lora_adapter \\
        --merged-dir ./outputs/deltacoder-9b-dpo-merged \\
        --gguf-dir ./outputs/deltacoder-9b-dpo-gguf

Then download GGUFs and push to HuggingFace:
    huggingface-cli upload danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF \\
        ./outputs/deltacoder-9b-dpo-gguf/ --repo-type model
"""

import argparse
import subprocess
import sys
from pathlib import Path

from unsloth import FastLanguageModel


SANITY_PROMPT = "Write a Python function that reverses a list."
QUANTS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge DPO LoRA and export to GGUF")
    parser.add_argument(
        "--base-model",
        type=str,
        default="danielcherubini/Qwen3.5-DeltaCoder-9B",
        help="HuggingFace model ID or local path for the merged v1 base model",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./outputs/deltacoder-9b-dpo/lora_adapter",
        help="Path to the DPO LoRA adapter directory",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default="./outputs/deltacoder-9b-dpo-merged",
        help="Output directory for the merged fp16 model (save_method='merged_16bit')",
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
        help="Skip the inference sanity check (not recommended)",
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
    print("\n=== Sanity Check: Loading merged model for inference ===")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=merged_dir,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_16bit=True,
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)

        messages = [{"role": "user", "content": SANITY_PROMPT}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.6,
            do_sample=True,
        )
        response = tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )

        print(f"\nSanity check prompt: {SANITY_PROMPT}")
        print(f"Model response:\n{response}")

        # Basic validation: response should contain "def" and "return"
        if "def" in response and "return" in response:
            print("\n✓ Sanity check PASSED — model produces valid Python code")
            return True
        else:
            print("\n✗ Sanity check FAILED — response does not look like Python code")
            print(
                "  Inspect the response above before proceeding with GGUF conversion."
            )
            return False

    except Exception as e:
        print(f"\n✗ Sanity check FAILED with exception: {e}", file=sys.stderr)
        return False


def setup_llama_cpp(llama_cpp_dir: str):
    """Clone and build llama.cpp if not already present."""
    llama_path = Path(llama_cpp_dir)
    if not llama_path.exists():
        print("\n=== Cloning llama.cpp ===")
        run(
            f"git clone --depth 1 https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}"
        )
        run(f"make -C {llama_cpp_dir} -j$(nproc) llama-quantize")
    else:
        print(f"\nUsing existing llama.cpp at {llama_cpp_dir}")
        # Ensure quantize binary exists
        if not (llama_path / "llama-quantize").exists():
            run(f"make -C {llama_cpp_dir} -j$(nproc) llama-quantize")


def main():
    args = parse_args()

    merged_dir = args.merged_dir
    gguf_dir = args.gguf_dir
    Path(gguf_dir).mkdir(parents=True, exist_ok=True)

    # ---------- Step 1: Load base + adapter and merge ----------
    print("\n=== Step 1: Loading base model + DPO LoRA adapter ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=4096,
        load_in_4bit=False,
        load_in_16bit=True,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {args.adapter}")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)

    print(f"\n=== Step 2: Merging LoRA into base model → {merged_dir} ===")
    model.save_pretrained_merged(
        merged_dir,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged model saved to: {merged_dir}")

    # ---------- Step 3: Sanity check ----------
    if not args.skip_sanity:
        passed = sanity_check(merged_dir)
        if not passed:
            print("\nAborting GGUF conversion due to failed sanity check.")
            print("Re-run with --skip-sanity to force conversion anyway.")
            sys.exit(1)
    else:
        print("\nSkipping sanity check (--skip-sanity flag set).")

    # ---------- Step 4: Setup llama.cpp ----------
    setup_llama_cpp(args.llama_cpp_dir)

    # ---------- Step 5: Convert to GGUF (f16) ----------
    f16_gguf = f"{gguf_dir}/DeltaCoder-9B-v1.1-DPO-f16.gguf"
    print(f"\n=== Step 5: Converting to GGUF (f16) → {f16_gguf} ===")
    run(
        f"python {args.llama_cpp_dir}/convert_hf_to_gguf.py {merged_dir} "
        f"--outfile {f16_gguf} --outtype f16"
    )

    # ---------- Step 6: Quantize ----------
    print("\n=== Step 6: Generating quantized GGUFs ===")
    quantize_bin = f"{args.llama_cpp_dir}/llama-quantize"
    for quant in QUANTS:
        out = f"{gguf_dir}/DeltaCoder-9B-v1.1-DPO-{quant}.gguf"
        print(f"  Quantizing {quant}...")
        run(f"{quantize_bin} {f16_gguf} {out} {quant}")

    # ---------- Done ----------
    print("\n=== Done! ===")
    run(f"ls -lh {gguf_dir}/*.gguf", check=False)
    print(f"""
Next steps:
  1. Download GGUFs from the cloud box:
       rsync -avP {gguf_dir}/*.gguf user@host:/path/to/local/

  2. Test locally with ik_llama:
       ./ik_llama-server -m {gguf_dir}/DeltaCoder-9B-v1.1-DPO-Q4_K_M.gguf \\
           -ngl 999 -c 4096 -fa 1 --jinja --port 8080

  3. Run Terminal-Bench eval:
       OPENAI_API_KEY=sk-none harbor run \\
           --path /home/daniel/Coding/AI/terminal-bench-2 \\
           --task-name fix-git --task-name cobol-modernization \\
           --task-name overfull-hbox --task-name prove-plus-comm \\
           --agent terminus-2 --model openai/deltacoder \\
           --ak api_base=http://romulus:11434/v1 --ak temperature=0.6 \\
           --ak 'model_info={{"max_input_tokens":65536,"max_output_tokens":8192,"input_cost_per_token":0,"output_cost_per_token":0,"litellm_provider":"openai","mode":"chat"}}' \\
           --ae GIT_PAGER=cat --ae GIT_EDITOR=true --ae GIT_SEQUENCE_EDITOR=true \\
           --ek GIT_PAGER=cat --ek GIT_EDITOR=true --ek GIT_SEQUENCE_EDITOR=true \\
           --env docker -n 1 --job-name deltacoder-v1.1-dpo-eval \\
           --jobs-dir /home/daniel/Coding/AI/terminal-bench-2/jobs

  4. Push to HuggingFace:
       huggingface-cli upload danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF \\
           {gguf_dir}/ --repo-type model
       huggingface-cli upload danielcherubini/Qwen3.5-DeltaCoder-9B \\
           {args.adapter}/ --repo-type model
""")


if __name__ == "__main__":
    main()
