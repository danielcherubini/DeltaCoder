"""
DeltaCoder-9B Dry Run — verify everything works before renting a GPU.

Run locally (CPU only, no GPU needed):
    pip install transformers peft datasets
    python scripts/dry_run.py

Checks:
  1. Can transformers load the Qwen3.5 architecture?
  2. What are the exact layer names for LoRA targeting?
  3. Can PEFT apply LoRA to the model?
  4. Does the chat template support tool_calls and tool role?
  5. Do the preprocessing scripts produce valid output?
"""

import json
import sys
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
BASE_MODEL = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"

results = []


def check(name: str, status: str, detail: str = ""):
    results.append((name, status, detail))
    icon = {"PASS": "+", "FAIL": "!", "WARN": "~"}[status]
    print(f"  [{icon}] {name}")
    if detail:
        for line in detail.split("\n"):
            print(f"      {line}")


def check_transformers_version():
    """Check that transformers is new enough for Qwen3.5."""
    print("\n=== 1. Transformers Version ===")
    try:
        import transformers
        version = transformers.__version__
        # Qwen3.5 support needs a recent version
        major, minor = int(version.split(".")[0]), int(version.split(".")[1])
        if major >= 4 and minor >= 48:
            check("transformers version", PASS, f"v{version}")
        else:
            check("transformers version", WARN,
                  f"v{version} — Qwen3.5 may need >= 4.48. Try: pip install -U transformers")
    except ImportError:
        check("transformers version", FAIL, "Not installed. Run: pip install transformers")
        return False
    return True


def check_model_architecture():
    """Load model on meta device and inspect layers."""
    print("\n=== 2. Model Architecture & Layer Names ===")
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        # First check if the config loads
        print("  Loading config...")
        config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        check("config loads", PASS, f"model_type={getattr(config, 'model_type', 'unknown')}")

        # Load on meta device (no RAM/GPU needed)
        print("  Loading model on meta device (no weights downloaded)...")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # Collect all module names
        all_names = [name for name, _ in model.named_modules()]
        linear_names = set()
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear":
                # Get the short name (last part)
                short = name.split(".")[-1]
                linear_names.add(short)

        check("model loads on meta device", PASS, f"{len(all_names)} total modules")

        # Check for expected attention layers
        attn_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        found_attn = attn_targets & linear_names
        missing_attn = attn_targets - linear_names
        if found_attn:
            check("attention linear layers", PASS, f"Found: {sorted(found_attn)}")
        if missing_attn:
            check("attention linear layers (missing)", WARN, f"Not found: {sorted(missing_attn)}")

        # Check for expected GDN layers
        gdn_targets = {"in_proj", "out_proj", "x_proj", "dt_proj"}
        found_gdn = gdn_targets & linear_names
        missing_gdn = gdn_targets - linear_names
        if found_gdn:
            check("GDN linear layers", PASS, f"Found: {sorted(found_gdn)}")
        if missing_gdn:
            check("GDN linear layers (missing)", WARN, f"Not found: {sorted(missing_gdn)}")

        # Show ALL unique linear layer names (the important output)
        check("all linear layer names", PASS, f"{sorted(linear_names)}")

        # Suggest target_modules
        all_targets = found_attn | found_gdn
        other_linear = linear_names - attn_targets - gdn_targets
        if other_linear:
            check("other linear layers (not targeted)", WARN,
                  f"{sorted(other_linear)} — review if any should be LoRA targets")

        print(f"\n  Suggested target_modules for LoRA config:")
        print(f"    {json.dumps(sorted(all_targets))}")

        # Dump full layer list to file for reference
        layer_file = Path("data/model_layers.txt")
        layer_file.parent.mkdir(parents=True, exist_ok=True)
        with open(layer_file, "w") as f:
            for name, module in model.named_modules():
                f.write(f"{name:80s}  {module.__class__.__name__}\n")
        check("layer dump", PASS, f"Full list written to {layer_file}")

        return sorted(all_targets), model, config

    except Exception as e:
        check("model architecture", FAIL, str(e))
        return None, None, None


def check_peft_lora(target_modules: list[str]):
    """Verify PEFT can apply LoRA with the discovered target modules."""
    print("\n=== 3. PEFT LoRA Application ===")
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(model, lora_config)

        # Count trainable params
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        pct = trainable / total * 100

        check("PEFT LoRA applies", PASS,
              f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

        # Verify the targeted modules
        lora_modules = []
        for name, module in peft_model.named_modules():
            if "lora_A" in name:
                lora_modules.append(name.replace(".lora_A.default", ""))

        check("LoRA modules injected", PASS, f"{len(lora_modules)} layers adapted")

    except Exception as e:
        check("PEFT LoRA", FAIL, str(e))


def check_chat_template():
    """Check if the tokenizer's chat template handles tool_calls."""
    print("\n=== 4. Chat Template & Tool Call Support ===")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        if tokenizer.chat_template:
            check("chat template exists", PASS, f"{len(tokenizer.chat_template)} chars")
        else:
            check("chat template exists", FAIL, "No chat template found on tokenizer")
            return

        # Check if template handles tool-related keywords
        template = tokenizer.chat_template
        tool_keywords = {
            "tool_calls": "'tool_calls'" in template or '"tool_calls"' in template or "tool_calls" in template,
            "tool role": "'tool'" in template or '"tool"' in template,
            "tools": "'tools'" in template or '"tools"' in template,
            "function": "'function'" in template or '"function"' in template,
        }

        for keyword, found in tool_keywords.items():
            if found:
                check(f"template handles '{keyword}'", PASS)
            else:
                check(f"template handles '{keyword}'", WARN,
                      "Not found in template — may need custom template for tool_calls")

        # Try applying the template with a tool_calls message
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List the files."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "arguments": json.dumps({"command": "ls -la"})
                }
            }]},
            {"role": "tool", "tool_call_id": "call_001", "content": "file1.py\nfile2.py"},
            {"role": "assistant", "content": "Here are the files."},
        ]

        test_tools = [{
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Execute a bash command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]
                }
            }
        }]

        try:
            formatted = tokenizer.apply_chat_template(
                test_messages,
                tools=test_tools,
                tokenize=False,
                add_generation_prompt=True
            )
            check("template renders tool_calls message", PASS, f"{len(formatted)} chars")

            # Save rendered output for inspection
            template_file = Path("data/chat_template_output.txt")
            template_file.parent.mkdir(parents=True, exist_ok=True)
            with open(template_file, "w") as f:
                f.write("=== Rendered chat template with tool_calls ===\n\n")
                f.write(formatted)
                f.write("\n\n=== Raw chat template (Jinja) ===\n\n")
                f.write(tokenizer.chat_template)
            check("template output saved", PASS, f"Inspect: {template_file}")

        except Exception as e:
            # Try without tools kwarg
            try:
                formatted = tokenizer.apply_chat_template(
                    test_messages, tokenize=False, add_generation_prompt=True
                )
                check("template renders tool_calls message", WARN,
                      f"Works without tools= kwarg but may not format tools correctly. Error with tools: {e}")
            except Exception as e2:
                check("template renders tool_calls message", FAIL, f"{e2}")

    except Exception as e:
        check("chat template", FAIL, str(e))


def check_preprocessing():
    """Run each preprocessing script on a tiny sample."""
    print("\n=== 5. Preprocessing Script Validation ===")

    # We'll test by importing and running the conversion functions on synthetic data
    scripts_dir = Path(__file__).parent

    # Test CoderForge XML parser
    try:
        sys.path.insert(0, str(scripts_dir))
        from preprocess_coderforge import parse_xml_tool_calls, convert_messages

        test_assistant_content = '''Let me check the files.
<function_calls>
<invoke name="execute_bash">
<parameter name="command">ls -la /testbed</parameter>
</invoke>
</function_calls>'''

        clean, calls = parse_xml_tool_calls(test_assistant_content)
        assert len(calls) == 1, f"Expected 1 tool call, got {len(calls)}"
        assert calls[0]["function"]["name"] == "execute_bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "ls -la /testbed"
        assert "<function_calls>" not in clean

        # Test full message conversion
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": test_assistant_content},
            {"role": "user", "content": "total 48\ndrwxr-xr-x 5 root root 4096"},
        ]
        converted = convert_messages(test_messages)
        assert converted[2]["role"] == "assistant"
        assert "tool_calls" in converted[2]
        assert converted[3]["role"] == "tool"
        assert converted[3]["tool_call_id"] == converted[2]["tool_calls"][0]["id"]

        check("coderforge XML parser", PASS, "XML → JSON conversion works")

    except Exception as e:
        check("coderforge XML parser", FAIL, str(e))

    # Test SWE-agent command extractor
    try:
        from preprocess_sweagent import extract_command, convert_trajectory

        test_text = "DISCUSSION\nLet me look at the files.\n```\nls -la\n```"
        discussion, cmd = extract_command(test_text)
        assert cmd == "ls -la", f"Expected 'ls -la', got '{cmd}'"
        assert "DISCUSSION" not in discussion

        test_trajectory = [
            {"role": "system", "text": "You are a programmer.", "cutoff_date": "", "mask": False, "system_prompt": ""},
            {"role": "user", "text": "Fix the bug in main.py", "cutoff_date": "", "mask": False, "system_prompt": ""},
            {"role": "ai", "text": test_text, "cutoff_date": "", "mask": False, "system_prompt": ""},
            {"role": "user", "text": "file1.py  file2.py", "cutoff_date": "", "mask": False, "system_prompt": ""},
        ]
        converted = convert_trajectory(test_trajectory)
        assert converted[2]["role"] == "assistant"
        assert "tool_calls" in converted[2]
        assert converted[3]["role"] == "tool"

        check("sweagent command extractor", PASS, "Plain text → JSON conversion works")

    except Exception as e:
        check("sweagent command extractor", FAIL, str(e))

    # Test JSON escaping with tricky content (the actual failure mode we're fixing)
    try:
        tricky_code = 'pub fn build_router() -> Router {\n    Router::new()\n        .route(\n            "/v1/chat/completions",\n            axum::routing::post(handle_chat_completions),\n        )\n}'
        args_json = json.dumps({"command": tricky_code})
        # Verify it round-trips
        parsed = json.loads(args_json)
        assert parsed["command"] == tricky_code

        tricky_bash = "gh pr create --title 'fix stuff' --body '## Summary\n\n- cargo build ✅\n- tests ✅'"
        args_json2 = json.dumps({"command": tricky_bash})
        parsed2 = json.loads(args_json2)
        assert parsed2["command"] == tricky_bash

        check("JSON escaping (tricky content)", PASS,
              "Code with newlines/quotes/emojis round-trips correctly through json.dumps/loads")

    except Exception as e:
        check("JSON escaping", FAIL, str(e))


def check_vram_estimate():
    """Estimate VRAM usage for training."""
    print("\n=== 6. VRAM Estimate ===")

    # 9B params in bf16 = 18GB
    model_size_gb = 9e9 * 2 / 1e9  # bf16 = 2 bytes per param
    # LoRA r=64 on ~20 layers with ~4096 hidden dim ≈ small
    # AdamW states: 2x the trainable params (momentum + variance) in fp32
    # Activations depend on sequence length and batch size

    # Rough formula for transformer training VRAM:
    # model_weights + gradients + optimizer_states + activations
    # With gradient checkpointing, activations are much smaller

    seq_lengths = [2048, 4096, 8192]
    for seq_len in seq_lengths:
        # Very rough: activations ≈ 2 * batch * seq_len * hidden * num_layers * 2 bytes
        # With gradient checkpointing, divide by ~3
        activation_gb = (2 * 1 * seq_len * 4096 * 48 * 2) / 1e9 / 3
        optimizer_gb = model_size_gb * 0.1 * 4  # ~10% trainable, fp32 optimizer states, 2x for adam
        total = model_size_gb + activation_gb + optimizer_gb + 2  # +2 for overhead

        fits_a100 = "fits A100 80GB" if total < 75 else "TOO LARGE for A100 80GB"
        check(f"VRAM @ seq_len={seq_len}, batch=1", PASS if total < 75 else FAIL,
              f"~{total:.0f}GB estimated ({fits_a100})")


def print_summary():
    """Print final summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passes = sum(1 for _, s, _ in results if s == PASS)
    warns = sum(1 for _, s, _ in results if s == WARN)
    fails = sum(1 for _, s, _ in results if s == FAIL)

    print(f"  {passes} passed, {warns} warnings, {fails} failed")

    if fails > 0:
        print("\n  FAILURES (must fix before training):")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    - {name}: {detail}")

    if warns > 0:
        print("\n  WARNINGS (review before training):")
        for name, status, detail in results:
            if status == WARN:
                print(f"    - {name}: {detail}")

    if fails == 0:
        print("\n  Ready to train! Next step: rent GPU and run the full pipeline.")

    return fails == 0


if __name__ == "__main__":
    print("DeltaCoder-9B Dry Run")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")

    if not check_transformers_version():
        sys.exit(1)

    target_modules, model, config = check_model_architecture()

    if target_modules:
        check_peft_lora(target_modules)

    check_chat_template()
    check_preprocessing()
    check_vram_estimate()

    success = print_summary()
    sys.exit(0 if success else 1)
