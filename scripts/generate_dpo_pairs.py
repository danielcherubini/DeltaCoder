"""Generate DPO preference pairs by running DeltaCoder v1 against AceCode-V2-122K problems.

Downloads a configurable subset of AceCode-V2-122K from HuggingFace, shuffles and takes
n-problems rows, then calls DeltaCoder API to generate n-samples completions per problem.
Executes each completion against test cases, keeping problems where ≥1 pass AND ≥1 fail.
Formats kept pairs as conversational JSONL.
"""

import argparse
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import openai
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def parse_python_code(text: str) -> str:
    """Extract Python code from response text.

    If response contains ```python ... ``` fences, extract the content of the FIRST code block.
    Otherwise strip leading/trailing whitespace and use as-is.
    Strip any remaining ``` markers.
    """
    # Match first ```python ... ``` block
    pattern = re.compile(r"```python\s*\n(.*?)\n\s*```", re.DOTALL)
    match = pattern.search(text)

    if match:
        return match.group(1).strip()

    # Fallback: strip whitespace and remove any remaining ``` markers
    cleaned = text.strip()
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```", "", cleaned)
    return cleaned.strip()


def execute_code(code: str, tests: list[str]) -> tuple[bool, Optional[str]]:
    """Execute code against test cases and return (passed, error_message).

    Combine: extracted_code + "\n\n" + "\n".join(tests)
    Run with timeout=10, capture_output=True, stdin=DEVNULL
    Pass = returncode 0. Fail = any exception, timeout, or non-zero returncode.
    """
    try:
        tmpdir = Path(tempfile.mkdtemp())
        tmpfile = tmpdir / "test_code.py"

        try:
            # Write combined code to temp file
            combined_code = code + "\n\n" + "\n".join(tests)
            tmpfile.write_text(combined_code)

            # Use the same Python interpreter that runs this script.
            # Copy the environment but strip Python-specific vars so generated code
            # cannot import from attacker-controlled PYTHONPATH locations.
            env = os.environ.copy()
            env["PATH"] = "/usr/bin:/usr/local/bin"
            for _var in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
                env.pop(_var, None)
            result = subprocess.run(
                [sys.executable, str(tmpfile)],
                timeout=10,
                cwd=tmpdir,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
            )

            passed = result.returncode == 0
            return passed, None

        finally:
            # Always clean up tempdir
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def call_api(
    problem: dict,
    n_samples: int,
    api_base: str,
    model: str,
    extra_body: Optional[dict] = None,
) -> list[dict]:
    """Call DeltaCoder API for n_samples completions.

    temperature=0.8, max_tokens=1024.
    Pass extra_body={"enable_thinking": False} to disable Qwen3.5 thinking mode.

    Returns list of completion dicts with 'choices' array.
    """
    client = openai.OpenAI(
        base_url=api_base,
        api_key=os.environ.get(
            "OPENAI_API_KEY", "sk-none"
        ),  # local inference doesn't need a real key
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": problem["question"]}],
            temperature=0.8,
            max_tokens=1024,
            n=n_samples,
            extra_body=extra_body or {},
        )
        return response.choices

    except Exception as e:
        # Network error, timeout, etc. - skip this problem
        print(f"  API call failed (skipping): {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    parser.add_argument(
        "--n-problems", type=int, default=10000, help="Number of problems"
    )
    parser.add_argument("--n-samples", type=int, default=8, help="Samples per problem")
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8080/v1", help="API base URL"
    )
    parser.add_argument("--model", type=str, default="deltacoder", help="Model name")
    parser.add_argument(
        "--output", type=str, default="data/dpo_pairs.jsonl", help="Output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load dataset
    print("Loading AceCode-V2-122K from HuggingFace...")
    ds = load_dataset(
        "TIGER-Lab/AceCode-V2-122K", split="train", trust_remote_code=True
    )

    # Shuffle and take n-problems rows
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_problems, len(ds))))
    print(f"Using {len(ds)} problems")

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint file
    checkpoint_file = str(Path(args.output).with_suffix(".checkpoint"))

    # Load checkpoint if exists
    checkpoint = {}
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint)} problems already processed")
    # Note: we iterate all of ds and skip via checkpoint dict — do NOT slice ds,
    # because some early problems may have been skipped (API error) and need retry.

    # Output file
    with open(args.output, "a", encoding="utf-8") as f:
        for i, problem in enumerate(tqdm(ds, desc="Processing problems")):
            # Fallback to index-based id if dataset row has no "id" field
            problem_id = problem.get("id") or f"idx-{i}"
            if problem_id in checkpoint:
                continue

            # Call API for n-samples completions
            completions = call_api(
                problem,
                n_samples=args.n_samples,
                api_base=args.api_base,
                model=args.model,
                extra_body={"enable_thinking": False},
            )

            if not completions:
                continue

            # Execute each completion
            results = []  # (passed, completion_text, extracted_code)

            for choice in completions:
                text = choice.message.content
                code = parse_python_code(text)
                passed, _error = execute_code(code, problem["tests"])
                results.append((passed, text, code))

            # Check if we have ≥1 pass AND ≥1 fail
            passes = [(t, c) for p, t, c in results if p]
            fails = [(t, c) for p, t, c in results if not p]

            if passes and fails:
                # chosen = passing completion with shortest extracted code (by character count)
                # x[0] = full response text, x[1] = extracted code; sort by x[1] length
                chosen = min(passes, key=lambda x: len(x[1]))[0]
                rejected = fails[np.random.randint(len(fails))][0]

                # Format as conversational JSONL
                pair = {
                    "prompt": [{"role": "user", "content": problem["question"]}],
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                }

                f.write(json.dumps(pair) + "\n")

                # Update checkpoint
                checkpoint[problem_id] = {
                    "status": "complete",
                    "completed_at": time.time(),
                }

                # Save checkpoint every 100 problems
                if len(checkpoint) % 100 == 0:
                    with open(checkpoint_file, "w") as cf:
                        json.dump(checkpoint, cf)
                    print(f"Checkpoint saved: {len(checkpoint)} problems")

    # Final checkpoint save
    with open(checkpoint_file, "w") as cf:
        json.dump(checkpoint, cf)

    # Print stats
    # total_tried = full dataset size (checkpoint entries are a subset of ds, not additive)
    total_tried = len(ds)
    output_path = Path(args.output)
    if output_path.exists():
        raw_lines = [
            line for line in output_path.read_text().splitlines() if line.strip()
        ]
        pairs_kept = len(raw_lines)
    else:
        raw_lines = []
        pairs_kept = 0
    keep_rate = (pairs_kept / total_tried * 100) if total_tried > 0 else 0

    # Calculate approximate word counts (whitespace split — proxy for token length;
    # use these to inform max_length in train_dpo.py, not as exact token counts)
    if pairs_kept > 0:
        word_lengths = []
        for line in raw_lines:
            pair = json.loads(line)
            combined = pair["chosen"][0]["content"] + pair["rejected"][0]["content"]
            word_lengths.append(len(combined.split()))

        word_lengths.sort()
        n = len(word_lengths)

        def _pct(ratio: float) -> int:
            return word_lengths[max(0, min(n - 1, int(n * ratio)))] if n else 0

        p50 = _pct(0.5)
        p90 = _pct(0.9)
        p99 = _pct(0.99)
    else:
        p50 = p90 = p99 = 0

    print("\nStats:")
    print(f"  Total problems tried: {total_tried}")
    print(f"  Pairs kept: {pairs_kept}")
    print(f"  Keep rate: {keep_rate:.1f}%")
    print(f"  Word count p50/p90/p99 (approx token proxy): {p50}/{p90}/{p99}")


if __name__ == "__main__":
    main()
