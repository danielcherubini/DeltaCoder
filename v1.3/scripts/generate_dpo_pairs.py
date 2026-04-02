"""Generate DPO preference pairs by running DeltaCoder v1.3 against AceCode-V2-122K problems.

Downloads a configurable subset of AceCode-V2-122K from HuggingFace, shuffles and takes
n-problems rows, then calls DeltaCoder API to generate n-samples completions per problem.
Executes each completion against test cases, keeping problems where >=1 pass AND >=1 fail.
Formats kept pairs as conversational JSONL.

Uses async HTTP to send multiple problems concurrently, keeping vLLM fully saturated.
"""

import argparse
import asyncio
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

import aiohttp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def parse_python_code(text: str) -> str:
    """Extract Python code from response text.

    If response contains ```python ... ``` fences, extract the content of the FIRST code block.
    Otherwise strip leading/trailing whitespace and use as-is.
    Strip any remaining ``` markers.
    """
    pattern = re.compile(r"```python\s*\n(.*?)\n\s*```", re.DOTALL)
    match = pattern.search(text)

    if match:
        return match.group(1).strip()

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
            combined_code = code + "\n\n" + "\n".join(tests)
            tmpfile.write_text(combined_code)

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
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


async def call_api_async(
    session: aiohttp.ClientSession,
    problem: dict,
    n_samples: int,
    api_base: str,
    model: str,
) -> list[str]:
    """Call DeltaCoder API asynchronously for n_samples completions.

    Returns list of completion text strings.
    """
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": problem["question"]}],
        "temperature": 0.8,
        "max_tokens": 1024,
        "n": n_samples,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"  API error {resp.status}: {text[:200]}", file=sys.stderr)
                return []
            data = await resp.json()
            return [choice["message"]["content"] for choice in data.get("choices", [])]
    except Exception as e:
        print(f"  API call failed (skipping): {e}", file=sys.stderr)
        return []


async def process_problem(
    session: aiohttp.ClientSession,
    problem: dict,
    n_samples: int,
    api_base: str,
    model: str,
    rng: np.random.RandomState,
) -> Optional[dict]:
    """Process a single problem: generate completions, execute, return pair or None."""
    completions = await call_api_async(session, problem, n_samples, api_base, model)

    if not completions:
        return None

    results = []
    for text in completions:
        code = parse_python_code(text)
        passed, _error = execute_code(code, problem["tests"])
        results.append((passed, text, code))

    passes = [(t, c) for p, t, c in results if p]
    fails = [(t, c) for p, t, c in results if not p]

    if not passes or not fails:
        return None

    chosen = min(passes, key=lambda x: len(x[1]))[0]
    rejected = fails[rng.randint(len(fails))][0]

    return {
        "prompt": [{"role": "user", "content": problem["question"]}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


async def main_async(args):
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print("Loading AceCode-V2-122K from HuggingFace...")
    ds = load_dataset(
        "TIGER-Lab/AceCode-V2-122K", split="train", trust_remote_code=True
    )
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n_problems, len(ds))))
    print(f"Using {len(ds)} problems")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file = str(Path(args.output).with_suffix(".checkpoint"))

    checkpoint = {}
    if Path(checkpoint_file).exists():
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint: {len(checkpoint)} problems already processed")

    # Filter out already-processed problems
    problems = [
        (i, p)
        for i, p in enumerate(ds)
        if (p.get("id") or f"idx-{i}") not in checkpoint
    ]
    print(f"Problems remaining: {len(problems)}")

    pairs_written = 0
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    timeout = aiohttp.ClientTimeout(total=120)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with open(args.output, "a", encoding="utf-8") as f:
            pbar = tqdm(total=len(problems), desc="Processing problems")

            # Process in batches of --concurrency
            for batch_start in range(0, len(problems), args.concurrency):
                batch = problems[batch_start : batch_start + args.concurrency]

                tasks = [
                    process_problem(
                        session, problem, args.n_samples, args.api_base, args.model, rng
                    )
                    for _, problem in batch
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                for (i, problem), result in zip(batch, results):
                    problem_id = problem.get("id") or f"idx-{i}"

                    if isinstance(result, Exception):
                        print(
                            f"  Problem {problem_id} failed: {result}", file=sys.stderr
                        )
                    elif result is not None:
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                        pairs_written += 1
                        checkpoint[problem_id] = {
                            "status": "complete",
                            "completed_at": time.time(),
                        }

                    pbar.update(1)

                # Save checkpoint every batch
                if len(checkpoint) % 100 == 0:
                    with open(checkpoint_file, "w") as cf:
                        json.dump(checkpoint, cf)

            pbar.close()

    # Final checkpoint save
    with open(checkpoint_file, "w") as cf:
        json.dump(checkpoint, cf)

    total_tried = len(problems)
    keep_rate = (pairs_written / total_tried * 100) if total_tried > 0 else 0

    print("\nStats:")
    print(f"  Total problems tried: {total_tried}")
    print(f"  Pairs kept: {pairs_written}")
    print(f"  Keep rate: {keep_rate:.1f}%")


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
        "--output", type=str, default="data/dpo_pairs_v1.3.jsonl", help="Output file"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Number of concurrent requests"
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
