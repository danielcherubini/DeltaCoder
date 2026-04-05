"""Restore v1.1 DPO adapter to HuggingFace model repo."""

import os
from pathlib import Path
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "danielcherubini/Qwen3.5-DeltaCoder-9B"
ADAPTER_DIR = Path("qwen3.5/v1.0/outputs/deltacoder-9b-dpo/lora_adapter")

api = HfApi(token=TOKEN)

# Upload all files from v1.1 adapter
for path in sorted(ADAPTER_DIR.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(ADAPTER_DIR)
    size_mb = path.stat().st_size / 1e6
    print(f"Uploading {rel} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=str(rel),
        repo_id=REPO,
        repo_type="model",
    )
    print(f"  OK")

print("Done - v1.1 adapter restored")
