"""Upload merged model and GGUFs to HuggingFace Hub."""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_TOKEN")
GGUF_DIR = Path("D:/AI/DeltaCoder/v1.3/gguf")
MERGED_DIR = Path("D:/AI/DeltaCoder/v1.3/merged_fixed")
# TODO: Create these repos on HuggingFace when v1.3 is ready
GGUF_REPO = "danielcherubini/Qwen3.6-DeltaCoder-9B-GGUF"
MODEL_REPO = "danielcherubini/Qwen3.6-DeltaCoder-9B"

if not HF_TOKEN:
    print("ERROR: Pass HF token as first argument or set HF_TOKEN env var")
    sys.exit(1)

api = HfApi(token=HF_TOKEN)


def ensure_repo(repo_id):
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Repo {repo_id} exists")
    except Exception:
        print(f"Creating repo {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)


# --- Upload merged model ---
print("=" * 60)
print(f"UPLOADING MERGED MODEL -> {MODEL_REPO}")
print("=" * 60)
ensure_repo(MODEL_REPO)

merged_files = list(MERGED_DIR.iterdir())
print(f"Found {len(merged_files)} files in merged model dir")

for i, path in enumerate(sorted(merged_files), 1):
    if not path.is_file():
        continue
    size_gb = path.stat().st_size / 1e9
    print(f"\n[{i}/{len(merged_files)}] Uploading {path.name} ({size_gb:.2f} GB)...")
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=MODEL_REPO,
            repo_type="model",
        )
        print(f"  OK: {path.name}")
    except Exception as e:
        print(f"  ERROR: {path.name}: {e}")

print(f"\nMerged model upload complete!")

# --- Upload GGUFs ---
print()
print("=" * 60)
print(f"UPLOADING GGUFs -> {GGUF_REPO}")
print("=" * 60)
ensure_repo(GGUF_REPO)

ggufs = sorted(GGUF_DIR.glob("*.gguf"))
print(f"Found {len(ggufs)} GGUFs to upload")

for i, path in enumerate(ggufs, 1):
    size_gb = path.stat().st_size / 1e9
    print(f"\n[{i}/{len(ggufs)}] Uploading {path.name} ({size_gb:.2f} GB)...")
    try:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=GGUF_REPO,
            repo_type="model",
        )
        print(f"  OK: {path.name}")
    except Exception as e:
        print(f"  ERROR: {path.name}: {e}")

print("\n" + "=" * 60)
print("ALL UPLOADS COMPLETE!")
print("=" * 60)
