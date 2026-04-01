"""Delete v1.2 GGUFs from HuggingFace GGUF repo."""

import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "danielcherubini/Qwen3.5-DeltaCoder-9B-GGUF"

api = HfApi(token=TOKEN)
files = api.list_repo_files(REPO)
v12 = [f for f in files if "v1.2" in f]

print(f"Found {len(v12)} v1.2 files to delete:")
for f in v12:
    print(f"  Deleting {f}...")
    api.delete_file(f, repo_id=REPO, repo_type="model")
    print(f"  OK")

print("Done")
