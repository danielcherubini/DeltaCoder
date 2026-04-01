"""Delete v1.2 merged model files from HuggingFace model repo."""

import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
REPO = "danielcherubini/Qwen3.5-DeltaCoder-9B"

api = HfApi(token=TOKEN)
files = api.list_repo_files(REPO)

print(f"Files in {REPO}:")
for f in files:
    print(f"  {f}")

# Delete all files except .gitattributes and README.md
to_delete = [f for f in files if f not in (".gitattributes", "README.md")]
print(f"\nDeleting {len(to_delete)} files:")
for f in to_delete:
    print(f"  Deleting {f}...")
    api.delete_file(f, repo_id=REPO, repo_type="model")
    print(f"  OK")

print("Done")
