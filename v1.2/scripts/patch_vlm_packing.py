"""
Patch Unsloth's trainer.py to remove the VLM packing block.

This enables sample packing for VLMs doing text-only training.
Based on the patch from https://github.com/unslothai/unsloth/issues/4160

Run this BEFORE training:
    python patch_vlm_packing.py

It modifies unsloth/trainer.py in-place to:
1. Remove the is_vlm variable and all VLM detection logic
2. Remove is_vlm from the 'blocked' condition
3. Remove the VLM-specific reason message
"""

import importlib
import os
import re
import sys


def find_trainer_py():
    """Find unsloth/trainer.py in the Python path."""
    import unsloth

    unsloth_dir = os.path.dirname(unsloth.__file__)
    trainer_path = os.path.join(unsloth_dir, "trainer.py")
    if os.path.exists(trainer_path):
        return trainer_path
    raise FileNotFoundError(f"Could not find unsloth/trainer.py at {trainer_path}")


def patch_trainer(path):
    """Apply the VLM packing unblock patch."""
    with open(path, "r") as f:
        content = f.read()

    original = content

    # 1. Remove the is_vlm variable initialization
    content = content.replace("        is_vlm = False\n", "")

    # 2. Remove the VLM detection block (architectures check + vision_config check)
    # This block looks like:
    #                 # Check if VLM
    #                 architectures = getattr(model_config, "architectures", None)
    #                 if architectures is None:
    #                     architectures = []
    #                 is_vlm = any(
    #                     x.endswith("ForConditionalGeneration") for x in architectures
    #                 )
    #                 is_vlm = is_vlm or hasattr(model_config, "vision_config")
    vlm_check_pattern = re.compile(
        r"\s*# Check if VLM\n"
        r'\s*architectures = getattr\(model_config, "architectures", None\)\n'
        r"\s*if architectures is None:\n"
        r"\s*architectures = \[\]\n"
        r"\s*is_vlm = any\(\n"
        r'\s*x\.endswith\("ForConditionalGeneration"\) for x in architectures\n'
        r"\s*\)\n"
        r'\s*is_vlm = is_vlm or hasattr\(model_config, "vision_config"\)\n',
        re.MULTILINE,
    )
    content = vlm_check_pattern.sub("\n", content)

    # 3. Remove is_vlm from the blocked condition
    content = content.replace("            or is_vlm\n", "")

    # 4. Remove the VLM reason message
    vlm_reason_pattern = re.compile(
        r"\s*elif is_vlm:\n"
        r'\s*reason = "vision-language model"\n',
        re.MULTILINE,
    )
    content = vlm_reason_pattern.sub("", content)

    if content == original:
        print(
            "WARNING: No changes made — patch may have already been applied or trainer.py format changed"
        )
        return False

    # Write patched file
    with open(path, "w") as f:
        f.write(content)

    return True


def verify_patch(path):
    """Verify the patch was applied correctly."""
    with open(path, "r") as f:
        content = f.read()

    issues = []
    if "is_vlm" in content:
        # Count occurrences — there might be some in comments
        lines_with_vlm = [
            line.strip()
            for line in content.split("\n")
            if "is_vlm" in line and not line.strip().startswith("#")
        ]
        if lines_with_vlm:
            issues.append(
                f"Found {len(lines_with_vlm)} non-comment lines with 'is_vlm'"
            )
            for line in lines_with_vlm:
                issues.append(f"  - {line}")

    if "vision-language model" in content:
        issues.append("Found 'vision-language model' reason string still present")

    return issues


def main():
    path = find_trainer_py()
    print(f"Found unsloth/trainer.py at: {path}")

    # Back up
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        import shutil

        shutil.copy2(path, backup_path)
        print(f"Backed up to: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    # Patch
    if patch_trainer(path):
        print("Patch applied successfully!")
    else:
        print("Patch may have already been applied.")

    # Verify
    issues = verify_patch(path)
    if issues:
        print("\nVerification issues:")
        for issue in issues:
            print(f"  {issue}")
        print("\nYou may need to manually inspect the file.")
    else:
        print("Verification passed — no is_vlm references remain in code.")


if __name__ == "__main__":
    main()
