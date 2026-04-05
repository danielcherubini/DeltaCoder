"""
Patch causal-conv1d setup.py to only compile for SM 9.0 (H100/Hopper).

The setup.py hardcodes all GPU architectures and ignores TORCH_CUDA_ARCH_LIST.
This script replaces the hardcoded arch list with just sm_90.

Usage:
    cd /workspace/causal-conv1d
    python /workspace/patch_causal_conv1d.py
"""

import sys
from pathlib import Path


def main():
    setup_py = Path("setup.py")
    if not setup_py.exists():
        print("ERROR: setup.py not found. Run this from the causal-conv1d directory.")
        sys.exit(1)

    content = setup_py.read_text()

    # The hardcoded arch flags we want to replace
    old_block = """        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")"""

    new_block = """        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")"""

    if old_block not in content:
        print("WARNING: Expected code block not found. setup.py may have changed.")
        print("Trying to check if already patched...")
        if "arch=compute_90,code=sm_90" in content and "arch=compute_75" not in content:
            print("Already patched for sm_90 only.")
            return
        print("ERROR: Cannot patch. Manual intervention needed.")
        sys.exit(1)

    # Replace the base arch flags
    content = content.replace(old_block, new_block)

    # Also remove the conditional arch flags for newer CUDA versions
    # These add sm_90, sm_100, sm_120, sm_103, sm_110, sm_121
    lines = content.split("\n")
    new_lines = []
    skip_until_unindent = False
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip "if bare_metal_version >= Version(...)" blocks that add more archs
        if (
            "bare_metal_version >= Version" in line
            and "cc_flag" in lines[min(i + 1, len(lines) - 1)]
        ):
            # Skip this if block and all its cc_flag lines
            skip_until_unindent = True
            i += 1
            continue
        if skip_until_unindent:
            stripped = line.strip()
            if stripped.startswith("cc_flag.append") or stripped == "":
                i += 1
                continue
            else:
                skip_until_unindent = False
        new_lines.append(line)
        i += 1

    content = "\n".join(new_lines)

    setup_py.write_text(content)
    print("PATCHED: setup.py now only compiles for sm_90 (H100)")

    # Verify
    verify = setup_py.read_text()
    archs = [line for line in verify.split("\n") if "arch=compute" in line]
    print(f"Architectures in patched setup.py: {len(archs)}")
    for a in archs:
        print(f"  {a.strip()}")


if __name__ == "__main__":
    main()
