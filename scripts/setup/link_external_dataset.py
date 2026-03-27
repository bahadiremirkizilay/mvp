"""
Link an external dataset directory into the workspace without copying files.

Windows strategy:
- Prefer NTFS directory junction (mklink /J) to avoid admin requirement.

Usage examples:
  python scripts/setup/link_external_dataset.py --external "D:/datasets/BagOfLies" --target "data/BagOfLies"
  python scripts/setup/link_external_dataset.py --external "E:/BagOfLies" --target "BagOfLies" --force
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def make_junction_windows(target: Path, external: Path) -> None:
    cmd = ["cmd", "/c", "mklink", "/J", str(target), str(external)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "mklink failed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Link external dataset folder into workspace")
    parser.add_argument("--external", required=True, help="External source directory path")
    parser.add_argument("--target", required=True, help="Target link path in workspace")
    parser.add_argument("--force", action="store_true", help="Replace target if it already exists")
    args = parser.parse_args()

    workspace = Path.cwd()
    external = Path(args.external).expanduser().resolve()
    target = Path(args.target)
    if not target.is_absolute():
        target = (workspace / target).resolve()

    if not external.exists() or not external.is_dir():
        raise FileNotFoundError(f"External directory not found: {external}")

    if target.exists() or target.is_symlink():
        if not args.force:
            raise FileExistsError(f"Target already exists: {target}. Use --force to replace.")
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        make_junction_windows(target, external)
    else:
        os.symlink(str(external), str(target), target_is_directory=True)

    print("=" * 80)
    print("EXTERNAL DATASET LINK CREATED")
    print("=" * 80)
    print(f"External source: {external}")
    print(f"Workspace link:  {target}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
