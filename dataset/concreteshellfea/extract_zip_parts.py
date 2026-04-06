#!/usr/bin/env python3
"""
Extract all `datasets-*.zip` shards from Bath BATH-01519 into one tree (e.g. PerfectShell_LinearFEA/...).

Example:
  python extract_zip_parts.py --zip-dir ./ConcreteShellFEA_raw --extract-to ./ConcreteShellFEA/datasets
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip-dir", type=Path, required=True, help="Folder containing datasets-01.zip ... datasets-28.zip")
    p.add_argument("--extract-to", type=Path, required=True, help="Destination root (creates PerfectShell_LinearFEA/... inside).")
    args = p.parse_args()

    zdir: Path = args.zip_dir
    dest: Path = args.extract_to
    dest.mkdir(parents=True, exist_ok=True)

    zips = sorted(zdir.glob("datasets-*.zip"))
    if not zips:
        raise SystemExit(f"No datasets-*.zip in {zdir}")

    for zp in zips:
        print(f"extracting {zp.name} ...")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(dest)
    print(f"Done. Graphs expected under e.g. {dest / 'PerfectShell_LinearFEA' / 'graphs'}")


if __name__ == "__main__":
    main()
