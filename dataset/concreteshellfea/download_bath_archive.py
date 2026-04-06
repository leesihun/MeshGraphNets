#!/usr/bin/env python3
"""
Download ConcreteShellFEA shards from University of Bath Research Data Archive
(DOI 10.15125/BATH-01519). The collection is split into 28 ZIP parts (~100+ GB total).

Example:
  python download_bath_archive.py --out-dir ./ConcreteShellFEA_raw --parts 1-28
  python download_bath_archive.py --out-dir ./ConcreteShellFEA_raw --parts 12
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path


def bath_zip_url(part_index: int) -> str:
    if not 1 <= part_index <= 28:
        raise ValueError(f"part_index must be 1..28, got {part_index}")
    doc_id = part_index + 3
    return f"https://researchdata.bath.ac.uk/1519/{doc_id}/datasets-{part_index:02d}.zip"


def parse_parts(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec and not spec.isdigit():
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def download_file(url: str, dest: Path, chunk: int = 8 * 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "MeshGraphNets-converter/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = resp.headers.get("Content-Length")
        total_n = int(total) if total else None
        done = 0
        with open(tmp, "wb") as f:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total_n and done % (64 * chunk) < chunk:
                    pct = 100.0 * done / total_n
                    print(f"\r  {dest.name}  {done / 1e9:.2f} / {total_n / 1e9:.2f} GB  ({pct:.1f}%)", end="", flush=True)
    print()
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, required=True, help="Directory to store datasets-XX.zip files.")
    p.add_argument(
        "--parts",
        type=str,
        default="1-28",
        help="Part numbers 1..28: e.g. '12' or '1-5' or '1,3,5-7' (default: all).",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip if the zip file already exists.")
    args = p.parse_args()

    parts = parse_parts(args.parts)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for part in parts:
        url = bath_zip_url(part)
        name = f"datasets-{part:02d}.zip"
        dest = out_dir / name
        if args.skip_existing and dest.exists():
            print(f"skip existing {dest}")
            continue
        print(f"downloading {url}\n  -> {dest.resolve()}")
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"ERROR part {part}: {e}", file=sys.stderr)
            sys.exit(1)

    print("Done. Extract all ZIPs into a single folder `datasets` per the archive README, e.g.")
    print("  ConcreteShellFEA/datasets/PerfectShell_LinearFEA/graphs/...")


if __name__ == "__main__":
    main()
