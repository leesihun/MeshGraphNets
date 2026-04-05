"""
Concatenate all ``{prefix}_s*_3D_iso.gif`` in the repo root into one timeline (same FPS).

  python scripts/append_prefix_gifs.py
  python scripts/append_prefix_gifs.py --prefix deforming_plate flag_simple sphere_simple
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

GIF_FPS = 20


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _sample_sort_key(path: Path) -> int:
    m = re.search(r"_s(\d+)_", path.name)
    return int(m.group(1)) if m else 0


def gif_frames(path: Path) -> list[Image.Image]:
    im = Image.open(path)
    frames: list[Image.Image] = []
    n = getattr(im, "n_frames", 1)
    for i in range(n):
        im.seek(i)
        frames.append(im.copy().convert("RGB"))
    im.close()
    return frames


def append_clips(paths: list[Path], out_path: Path, gif_fps: int) -> int:
    combined: list[Image.Image] = []
    for p in paths:
        combined.extend(gif_frames(p))
    if not combined:
        return 0
    dur_ms = int(1000 / gif_fps)
    combined[0].save(
        out_path,
        save_all=True,
        append_images=combined[1:],
        duration=dur_ms,
        loop=0,
    )
    return len(combined)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append same-prefix 3D_iso GIFs into one file.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory containing GIFs (default: repo root)",
    )
    parser.add_argument(
        "--prefix",
        nargs="+",
        default=["deforming_plate", "flag_simple", "sphere_simple"],
        metavar="NAME",
        help="Filename stem prefixes to combine",
    )
    parser.add_argument("--gif-fps", type=int, default=GIF_FPS, help="Output GIF FPS")
    parser.add_argument(
        "--suffix",
        default="_3D_iso.gif",
        help="Glob suffix after s<id> (default: _3D_iso.gif)",
    )
    args = parser.parse_args()

    root = args.root.resolve() if args.root else _repo_root()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    for prefix in args.prefix:
        pattern = f"{prefix}_s*{args.suffix}"
        paths = sorted(root.glob(pattern), key=_sample_sort_key)
        if not paths:
            print(f"No matches for {pattern!r} under {root}")
            continue
        out = root / f"{prefix}_combined{args.suffix}"
        n = append_clips(paths, out, args.gif_fps)
        print(f"{out.name}: {n} frames, {len(paths)} clips -> {[p.name for p in paths]}")


if __name__ == "__main__":
    main()
