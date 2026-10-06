"""Visual block analysis: which double-stream blocks carry the reference style.

Uses only images that already exist after the sweep and ablation stages (no generation):
  T2I           val/alpha_<A>/origin
  blocks 1-4    val/ablation/blocks_0-3/steered
  blocks 5-8    val/ablation/blocks_4-7/steered
  blocks 1-8    val/alpha_<A>/steered

For every chosen validation prompt it writes one grid: rows = references,
columns = reference | T2I | blocks 1-4 | blocks 5-8 | blocks 1-8.

    python scripts/make_block_figures.py                     # prompts 9 and 16, alpha 6
    python scripts/make_block_figures.py --prompts 9 16 3 --alpha 6

Default prompts (prompts_collection/klein_style/val_prompts.txt, 0-based):
  9  = "a street cafe in the evening",  16 = "a girl with a dog"
Output: paper_results/figs/blocks_grid_p<idx>.jpg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

INDEX_RE = re.compile(r"^(\d+)_")
ORDER = ["hartley_movement", "kandinsky_improvisation30", "vangogh_self_portrait", "seurat_grande_jatte",
         "monet_water_lilies", "hokusai_great_wave", "degas_jockey"]


def by_index(directory: Path) -> dict:
    out = {}
    if directory.is_dir():
        for p in directory.iterdir():
            m = INDEX_RE.match(p.stem)
            if m and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                out[int(m.group(1))] = p
    return out


def alpha_dir(exp: Path, alpha: float) -> Path:
    for d in exp.glob("val/alpha_*"):
        try:
            if float(d.name.split("_", 1)[1]) == alpha:
                return d
        except ValueError:
            pass
    raise FileNotFoundError(f"no val/alpha_{alpha:g} in {exp}")


def tile(path, size):
    canvas = Image.new("RGB", (size, size), "white")
    if path is None:
        return canvas
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="experiments/klein_9b/paper")
    ap.add_argument("--out_dir", default="paper_results/figs")
    ap.add_argument("--prompts", type=int, nargs="+", default=[9, 16])
    ap.add_argument("--alpha", type=float, default=6.0)
    ap.add_argument("--cell", type=int, default=320)
    args = ap.parse_args()

    root = Path(args.exp_root)
    refs = [r for r in ORDER if (root / r).is_dir()] + sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in ORDER)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    C, g = args.cell, 8
    for idx in args.prompts:
        rows = []
        for stem in refs:
            exp = root / stem
            ad = alpha_dir(exp, args.alpha)
            ref = next(iter(sorted(exp.glob("reference.*"))), None)
            cells = [ref,
                     by_index(ad / "origin").get(idx),
                     by_index(exp / "val/ablation/blocks_0-3/steered").get(idx),
                     by_index(exp / "val/ablation/blocks_4-7/steered").get(idx),
                     by_index(ad / "steered").get(idx)]
            missing = [n for n, c in zip(["reference", "T2I", "1-4", "5-8", "1-8"], cells) if c is None]
            if missing:
                print(f"  {stem}: missing {missing} for prompt {idx}")
            rows.append(cells)
        W = 5 * C + 4 * g
        H = len(rows) * C + (len(rows) - 1) * g
        canvas = Image.new("RGB", (W, H), "white")
        for r, cells in enumerate(rows):
            for c, path in enumerate(cells):
                canvas.paste(tile(path, C), (c * (C + g), r * (C + g)))
        path = out / f"blocks_grid_p{idx}.jpg"
        canvas.save(path, quality=90)
        print(f"wrote {path}  (rows: {', '.join(refs)}; cols: reference | T2I | 1-4 | 5-8 | 1-8)")


if __name__ == "__main__":
    main()
