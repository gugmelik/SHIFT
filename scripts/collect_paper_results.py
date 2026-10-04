"""Collect finished Klein experiments into paper_results/ (input for the next paper revision).

Reads   experiments/klein_9b/paper/<reference>/...   (written by scripts/paper_klein_experiments.sh)
Writes  paper_results/
          summary.json     every number the paper tables need, with 95% bootstrap CIs
          tables.md        the same numbers laid out like paper Tables 1, 2, 4, 5, 6 + alpha sweep
          figs/            qualitative grids (test set) and alpha-sweep grids (val set) per reference
          sources/         SOURCES.md + references.bib of the reference images
          MISSING.md       what was expected but not found (empty if everything ran)

Run from the repository root after the `score` stage:

    python scripts/collect_paper_results.py
    python scripts/collect_paper_results.py --fig_indices 0 5 17 33 61 90 --alpha 6

Only numpy and Pillow are needed (no GPU).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

INDEX_RE = re.compile(r"^(\d+)_")
METHOD_ORDER = ["t2i", "shift_text", "teacher", "ours"]
METHOD_LABEL = {"t2i": "T2I", "shift_text": "SHIFT (text)", "teacher": "Teacher (I2I)", "ours": "Ours"}
MAIN_METRICS = ["csd_to_reference", "gram_to_reference", "dinov2_to_reference", "dinov3_to_reference",
                "dinov2_to_origin", "lpips_to_origin", "dists_to_origin", "clip_text"]
BLOCK_GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7)]


def bootstrap(values: List[float], n_boot: int = 10000, seed: int = 0) -> Optional[Dict]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(np.percentile(means, 2.5)),
                                                float(np.percentile(means, 97.5))], "n": int(arr.size)}


def load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def pooled_metrics(score_files: List[Path]) -> Dict[str, Dict]:
    """Pool per-image values over seeds (each seed = independent sample per prompt)."""
    pooled: Dict[str, List[float]] = defaultdict(list)
    set_level: Dict[str, List[float]] = defaultdict(list)
    for path in score_files:
        data = load(path)
        if not data:
            continue
        for metric, vals in data.get("per_image", {}).items():
            pooled[metric].extend(vals.values())
        for key in ("fid", "kid_mean"):
            if key in data.get("set_level", {}):
                set_level[key].append(data["set_level"][key])
    out = {m: bootstrap(v) for m, v in pooled.items()}
    for key, vals in set_level.items():
        out[key] = {"mean": float(np.mean(vals)), "ci95": None, "n": len(vals)}
    return out


def fmt(stat: Optional[Dict], digits: int = 3) -> str:
    if not stat:
        return "—"
    if stat.get("ci95"):
        lo, hi = stat["ci95"]
        return f"{stat['mean']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return f"{stat['mean']:.{digits}f}"


def map_by_index(directory: Path) -> Dict[int, Path]:
    out = {}
    if directory.is_dir():
        for p in sorted(directory.iterdir()):
            m = INDEX_RE.match(p.stem)
            if m and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                out[int(m.group(1))] = p
    return out


def grid(rows: List[List[Optional[Path]]], header: List[str], cell: int = 320, pad: int = 6) -> Image.Image:
    head = 22
    w = len(header) * cell + (len(header) - 1) * pad
    h = head + len(rows) * cell + (len(rows) - 1) * pad
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    for c, label in enumerate(header):
        draw.text((c * (cell + pad) + 4, 4), label, fill="black")
    for r, row in enumerate(rows):
        for c, path in enumerate(row):
            if path is None:
                continue
            img = Image.open(path).convert("RGB")
            img.thumbnail((cell, cell), Image.LANCZOS)
            canvas.paste(img, (c * (cell + pad), head + r * (cell + pad)))
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="experiments/klein_9b/paper")
    ap.add_argument("--ref_dir", default="data/reference_images/open")
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--alpha", type=float, default=None, help="Alpha used for the test runs (for captions)")
    ap.add_argument("--fig_indices", type=int, nargs="*", default=[0, 10, 20, 40, 60, 80],
                    help="Test-prompt indices shown in the qualitative figure")
    ap.add_argument("--sweep_indices", type=int, nargs="*", default=[0, 6, 12, 18])
    args = ap.parse_args()

    exp_root, out = Path(args.exp_root), Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)
    (out / "sources").mkdir(parents=True, exist_ok=True)
    missing: List[str] = []
    sources = {r["slug"]: r for r in (load(Path(args.ref_dir) / "sources.json") or [])}
    for name in ("SOURCES.md", "references.bib", "sources.json"):
        src = Path(args.ref_dir) / name
        if src.is_file():
            shutil.copy(src, out / "sources" / name)
        else:
            missing.append(f"{src} (run scripts/download_open_references.py)")

    refs = sorted(p for p in exp_root.iterdir() if p.is_dir()) if exp_root.is_dir() else []
    if not refs:
        missing.append(f"{exp_root}/<reference>/ (run scripts/paper_klein_experiments.sh)")

    summary = {"references": {}, "alpha_test": args.alpha}
    for ref in refs:
        stem, S = ref.name, ref / "scores"
        entry: Dict = {"source": sources.get(stem)}

        # --- Table 5: method comparison on the test set (pooled over seeds)
        methods = {}
        for m in METHOD_ORDER:
            files = [S / "teacher.json"] if m == "teacher" else sorted(S.glob(f"{m}_seed_*.json"))
            files = [f for f in files if f.is_file()]
            if files:
                methods[m] = pooled_metrics(files)
            elif m != "shift_text":
                missing.append(f"{S}/{m}{'' if m == 'teacher' else '_seed_*'}.json")
        entry["test"] = methods
        entry["wilcoxon"] = load(S / "compare_seed_42.json")

        # --- alpha sweep and ablations (validation set)
        entry["val_sweep"] = {}
        for f in sorted(S.glob("val_alpha_*.json")):
            alpha = f.stem.replace("val_alpha_", "")
            entry["val_sweep"][alpha] = pooled_metrics([f])
        entry["val_ablation"] = {f.stem.replace("val_", ""): pooled_metrics([f])
                                 for f in sorted(S.glob("val_*.json")) if not f.stem.startswith("val_alpha_")}
        if not entry["val_sweep"]:
            missing.append(f"{S}/val_alpha_*.json")

        # --- Table 1: token norms (from apply stats; block output before steering at that block)
        norms = defaultdict(list)
        for stats_file in ref.glob("val/alpha_*/stats.json"):
            for key, value in (load(stats_file) or {}).get("mean_token_norm", {}).items():
                _step, layer, stream = key.split("|")
                norms[(int(layer), stream)].append(value)
        entry["token_norms"] = {
            f"blocks_{a + 1}-{b + 1}": {
                stream: float(np.mean(norms[(a, stream)] + norms[(b, stream)]))
                if norms[(a, stream)] or norms[(b, stream)] else None
                for stream in ("img", "txt")}
            for a, b in BLOCK_GROUPS}
        if not norms:
            missing.append(f"{ref}/val/alpha_*/stats.json (mean_token_norm)")

        # --- Table 2: timing
        origin_t, steered_t = [], []
        for stats_file in ref.glob("test/ours/seed_*/stats.json"):
            st = load(stats_file) or {}
            origin_t += st.get("origin_seconds", [])
            steered_t += st.get("steered_seconds", [])
        i2i_t = (load(ref / "test/teacher/i2i_timing.json") or {}).get("seconds", [])
        extract = (load(ref / "ref/extract_timing.json") or {}).get("seconds")

        def mean_std(xs):
            return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                    "n": len(xs)} if xs else None
        entry["timing"] = {"t2i": mean_std(origin_t), "steered": mean_std(steered_t),
                           "i2i": mean_std(i2i_t), "extraction_total": extract}

        # --- figures
        seed_dirs = sorted(ref.glob("test/ours/seed_*"))
        if seed_dirs:
            sd = seed_dirs[0]
            origin, steered = map_by_index(sd / "origin"), map_by_index(sd / "steered")
            teacher = map_by_index(ref / "test/teacher/i2i")
            rows = [[origin.get(i), steered.get(i), teacher.get(i)] for i in args.fig_indices if i in origin]
            if rows:
                grid(rows, ["T2I", f"ours (alpha={args.alpha})", "teacher I2I"]).save(
                    out / "figs" / f"{stem}_test_grid.jpg", quality=92)
        alpha_dirs = sorted(ref.glob("val/alpha_*"), key=lambda p: float(p.name.split("_")[1]))
        if alpha_dirs:
            origin = map_by_index(alpha_dirs[0] / "origin")
            cols = [map_by_index(d / "steered") for d in alpha_dirs]
            rows = [[origin.get(i)] + [c.get(i) for c in cols] for i in args.sweep_indices if i in origin]
            if rows:
                grid(rows, ["T2I"] + [d.name.replace("alpha_", "a=") for d in alpha_dirs]).save(
                    out / "figs" / f"{stem}_val_alpha_sweep.jpg", quality=92)
        ref_img = next(iter(sorted(ref.glob("reference.*"))), None)
        if ref_img:
            shutil.copy(ref_img, out / "figs" / f"{stem}_reference{ref_img.suffix}")

        summary["references"][stem] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- human-readable tables
    md = ["# Paper results (generated by scripts/collect_paper_results.py)", ""]
    for stem, e in summary["references"].items():
        src = e.get("source") or {}
        md += [f"## {stem}", f"{src.get('artist', '?')}, *{src.get('title', '?')}*, {src.get('date', '?')} "
               f"— {src.get('style', '')}", ""]
        md += ["**Table 5 (test, pooled over seeds; mean [95% CI])**", "",
               "| Method | " + " | ".join(MAIN_METRICS) + " | FID | KID |",
               "|---" * (len(MAIN_METRICS) + 3) + "|"]
        for m, stats in e["test"].items():
            md.append(f"| {METHOD_LABEL[m]} | " + " | ".join(fmt(stats.get(k)) for k in MAIN_METRICS)
                      + f" | {fmt(stats.get('fid'), 2)} | {fmt(stats.get('kid_mean'), 4)} |")
        md += ["", "**Alpha sweep (val)**", "", "| alpha | csd | gram | dinov2_to_origin | clip_text |",
               "|---|---|---|---|---|"]
        for a, st in sorted(e["val_sweep"].items(), key=lambda kv: float(kv[0])):
            md.append(f"| {a} | {fmt(st.get('csd_to_reference'))} | {fmt(st.get('gram_to_reference'), 5)} | "
                      f"{fmt(st.get('dinov2_to_origin'))} | {fmt(st.get('clip_text'))} |")
        md += ["", "**Table 6: ablations (val)**", "", "| config | csd | dinov2_to_origin | clip_text |",
               "|---|---|---|---|"]
        for name, st in e["val_ablation"].items():
            md.append(f"| {name} | {fmt(st.get('csd_to_reference'))} | {fmt(st.get('dinov2_to_origin'))} | "
                      f"{fmt(st.get('clip_text'))} |")
        md += ["", "**Table 1: mean token norm**", "", "| blocks | img | txt |", "|---|---|---|"]
        for g, v in e["token_norms"].items():
            md.append(f"| {g} | {v['img'] if v['img'] is None else round(v['img'], 2)} | "
                      f"{v['txt'] if v['txt'] is None else round(v['txt'], 2)} |")
        t = e["timing"]
        md += ["", "**Table 2: timing (s/image)**", ""]
        for k in ("t2i", "steered", "i2i"):
            md.append(f"- {k}: " + (f"{t[k]['mean']:.3f} ± {t[k]['std']:.3f} (n={t[k]['n']})" if t[k] else "—"))
        md.append(f"- extraction total: {t['extraction_total'] if t['extraction_total'] else '—'} s")
        if e.get("wilcoxon"):
            md += ["", "**Wilcoxon (seed 42, Holm), ours vs others**", "",
                   "| other | metric | mean diff | p_holm |", "|---|---|---|---|"]
            for row in e["wilcoxon"]:
                md.append(f"| {row['other']} | {row['metric']} | {row['mean_diff']:.4f} | {row['p_holm']:.4g}"
                          f"{' *' if row.get('significant_0.05') else ''} |")
        md.append("")
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "MISSING.md").write_text(
        "# Missing inputs\n\n" + ("\n".join(f"- {m}" for m in missing) if missing else "Nothing missing.\n"),
        encoding="utf-8")
    print(f"wrote {out}/summary.json, tables.md, figs/, sources/; {len(missing)} missing item(s)")


if __name__ == "__main__":
    main()
