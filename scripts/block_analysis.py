"""Per-block analysis of the eight double-stream blocks (CPU; reads score files of the `blocks` stage).

Configurations on the validation set at the test alpha (25 prompts, seed 42):
  only_k   steering in block k only            -> contribution of block k on its own
  drop_k   steering in all blocks except k     -> necessity of block k given the others
  first_k  steering in blocks 1..k             -> accumulation of the effect with depth
  full     all eight blocks (val_alpha_<A>);  t2i = no steering (val_t2i)

Per-prompt effect e of a configuration (larger = stronger effect), per metric:
  shift towards the reference:  DINOv2 / DINOv3 / CSD:  x_cfg - x_t2i;   Gram: ln(Gram_t2i / Gram_cfg)
  change of the image:          LPIPS, DISTS, 1 - DINOv2-to-origin  (relative to the same-seed T2I)
  prompt alignment:             CLIP: x_cfg - x_t2i
Reported per reference and averaged over references:
  frac_only_k = E(only_k) / E(full),   loss_drop_k = (E(full) - E(drop_k)) / E(full),
  additivity  = sum_k E(only_k) / E(full),  curve E(first_k) / E(full).
Tests: only_k vs T2I for the reference metrics (Wilcoxon of e against 0) and drop_k vs full (paired),
Holm over the 8 blocks per metric; per reference and pooled (per-prompt mean over references).
Link to geometry: per-block mean token norm ||h|| (stats.json of the alpha sweep), rotation angle
arctan(alpha/||h||), steering-vector norm ||v|| (vectors/*_diff.pt, if torch is available), and the
Spearman correlation over blocks of frac_only with the angle.

    python scripts/block_analysis.py --alpha 6
Writes paper_results/blocks.json, blocks.md, figs/per_block.{pdf,png}, figs/per_block_grid_p<i>.jpg
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

REF_METRICS = ["dinov3_to_reference", "dinov2_to_reference", "csd_to_reference", "gram_to_reference"]
CHANGE_METRICS = ["lpips_to_origin", "dists_to_origin", "dinov2_to_origin"]
ALL = REF_METRICS + CHANGE_METRICS + ["clip_text"]
LABEL = {"dinov3_to_reference": "DINOv3 к реф.", "dinov2_to_reference": "DINOv2 к реф.", "csd_to_reference": "CSD к реф.",
         "gram_to_reference": "ln Грам T2I/упр.", "lpips_to_origin": "LPIPS к T2I", "dists_to_origin": "DISTS к T2I",
         "dinov2_to_origin": "1 − DINOv2 к T2I", "clip_text": "Δ CLIP"}
INDEX_RE = re.compile(r"^(\d+)_")
BLOCKS = range(1, 9)


def per_image(path: Path):
    if not path.is_file():
        return None
    return {m: {int(k): v for k, v in vals.items()}
            for m, vals in json.loads(path.read_text(encoding="utf-8"))["per_image"].items()}


def effect(cfg, t2i, metric):
    """Per-prompt effect dict, or None."""
    if cfg is None or metric not in cfg:
        return None
    x = cfg[metric]
    if metric in CHANGE_METRICS:
        return {k: (1 - v) if metric == "dinov2_to_origin" else v for k, v in x.items()}
    if t2i is None or metric not in t2i:
        return None
    b = t2i[metric]
    common = set(x) & set(b)
    if metric == "gram_to_reference":
        return {k: math.log(b[k] / x[k]) for k in common if x[k] > 0 and b[k] > 0}
    return {k: x[k] - b[k] for k in common}


def mean(d):
    return float(np.mean(list(d.values()))) if d else None


def holm(p):
    order, m, run, adj = np.argsort(p), len(p), 0.0, [0.0] * len(p)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def w_one(e):
    v = np.array(list(e.values()))
    return float(wilcoxon(v).pvalue) if len(v) >= 5 and not np.allclose(v, 0) else 1.0


def w_pair(a, b):
    k = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in k]), np.array([b[i] for i in k])
    return float(wilcoxon(x, y).pvalue) if len(k) >= 5 and not np.allclose(x, y) else 1.0


def token_norms(ref: Path):
    acc = defaultdict(list)
    for f in ref.glob("val/alpha_*/stats.json"):
        for key, v in json.loads(f.read_text()).get("mean_token_norm", {}).items():
            _st, layer, stream = key.split("|")
            if stream == "img":
                acc[int(layer) + 1].append(v)
    return {b: float(np.mean(v)) for b, v in acc.items()}


def vector_norms(ref: Path):
    try:
        import torch
    except ImportError:
        return {}
    files = [p for p in (ref / "ref" / "vectors").glob("*_diff.pt") if "text" not in p.name]
    if not files:
        return {}
    try:
        vec = torch.load(files[0], map_location="cpu", mmap=True, weights_only=False)
    except Exception:
        vec = torch.load(files[0], map_location="cpu", weights_only=False)
    out = defaultdict(list)
    for step in vec:
        if isinstance(step, int):
            for layer, val in vec[step].items():
                t = val["img"] if isinstance(val, dict) else val
                while t.dim() > 2:
                    t = t.squeeze(0)
                out[int(layer.split("_")[1]) + 1].append(float(t.float().norm(dim=-1).mean()))
    return {b: float(np.mean(v)) for b, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="experiments/klein_9b/paper")
    ap.add_argument("--alpha", default="6")
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--grid_prompts", type=int, nargs="*", default=[3, 15])
    a = ap.parse_args()
    out = Path(a.out); (out / "figs").mkdir(parents=True, exist_ok=True)
    alpha = float(a.alpha)
    refs = sorted(p for p in Path(a.exp_root).iterdir() if (p / "scores").is_dir())
    rep = {"per_ref": {}, "pooled": {}, "geometry": {}}
    pooled_e = defaultdict(lambda: defaultdict(list))  # (cfg) -> metric -> [per-ref effect dicts]

    for ref in refs:
        S = ref / "scores"
        t2i = per_image(S / "val_t2i.json")
        cfgs = {"full": per_image(S / f"val_alpha_{a.alpha}.json")}
        for k in BLOCKS:
            for kind in ("only", "drop", "first"):
                cfgs[f"{kind}_{k}"] = per_image(S / f"val_blk_{kind}_{k}.json")
        cfgs["first_1"] = cfgs["first_1"] or cfgs["only_1"]
        cfgs["first_4"] = cfgs["first_4"] or per_image(S / "val_blocks_0-3.json")
        cfgs["first_8"] = cfgs["full"]
        cfgs["drop_none"] = cfgs["full"]
        if cfgs["full"] is None or t2i is None or not any(cfgs.get(f"only_{k}") for k in BLOCKS):
            print(f"skip {ref.name}: run STAGES='blocks score' first (needs val_t2i, val_alpha_{a.alpha}, val_blk_*)")
            continue
        r = {"effects": {}, "frac_only": {}, "loss_drop": {}, "curve": {}, "additivity": {}, "tests": []}
        E = {c: {m: effect(v, t2i, m) for m in ALL} for c, v in cfgs.items() if v is not None}
        for c in E:
            r["effects"][c] = {m: mean(e) for m, e in E[c].items() if e}
            for m, e in E[c].items():
                if e:
                    pooled_e[c][m].append(e)
        for m in ALL:
            full = r["effects"]["full"].get(m)
            if not full:
                continue
            r["frac_only"][m] = {k: r["effects"][f"only_{k}"].get(m) / full for k in BLOCKS
                                 if f"only_{k}" in r["effects"] and r["effects"][f"only_{k}"].get(m) is not None}
            r["loss_drop"][m] = {k: (full - r["effects"][f"drop_{k}"][m]) / full for k in BLOCKS
                                 if f"drop_{k}" in r["effects"] and r["effects"][f"drop_{k}"].get(m) is not None}
            r["curve"][m] = {k: r["effects"][f"first_{k}"][m] / full for k in BLOCKS
                             if f"first_{k}" in r["effects"] and r["effects"][f"first_{k}"].get(m) is not None}
            if len(r["frac_only"][m]) == 8:
                r["additivity"][m] = float(sum(r["frac_only"][m].values()))
        # tests
        for m in ALL:
            for kind in ("only", "drop"):
                rows = []
                for k in BLOCKS:
                    c = f"{kind}_{k}"
                    if c not in E or not E[c].get(m):
                        continue
                    if kind == "only":
                        if m not in REF_METRICS + ["clip_text"]:
                            continue
                        p = w_one(E[c][m])
                    else:
                        p = w_pair(E[c][m], E["full"][m])
                    rows.append({"cfg": c, "metric": m, "p": p})
                for row, ph in zip(rows, holm([x["p"] for x in rows]) if rows else []):
                    row["p_holm"], row["sig"] = ph, ph < 0.05
                r["tests"] += rows
        r["token_norm"] = token_norms(ref)
        r["vector_norm"] = vector_norms(ref)
        rep["per_ref"][ref.name] = r

    names = list(rep["per_ref"])
    if not names:
        raise SystemExit("no reference has per-block results")

    # pooled over references: per-prompt mean of the effect over references
    def ref_avg(lst):
        common = set.intersection(*[set(x) for x in lst])
        return {k: float(np.mean([x[k] for x in lst])) for k in common}
    P = {c: {m: ref_avg(v) for m, v in mm.items() if len(v) == len(names)} for c, mm in pooled_e.items()}
    pooled = {"effects": {c: {m: mean(e) for m, e in mm.items()} for c, mm in P.items()}, "tests": [],
              "frac_only": {}, "loss_drop": {}, "curve": {}, "additivity": {}, "refs_sig": {}}
    for m in ALL:
        full = pooled["effects"].get("full", {}).get(m)
        if not full:
            continue
        for key, kind, f in (("frac_only", "only", lambda e: e / full), ("loss_drop", "drop", lambda e: (full - e) / full),
                             ("curve", "first", lambda e: e / full)):
            pooled[key][m] = {k: f(pooled["effects"][f"{kind}_{k}"][m]) for k in BLOCKS
                              if pooled["effects"].get(f"{kind}_{k}", {}).get(m) is not None}
        if len(pooled["frac_only"][m]) == 8:
            pooled["additivity"][m] = float(sum(pooled["frac_only"][m].values()))
        for kind in ("only", "drop"):
            rows = []
            for k in BLOCKS:
                c = f"{kind}_{k}"
                if c not in P or m not in P[c] or (kind == "only" and m in CHANGE_METRICS):
                    continue
                p = w_one(P[c][m]) if kind == "only" else w_pair(P[c][m], P["full"][m])
                nsig = sum(1 for n in names for t in rep["per_ref"][n]["tests"]
                           if t["cfg"] == c and t["metric"] == m and t["sig"])
                rows.append({"cfg": c, "metric": m, "p": p, "refs_sig": nsig})
            for row, ph in zip(rows, holm([x["p"] for x in rows]) if rows else []):
                row["p_holm"], row["sig"] = ph, ph < 0.05
            pooled["tests"] += rows
    rep["pooled"] = pooled

    # geometry: norms, angle, correlation with the single-block effect
    hn = {k: float(np.mean([rep["per_ref"][n]["token_norm"][k] for n in names if k in rep["per_ref"][n]["token_norm"]]))
          for k in BLOCKS if any(k in rep["per_ref"][n]["token_norm"] for n in names)}
    vn = {k: float(np.mean([rep["per_ref"][n]["vector_norm"][k] for n in names if k in rep["per_ref"][n]["vector_norm"]]))
          for k in BLOCKS if any(k in rep["per_ref"][n]["vector_norm"] for n in names)}
    geo = {"token_norm": hn, "angle_deg": {k: math.degrees(math.atan(alpha / v)) for k, v in hn.items()},
           "vector_norm": vn, "vector_to_token_norm": {k: vn[k] / hn[k] for k in vn if k in hn}, "spearman": {}}
    for m, fr in pooled["frac_only"].items():
        ks = [k for k in BLOCKS if k in fr and k in geo["angle_deg"]]
        if len(ks) >= 5:
            rho, p = spearmanr([geo["angle_deg"][k] for k in ks], [fr[k] for k in ks])
            geo["spearman"][m] = {"rho_angle": float(rho), "p": float(p)}
            if all(k in geo["vector_to_token_norm"] for k in ks):
                rho2, p2 = spearmanr([geo["vector_to_token_norm"][k] for k in ks], [fr[k] for k in ks])
                geo["spearman"][m].update(rho_v_over_h=float(rho2), p_v_over_h=float(p2))
    rep["geometry"] = geo
    (out / "blocks.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- markdown
    hdr = "| metric | " + " | ".join(f"b{k}" for k in BLOCKS) + " | Σ |"
    sep = "|---" * 10 + "|"
    hdr8 = "| metric | " + " | ".join(f"b{k}" for k in BLOCKS) + " |"
    sep8 = "|---" * 9 + "|"
    md = [f"# Per-block analysis (alpha_img = {a.alpha}, validation, {len(names)} references)", "",
          "Effects relative to T2I; fractions relative to steering all 8 blocks (=1).", "",
          "## Contribution of a single block: E(only_k) / E(all)  (Σ = additivity)", "", hdr, sep]
    for m, fr in pooled["frac_only"].items():
        md.append(f"| {LABEL[m]} | " + " | ".join(f"{fr.get(k, float('nan')):.2f}" for k in BLOCKS)
                  + f" | {pooled['additivity'].get(m, float('nan')):.2f} |")
    md += ["", "## Necessity: loss when block k is removed, (E(all) − E(drop_k)) / E(all)", "", hdr8, sep8]
    for m, fr in pooled["loss_drop"].items():
        md.append(f"| {LABEL[m]} | " + " | ".join(f"{fr.get(k, float('nan')):.2f}" for k in BLOCKS) + " |")
    md += ["", "## Accumulation: E(blocks 1..k) / E(all)", "", hdr8, sep8]
    for m, fr in pooled["curve"].items():
        md.append(f"| {LABEL[m]} | " + " | ".join(f"{fr.get(k, float('nan')):.2f}" for k in BLOCKS) + " |")
    md += ["", "## Significance (pooled over references; Holm over 8 blocks; refs = references significant on their own)", "",
           "| test | metric | " + " | ".join(f"b{k}" for k in BLOCKS) + " |", "|---" * 10 + "|"]
    for kind, title in (("only", "only_k vs T2I"), ("drop", "drop_k vs all")):
        for m in ALL:
            rows = {t["cfg"]: t for t in pooled["tests"] if t["metric"] == m and t["cfg"].startswith(kind)}
            if rows:
                md.append(f"| {title} | {LABEL[m]} | " + " | ".join(
                    (f"{rows[f'{kind}_{k}']['p_holm']:.2g}{'*' if rows[f'{kind}_{k}']['sig'] else ''} "
                     f"({rows[f'{kind}_{k}']['refs_sig']}/{len(names)})") if f"{kind}_{k}" in rows else "—"
                    for k in BLOCKS) + " |")
    md += ["", "## Geometry", "", "| | " + " | ".join(f"b{k}" for k in BLOCKS) + " |", "|---" * 9 + "|"]
    for key, title, f in (("token_norm", "‖h‖ (img)", "{:.0f}"), ("angle_deg", f"arctan(α/‖h‖), °", "{:.2f}"),
                          ("vector_norm", "‖v‖ (img)", "{:.1f}"), ("vector_to_token_norm", "‖v‖/‖h‖", "{:.3f}")):
        if geo[key]:
            md.append(f"| {title} | " + " | ".join(f.format(geo[key][k]) if k in geo[key] else "—" for k in BLOCKS) + " |")
    md += ["", "Spearman over blocks, single-block contribution vs geometry:", ""]
    md += [f"- {LABEL[m]}: ρ(angle) = {v['rho_angle']:+.2f} (p = {v['p']:.2g})"
           + (f", ρ(‖v‖/‖h‖) = {v['rho_v_over_h']:+.2f} (p = {v['p_v_over_h']:.2g})" if "rho_v_over_h" in v else "")
           for m, v in geo["spearman"].items()]
    md += ["", "## Per reference: E(only_k)/E(all), DINOv3 to reference", "",
           "| reference | " + " | ".join(f"b{k}" for k in BLOCKS) + " | Σ |", sep]
    for n in names:
        fr = rep["per_ref"][n]["frac_only"].get("dinov3_to_reference", {})
        md.append(f"| {n} | " + " | ".join(f"{fr.get(k, float('nan')):.2f}" for k in BLOCKS)
                  + f" | {rep['per_ref'][n]['additivity'].get('dinov3_to_reference', float('nan')):.2f} |")
    (out / "blocks.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42})
        comma = FuncFormatter(lambda v, p: ("%g" % round(v, 3)).replace(".", ","))
        main_m = [m for m in ("dinov3_to_reference", "csd_to_reference", "gram_to_reference", "lpips_to_origin")
                  if m in pooled["frac_only"]]
        panels = [("frac_only", m) for m in main_m] + [("loss_drop", main_m[0]), ("curve", main_m[0])]
        titles = {"frac_only": "вклад блока k", "loss_drop": "потеря без блока k", "curve": "блоки 1..k"}
        fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8))
        for ax, (key, m) in zip(axs.flat, panels):
            for n in names:
                d = rep["per_ref"][n][key].get(m, {})
                ax.plot([k for k in BLOCKS if k in d], [d[k] for k in BLOCKS if k in d], color="#a0a0a0", lw=0.7,
                        marker="o", ms=1.8)
            d = pooled[key].get(m, {})
            ax.plot([k for k in BLOCKS if k in d], [d[k] for k in BLOCKS if k in d], color="black", lw=1.5, marker="o", ms=3)
            ax.axhline(1 if key == "curve" else 0, color="#cccccc", lw=0.6, zorder=0)
            ax.set_title(f"{titles[key]}\n{LABEL[m]}", fontsize=7, loc="left")
            ax.set_xticks(list(BLOCKS)); ax.yaxis.set_major_formatter(comma)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        ax = axs.flat[len(panels)] if len(panels) < 8 else None
        if ax is not None and geo["angle_deg"]:
            ax.plot(list(geo["angle_deg"]), list(geo["angle_deg"].values()), color="black", marker="s", ms=3, lw=1.3)
            ax.set_title("угол arctg(α/‖h‖), °", fontsize=7, loc="left"); ax.set_xticks(list(BLOCKS))
            ax.yaxis.set_major_formatter(comma)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for extra in axs.flat[len(panels) + 1:]:
            extra.axis("off")
        for ax in axs[-1]:
            ax.set_xlabel("блок", fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figs" / "per_block.pdf", bbox_inches="tight")
        fig.savefig(out / "figs" / "per_block.png", dpi=150, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print(f"figure skipped: {exc}")

    # ---- qualitative grid: rows = references, columns = T2I, only_1..only_8, all
    try:
        from PIL import Image, ImageDraw
        def by_index(d):
            return {int(INDEX_RE.match(p.stem).group(1)): p for p in d.glob("*.png") if INDEX_RE.match(p.stem)} \
                if d.is_dir() else {}
        cell, pad, head = 192, 4, 18
        cols = ["T2I"] + [f"блок {k}" for k in BLOCKS] + ["все 8"]
        for idx in a.grid_prompts:
            rows = []
            for n in names:
                ref = Path(a.exp_root) / n
                cells = [by_index(ref / "val/alpha_2/origin").get(idx)]
                cells += [by_index(ref / f"val/blocks/only_{k}/steered").get(idx) for k in BLOCKS]
                cells += [by_index(ref / f"val/alpha_{a.alpha}/steered").get(idx)]
                rows.append(cells)
            W = len(cols) * (cell + pad); H = head + len(rows) * (cell + pad)
            canvas = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(canvas)
            for c, t in enumerate(cols):
                dr.text((c * (cell + pad) + 4, 3), t, fill="black")
            for r_, cells in enumerate(rows):
                for c, pth in enumerate(cells):
                    if pth:
                        im = Image.open(pth).convert("RGB"); im.thumbnail((cell, cell))
                        canvas.paste(im, (c * (cell + pad), head + r_ * (cell + pad)))
            canvas.save(out / "figs" / f"per_block_grid_p{idx}.jpg", quality=92)
    except Exception as exc:  # noqa: BLE001
        print(f"grid skipped: {exc}")
    print(f"wrote {out}/blocks.json, blocks.md, figs/per_block.*, figs/per_block_grid_p*.jpg")


if __name__ == "__main__":
    main()
