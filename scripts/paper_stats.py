"""Statistics for the paper revision (review item 12). CPU only; reads the score JSONs.

Input: <exp_root>/<ref>/scores/*.json written by the `score` stage
       (ours_seed_S, t2i_seed_S, shift_text_seed_S, teacher_seed_S | teacher, ours_rule_seed_S,
        val_alpha_A, val_<ablation>).

1. Test set, per reference and method: per-prompt mean over the available seeds, then
   mean and 95% CI by a *cluster* bootstrap over prompts (10 000 resamples). This removes the
   dependence of the three images of one prompt that the image-level bootstrap ignored.
2. Ours vs each other method: Wilcoxon signed-rank on the per-prompt seed means (seeds common
   to both runs; 100 pairs), Holm correction over all tests of one reference. Effect size: the
   matched-pairs rank-biserial correlation r.
3. Aggregate over references: mean of the per-reference means and the number of references in
   which the difference has the same sign / is significant.
4. Ablations (validation, 25 prompts, 1 seed): each configuration vs the main one (val_alpha_<A>),
   Wilcoxon per reference with Holm over configurations x metrics, and a pooled test on the
   reference-averaged per-prompt values (25 pairs).

    python scripts/paper_stats.py --alpha 6
Writes paper_results/stats.json and paper_results/stats.md.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["csd_to_reference", "gram_to_reference", "dinov2_to_reference", "dinov3_to_reference",
           "dinov2_to_origin", "lpips_to_origin", "dists_to_origin", "clip_text"]
LOWER_BETTER = {"gram_to_reference", "lpips_to_origin", "dists_to_origin"}
SCALE = {"gram_to_reference": 1e5}
METHODS = ["t2i", "shift_text", "teacher", "ours", "ours_rule", "act", "casteer", "ipadapter"]
SEED_RE = re.compile(r"_seed_(\d+)$")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def runs_of(S: Path, method: str) -> Dict[str, dict]:
    """seed -> per_image dict. The unpaired legacy teacher.json gets seed 'legacy'."""
    out = {}
    for f in S.glob(f"{method}_seed_*.json"):
        out[SEED_RE.search(f.stem).group(1)] = load(f)["per_image"]
    if method == "teacher" and not out and (S / "teacher.json").is_file():
        out["legacy"] = load(S / "teacher.json")["per_image"]
    return out


def prompt_means(runs: Dict[str, dict], metric: str, seeds: Optional[List[str]] = None) -> Dict[int, float]:
    acc = defaultdict(list)
    for seed, per in runs.items():
        if seeds is not None and seed not in seeds:
            continue
        for k, v in per.get(metric, {}).items():
            acc[int(k)].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cluster_ci(vals: Dict[int, float], n_boot: int = 10000, seed: int = 0) -> Optional[dict]:
    if not vals:
        return None
    a = np.array(list(vals.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, (n_boot, a.size))].mean(1)
    return {"mean": float(a.mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n_prompts": int(a.size)}


def holm(p: List[float]) -> List[float]:
    order, m, run, adj = np.argsort(p), len(p), 0.0, [0.0] * len(p)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def paired(a: Dict[int, float], b: Dict[int, float]) -> Optional[dict]:
    common = sorted(set(a) & set(b))
    if len(common) < 5:
        return None
    x, y = np.array([a[k] for k in common]), np.array([b[k] for k in common])
    d = x - y
    if np.allclose(d, 0):
        return {"n": len(common), "mean_diff": 0.0, "p": 1.0, "r_rb": 0.0}
    res = wilcoxon(x, y, zero_method="wilcox")
    nz = d[d != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    r_rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {"n": len(common), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "p": float(res.pvalue), "r_rb": r_rb}


def fmt(x, metric, digits=3):
    return "—" if x is None else f"{x * SCALE.get(metric, 1):.{digits}f}"


def better(metric, diff):  # is "ours - other = diff" an improvement?
    return diff < 0 if metric in LOWER_BETTER else diff > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="experiments/klein_9b/paper")
    ap.add_argument("--alpha", default="6", help="main configuration for the ablation tests (val_alpha_<A>)")
    ap.add_argument("--out", default="paper_results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    refs = sorted(p for p in Path(args.exp_root).iterdir() if (p / "scores").is_dir())
    report = {"test": {}, "tests": {}, "aggregate": {}, "ablation": {}, "ablation_pooled": {}}
    val_by_cfg = defaultdict(lambda: defaultdict(list))  # cfg -> metric -> [per-ref {prompt: v}]

    for ref in refs:
        S, name = ref / "scores", ref.name
        runs = {m: runs_of(S, m) for m in METHODS}
        report["test"][name] = {m: {met: cluster_ci(prompt_means(r, met)) for met in METRICS}
                                for m, r in runs.items() if r}
        rows = []
        for other in ("t2i", "shift_text", "teacher", "ours_rule", "act", "casteer", "ipadapter"):
            if not runs[other] or not runs["ours"]:
                continue
            common = sorted(set(runs["ours"]) & set(runs[other]))
            seeds_o, seeds_x = (common, common) if common else (None, None)  # legacy teacher: unpaired noise
            for met in METRICS:
                t = paired(prompt_means(runs["ours"], met, seeds_o), prompt_means(runs[other], met, seeds_x))
                if t:
                    rows.append({"other": other, "metric": met, "seeds": common or "all/legacy", **t})
        for row, p in zip(rows, holm([r["p"] for r in rows])):
            row["p_holm"], row["sig"] = p, p < 0.05
        report["tests"][name] = rows

        # validation configs for the ablation table
        main_f = S / f"val_alpha_{args.alpha}.json"
        if main_f.is_file():
            main = load(main_f)["per_image"]
            abl = {f.stem[4:]: load(f)["per_image"] for f in S.glob("val_*.json")
                   if not f.stem.startswith(("val_alpha_", "val_act_", "val_casteer_", "val_ipadapter_", "val_blk_", "val_t2i", "val_nameprobe"))}
            rows = []
            for cfg, per in sorted(abl.items()):
                for met in METRICS:
                    a = {int(k): v for k, v in per.get(met, {}).items()}
                    b = {int(k): v for k, v in main.get(met, {}).items()}
                    t = paired(a, b)
                    if t:
                        rows.append({"config": cfg, "metric": met, "mean_cfg": float(np.mean(list(a.values()))),
                                     "mean_main": float(np.mean(list(b.values()))), **t})
                    if a:
                        val_by_cfg[cfg][met].append(a)
                    if b and cfg == sorted(abl)[0]:
                        val_by_cfg["main"][met].append(b)
            for row, p in zip(rows, holm([r["p"] for r in rows])):
                row["p_holm"], row["sig"] = p, p < 0.05
            report["ablation"][name] = rows

    # ---- aggregate over references (test)
    names = list(report["test"])
    for m in METHODS:
        agg = {}
        for met in METRICS:
            vals = [report["test"][n][m][met]["mean"] for n in names
                    if m in report["test"][n] and report["test"][n][m].get(met)]
            if vals:
                agg[met] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
                            "n_refs": len(vals)}
        if agg:
            report["aggregate"][m] = agg
    sign = defaultdict(lambda: {"better": 0, "sig_better": 0, "sig_worse": 0, "n": 0})
    for n in names:
        for r in report["tests"][n]:
            s = sign[f"{r['other']}|{r['metric']}"]
            s["n"] += 1
            s["better"] += int(better(r["metric"], r["mean_diff"]))
            s["sig_better"] += int(r["sig"] and better(r["metric"], r["mean_diff"]))
            s["sig_worse"] += int(r["sig"] and not better(r["metric"], r["mean_diff"]))
    report["sign_counts"] = dict(sign)
    # relative Gram change vs T2I per reference
    report["delta_gram_vs_t2i_pct"] = {}
    for m in ("ours", "teacher", "shift_text", "ours_rule", "act"):
        d = {n: 100 * (report["test"][n][m]["gram_to_reference"]["mean"] /
                       report["test"][n]["t2i"]["gram_to_reference"]["mean"] - 1)
             for n in names if m in report["test"][n] and "t2i" in report["test"][n]
             and report["test"][n][m].get("gram_to_reference")}
        if d:
            report["delta_gram_vs_t2i_pct"][m] = {"per_ref": d, "mean": float(np.mean(list(d.values())))}

    # ---- pooled ablation test: average over references per prompt (same 25 val prompts)
    def ref_avg(lst):
        common = set.intersection(*[set(x) for x in lst]) if lst else set()
        return {k: float(np.mean([x[k] for x in lst])) for k in common}
    rows = []
    for cfg in sorted(c for c in val_by_cfg if c != "main"):
        for met in METRICS:
            if val_by_cfg[cfg][met] and val_by_cfg["main"][met]:
                t = paired(ref_avg(val_by_cfg[cfg][met]), ref_avg(val_by_cfg["main"][met]))
                if t:
                    n_better = sum(1 for n in report["ablation"] for r in report["ablation"][n]
                                   if r["config"] == cfg and r["metric"] == met
                                   and better(met, r["mean_cfg"] - r["mean_main"]))
                    n_sig = sum(1 for n in report["ablation"] for r in report["ablation"][n]
                                if r["config"] == cfg and r["metric"] == met and r["sig"])
                    rows.append({"config": cfg, "metric": met, **t, "refs_better": n_better, "refs_sig": n_sig})
    for row, p in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"], row["sig"] = p, p < 0.05
    report["ablation_pooled"] = rows

    # ---- name probe: does the model know the style by name? (val, 25 prompts, seed 42)
    report["name_probe"] = {}
    for ref in refs:
        S = ref / "scores"
        if not (S / "val_nameprobe.json").is_file() or not (S / "val_t2i.json").is_file():
            continue
        probe, t2i = load(S / "val_nameprobe.json")["per_image"], load(S / "val_t2i.json")["per_image"]
        ours = load(S / f"val_alpha_{args.alpha}.json")["per_image"] if (S / f"val_alpha_{args.alpha}.json").is_file() else {}
        rows = {}
        for met in ("csd_to_reference", "gram_to_reference", "dinov2_to_reference", "dinov3_to_reference"):
            if met not in probe or met not in t2i:
                continue
            a = {int(k): v for k, v in probe[met].items()}; b = {int(k): v for k, v in t2i[met].items()}
            row = {"t2i": float(np.mean(list(b.values()))), "name": float(np.mean(list(a.values())))}
            tt = paired(a, b)
            if tt:
                row.update(p_name_vs_t2i=tt["p"])
            if met in ours:
                o = {int(k): v for k, v in ours[met].items()}
                row["ours"] = float(np.mean(list(o.values())))
                # share of our shift that the artist name alone reaches
                d_ours = row["ours"] - row["t2i"]
                row["name_share_of_ours"] = (row["name"] - row["t2i"]) / d_ours if abs(d_ours) > 1e-12 else None
            rows[met] = row
        report["name_probe"][ref.name] = rows

    (out / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- markdown
    md = ["# Statistics (scripts/paper_stats.py)", "",
          "Per-prompt means over seeds; CI = cluster bootstrap over prompts; Wilcoxon on per-prompt means, "
          "Holm within reference. Gram ×1e5.", "", "## Aggregate over references (test)", "",
          "| method | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for m, agg in report["aggregate"].items():
        md.append(f"| {m} | " + " | ".join(
            f"{fmt(agg[k]['mean'], k)} ({fmt(agg[k]['min'], k)}–{fmt(agg[k]['max'], k)})" if k in agg else "—"
            for k in METRICS) + " |")
    md += ["", "ΔGram vs T2I, %: " + "; ".join(
        f"{m} {v['mean']:+.1f}" for m, v in report["delta_gram_vs_t2i_pct"].items()), "",
        "## Sign / significance counts over references (ours − other)", "",
        "| comparison | better | sig. better | sig. worse | n refs |", "|---|---|---|---|---|"]
    for k, s in sorted(report["sign_counts"].items()):
        md.append(f"| {k} | {s['better']} | {s['sig_better']} | {s['sig_worse']} | {s['n']} |")
    for n in names:
        md += ["", f"## {n}", "", "| method | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
        for m, st in report["test"][n].items():
            md.append(f"| {m} | " + " | ".join(
                f"{fmt(st[k]['mean'], k)} [{fmt(st[k]['ci95'][0], k)}; {fmt(st[k]['ci95'][1], k)}]"
                if st.get(k) else "—" for k in METRICS) + " |")
        md += ["", "| ours vs | metric | seeds | mean diff | r_rb | p_holm |", "|---|---|---|---|---|---|"]
        for r in report["tests"][n]:
            md.append(f"| {r['other']} | {r['metric']} | {r['seeds']} | {fmt(r['mean_diff'], r['metric'], 4)} | "
                      f"{r['r_rb']:+.2f} | {r['p_holm']:.3g}{' *' if r['sig'] else ''} |")
    md += ["", f"## Ablations vs main (val_alpha_{args.alpha}), pooled over references (25 prompts)", "",
           "| config | metric | mean diff (cfg − main) | r_rb | p_holm | refs better | refs sig. |",
           "|---|---|---|---|---|---|---|"]
    for r in report["ablation_pooled"]:
        md.append(f"| {r['config']} | {r['metric']} | {fmt(r['mean_diff'], r['metric'], 4)} | {r['r_rb']:+.2f} | "
                  f"{r['p_holm']:.3g}{' *' if r['sig'] else ''} | {r['refs_better']}/{len(report['ablation'])} | {r['refs_sig']}/{len(report['ablation'])} |")
    if report["name_probe"]:
        md += ["", "## Name probe (val): prompt + ', in the style of <artist>' vs T2I vs ours", "",
               "| reference | metric | T2I | name only | ours | name / ours shift | p (name vs T2I) |",
               "|---|---|---|---|---|---|---|"]
        for n, rows in report["name_probe"].items():
            for met, r in rows.items():
                sh = r.get("name_share_of_ours")
                md.append(f"| {n} | {met} | {fmt(r['t2i'], met)} | {fmt(r['name'], met)} | {fmt(r.get('ours'), met)} | "
                          f"{'—' if sh is None else f'{sh:.2f}'} | {r.get('p_name_vs_t2i', float('nan')):.3g} |")
    (out / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}/stats.json, stats.md")


if __name__ == "__main__":
    main()
