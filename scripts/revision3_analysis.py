"""Revision 3 analysis (CPU): angle-matched control and held-out confirmation of the per-block results.

Inputs (experiments/klein_9b/paper/<ref>/scores/, written by scripts/run_revision3.sh):
  val_angle_<th>_{only_k,all}.json, val_t2i.json                    angle control, VAL, seed 42
  test_angle_<th>_{only_k,all}_seed_42.json, t2i_seed_42.json      angle control, TEST, seed 42
  test_blk_<cfg>_seed_<S>.json, ours_seed_<S>.json, t2i_seed_<S>.json   held-out blocks, 3 seeds
  shift_rule_seed_<S>.json, val_shift_alpha_<A>.json                SHIFT with the selection rule

Pre-registered reading of the angle control (stated in the paper before the run):
  if, at one and the same rotation angle in every block, the single-block effect still decreases
  with depth (Spearman rho with depth > 0, permutation p < 0.05, pooled over references), the decrease
  is NOT explained by the angle alone; if the profile is flat (|rho| small, n.s.) and block 8 reaches
  the effect of block 1 (paired Wilcoxon n.s. and ratio CI covering 1), the angle account is supported.

Effects per prompt (as in block_analysis.py): CSD/DINO: x - x_T2I; Gram: ln(G_T2I / G); LPIPS: value.
Held-out tests: per-prompt means over seeds; cfg vs full (ours, alpha) paired Wilcoxon, rank-biserial r,
cluster-bootstrap 95 % CI of the mean difference and of the relative difference; Holm over the
configurations x metrics within each reference and within the pooled analysis.

    python scripts/revision3_analysis.py --exp_root experiments/klein_9b/paper --out paper_results_revision3
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["csd_to_reference", "gram_to_reference", "dinov3_to_reference", "lpips_to_origin", "dinov2_to_origin"]
BLK_CFGS = ["only_1", "drop_1", "drop_8", "blocks_1-4", "blocks_5-8", "steps_3-4"]


def load(p: Path):
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))["per_image"]
    return {m: {int(k): float(v) for k, v in vals.items()} for m, vals in d.items()}


def effect(cfg, t2i, m):
    if cfg is None or m not in cfg:
        return None
    x = cfg[m]
    if m in ("lpips_to_origin",):
        return dict(x)
    if m == "dinov2_to_origin":
        return {k: 1 - v for k, v in x.items()}
    if t2i is None or m not in t2i:
        return None
    b = t2i[m]; ks = set(x) & set(b)
    if m == "gram_to_reference":
        return {k: math.log(b[k] / x[k]) for k in ks if x[k] > 0 and b[k] > 0}
    return {k: x[k] - b[k] for k in ks}


def avg_seeds(dicts):
    dicts = [d for d in dicts if d]
    if not dicts:
        return None
    ks = set.intersection(*(set(d) for d in dicts))
    return {k: float(np.mean([d[k] for d in dicts])) for k in ks}


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v); rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def rank_biserial(d):
    d = np.asarray(d); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired(a, b):
    ks = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in ks]), np.array([b[k] for k in ks])
    d = x - y
    p = float(wilcoxon(x, y).pvalue) if len(ks) >= 5 and not np.allclose(d, 0) else 1.0
    rel = d / np.where(np.abs(y.mean()) > 1e-12, abs(y.mean()), 1.0)
    return {"n": len(ks), "mean_cfg": float(x.mean()), "mean_ref": float(y.mean()), "mean_diff": float(d.mean()),
            "ci_diff": boot_ci(d), "rel_diff": float(rel.mean()), "ci_rel": boot_ci(rel), "r_rb": rank_biserial(d), "p": p}


def holm(rows):
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"]); run = 0.0; m = len(rows)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * rows[i]["p"])); rows[i]["p_holm"] = run
    return rows


_PERMS = None


def perm_spearman(x, y):
    """Spearman rho and exact two-sided permutation p (all n! orderings; n = 8 blocks)."""
    global _PERMS
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    n = len(x)
    if _PERMS is None or _PERMS.shape[1] != n:
        _PERMS = np.array(list(itertools.permutations(range(n))), dtype=float)
    rho = lambda a: 1 - 6 * ((a - rx) ** 2).sum(-1) / (n * (n * n - 1))
    r0 = float(rho(ry))
    return r0, float((np.abs(rho(_PERMS)) >= abs(r0) - 1e-12).mean())


def angle_control(refs, split, th):
    """Single-block effects at a fixed rotation angle; pooled = per-prompt mean over references."""
    per_ref, pooled_e = {}, {m: {k: [] for k in range(1, 10)} for m in METRICS}
    for ref in refs:
        S = ref / "scores"
        t2i = load(S / ("val_t2i.json" if split == "val" else "t2i_seed_42.json"))
        suf = "" if split == "val" else "_seed_42"
        res = {}
        for m in METRICS:
            eff = {}
            for k in range(1, 9):
                eff[k] = effect(load(S / f"{split}_angle_{th}_only_{k}{suf}.json"), t2i, m)
            eff[9] = effect(load(S / f"{split}_angle_{th}_all{suf}.json"), t2i, m)  # 9 = all blocks
            if any(e is None for e in eff.values()):
                continue
            means = {k: float(np.mean(list(e.values()))) for k, e in eff.items()}
            rho, p = perm_spearman(list(range(1, 9)), [-means[k] for k in range(1, 9)])
            res[m] = {"mean_effect": means, "frac_only": {k: means[k] / means[9] if abs(means[9]) > 1e-9 else None for k in range(1, 9)},
                      "rho_depth_decrease": rho, "p_perm": p, "block1_vs_block8": paired(eff[1], eff[8])}
            for k in range(1, 10):
                pooled_e[m][k].append(eff[k])
        per_ref[ref.name] = res
    pooled = {}
    for m in METRICS:
        if not all(pooled_e[m][k] for k in range(1, 10)):
            continue
        eff = {k: avg_seeds(pooled_e[m][k]) for k in range(1, 10)}
        means = {k: float(np.mean(list(e.values()))) for k, e in eff.items()}
        rho, p = perm_spearman(list(range(1, 9)), [-means[k] for k in range(1, 9)])
        pooled[m] = {"mean_effect": means, "rho_depth_decrease": rho, "p_perm": p,
                     "block1_vs_block8": paired(eff[1], eff[8]),
                     "only_k_vs_zero_p": {k: float(wilcoxon(list(eff[k].values())).pvalue) for k in range(1, 9)}}
    return {"per_ref": per_ref, "pooled": pooled}


def heldout_blocks(refs, seeds):
    per_ref, pooled_raw = {}, {}
    for ref in refs:
        S = ref / "scores"; rows = []
        t2i = {s: load(S / f"t2i_seed_{s}.json") for s in seeds}
        full = {s: load(S / f"ours_seed_{s}.json") for s in seeds}
        for c in BLK_CFGS:
            cfg = {s: load(S / f"test_blk_{c}_seed_{s}.json") for s in seeds}
            if not all(cfg.values()):
                continue
            for m in METRICS:
                e_c = avg_seeds([effect(cfg[s], t2i[s], m) for s in seeds])
                e_f = avg_seeds([effect(full[s], t2i[s], m) for s in seeds])
                if e_c is None or e_f is None:
                    continue
                row = {"cfg": c, "metric": m, **paired(e_c, e_f)}
                if c == "only_1":  # only_1 vs no steering
                    row["p_vs_t2i"] = float(wilcoxon(list(e_c.values())).pvalue)
                rows.append(row)
                pooled_raw.setdefault((c, m), {"c": [], "f": []})
                pooled_raw[(c, m)]["c"].append(e_c); pooled_raw[(c, m)]["f"].append(e_f)
        per_ref[ref.name] = holm(rows) if rows else []
    pooled = [{"cfg": c, "metric": m, **paired(avg_seeds(v["c"]), avg_seeds(v["f"]))}
              for (c, m), v in pooled_raw.items() if len(v["c"]) == len(refs)]
    return {"per_ref": per_ref, "pooled": holm(pooled) if pooled else []}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="experiments/klein_9b/paper")
    ap.add_argument("--alpha", default="6")
    ap.add_argument("--seeds", nargs="*", default=["42", "1042", "2042"])
    ap.add_argument("--thetas", nargs="*", default=["2.6", "3.0"])
    ap.add_argument("--out", default="paper_results_revision3")
    a = ap.parse_args()
    refs = sorted(p for p in Path(a.exp_root).iterdir() if (p / "scores").is_dir())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"angle_val": {th: angle_control(refs, "val", th) for th in a.thetas},
           "angle_test": {th: angle_control(refs, "test", th) for th in a.thetas},
           "blocks_test": heldout_blocks(refs, a.seeds),
           "shift_rule_alpha": {r.name: (r / "shift_rule_alpha.txt").read_text().strip()
                                for r in refs if (r / "shift_rule_alpha.txt").is_file()}}
    (out / "revision3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["# Revision 3 analysis", ""]
    for split in ("angle_val", "angle_test"):
        for th, r in res[split].items():
            for m, v in r["pooled"].items():
                lines.append(f"{split} theta={th} {m}: rho(depth, decrease)={v['rho_depth_decrease']:.2f} p={v['p_perm']:.4f}; "
                             f"block1-block8 diff={v['block1_vs_block8']['mean_diff']:.4f} p={v['block1_vs_block8']['p']:.3g}; "
                             f"means={ {k: round(x, 4) for k, x in v['mean_effect'].items()} }")
    lines.append("")
    for row in res["blocks_test"]["pooled"]:
        lines.append(f"heldout {row['cfg']:<11} {row['metric']:<20} cfg-full={row['mean_diff']:+.4f} CI={row['ci_diff']} "
                     f"rel={row['rel_diff']:+.2f} CIrel={row['ci_rel']} r={row['r_rb']:+.2f} p_holm={row['p_holm']:.3g}")
    (out / "revision3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
