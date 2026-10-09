"""Stability and reference-specificity of the extracted steering vectors (review item 3).

CPU only, no generation. Reads what the extract stage already saved:
  <exp_root>/<ref>/ref/data_vectors/*_{pos,neg}_block.pt   per-prompt token-mean activations ("pooled")
  <exp_root>/<ref>/ref/vectors/*_diff.pt                   per-token mean-difference field (Eq. 10)
  <exp_root>/<ref>/text/vectors/*_diff.pt                  text-SHIFT vector (optional)
and optionally a second root with same-style "twin" references (--twin_root).

Per reference, step, block and stream (from the pooled per-prompt differences d_k = h+_k - h-_k):
  split_half     cosine between the mean of d over two random disjoint halves of the prompts
                 (200 splits; mean and 2.5/97.5 percentiles): is the direction reproducible
                 when the extraction prompts change?
  loo_cos        mean cosine of d_k to the mean of the other prompts: how prompt-independent is
                 the per-prompt effect of the reference (the hypothesis of Section 2.3)?
  signal_ratio   ||mean_k d_k|| / mean_k ||d_k||  (1 = identical shifts for all prompts)
Across references (img stream):
  cross_pooled   cosine of the pooled mean differences of two references
  cross_field    cosine of the full per-token fields (64x64 positions x D, flattened)
  specific       the same after subtracting the mean over the seven references
                 ("presence of a reference" component), i.e. the reference-specific part
  text_vs_ref    cosine of text-SHIFT and reference vectors of the same reference
Twins: cosine of each twin to its own main reference vs. to the other references.

    python scripts/vector_stability.py
    python scripts/vector_stability.py --twin_root experiments/klein_9b/twins

Writes paper_results/stability.json, stability.md and figs/vector_similarity.{pdf,png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

TWIN_OF = {"monet_water_lily_pond": "monet_water_lilies", "vangogh_bedroom": "vangogh_self_portrait"}
STREAMS = ("img", "txt")


def tload(path: Path):
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except Exception:  # old serialization format or torch without mmap
        return torch.load(path, map_location="cpu", weights_only=False)


def one(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(p for p in directory.glob(pattern)
                   if not any(s in p.name for s in ("text_diff", "svm", "scores", "normals")))
    return files[0] if files else None


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def pooled_diffs(ref_dir: Path):
    dv = ref_dir / "ref" / "data_vectors"
    pos_p, neg_p = one(dv, "*_pos_block.pt"), one(dv, "*_neg_block.pt")
    if not pos_p or not neg_p:
        return None
    pos, neg = tload(pos_p)["pooled"], tload(neg_p)["pooled"]
    out = {}
    for step in pos:
        for layer in pos[step]:
            for s in STREAMS:
                if s in pos[step][layer] and s in neg[step][layer]:
                    out[(int(step), int(layer.split("_")[1]), s)] = (
                        pos[step][layer][s].double() - neg[step][layer][s].double())
    return out


def within_reference(d: Dict, n_splits: int, rng) -> Dict:
    res = {}
    for key, D in d.items():  # D: (N, dim)
        n = D.shape[0]
        sh = []
        for _ in range(n_splits):
            perm = torch.from_numpy(rng.permutation(n))
            a, b = perm[: n // 2], perm[n // 2:]
            sh.append(cos(D[a].mean(0), D[b].mean(0)))
        total = D.sum(0)
        loo = [cos(D[k], (total - D[k]) / (n - 1)) for k in range(n)]
        res[key] = {"split_half": float(np.mean(sh)), "split_half_ci": [float(np.percentile(sh, 2.5)),
                                                                          float(np.percentile(sh, 97.5))],
                    "loo_cos": float(np.mean(loo)),
                    "signal_ratio": float(D.mean(0).norm() / D.norm(dim=1).mean())}
    return res


def summarize(res: Dict, stream: str, by: str = "all") -> Dict:
    groups: Dict[str, List[dict]] = {}
    for (step, layer, s), v in res.items():
        if s != stream:
            continue
        g = {"all": "all", "block": f"blocks_{1 + 4 * (layer // 4)}-{4 + 4 * (layer // 4)}",
             "layer": f"block_{layer + 1}", "step": f"step_{step + 1}"}[by]
        groups.setdefault(g, []).append(v)
    return {g: {m: float(np.mean([x[m] for x in vs])) for m in ("split_half", "loo_cos", "signal_ratio")}
            for g, vs in sorted(groups.items())}


def mean_vectors(ref_dir: Path, kind: str = "ref"):
    p = one(ref_dir / kind / "vectors", "*_diff.pt")
    return tload(p) if p else None


def matrix(names, get, pairs_fn=cos):
    m = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j >= i:
                m[i, j] = m[j, i] = pairs_fn(get(a), get(b)) if i != j else 1.0
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_root", default="experiments/klein_9b/paper")
    ap.add_argument("--twin_root", default="experiments/klein_9b/twins")
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--n_splits", type=int, default=200)
    ap.add_argument("--step", type=int, default=None, help="restrict field cosines to one step (0-3)")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out); (out / "figs").mkdir(parents=True, exist_ok=True)

    roots = {p.name: p for p in sorted(Path(args.exp_root).iterdir()) if p.is_dir()}
    twins = {p.name: p for p in sorted(Path(args.twin_root).iterdir()) if p.is_dir()} \
        if Path(args.twin_root).is_dir() else {}
    report = {"within": {}, "cross": {}, "twins": {}, "text_vs_ref": {}}

    # ---- within-reference stability (pooled per-prompt differences)
    pooled_mean = {}
    for name, d in {**roots, **twins}.items():
        pdiff = pooled_diffs(d)
        if pdiff is None:
            print(f"skip {name}: no data_vectors")
            continue
        w = within_reference(pdiff, args.n_splits, rng)
        report["within"][name] = {s: {"all": summarize(w, s)["all"], "by_block": summarize(w, s, "block"),
                                      "by_layer": summarize(w, s, "layer"),
                                      "by_step": summarize(w, s, "step")} for s in STREAMS}
        pooled_mean[name] = {k: v.mean(0) for k, v in pdiff.items()}
        print(f"{name}: split-half img {report['within'][name]['img']['all']['split_half']:.3f}, "
              f"loo {report['within'][name]['img']['all']['loo_cos']:.3f}")

    # ---- cross-reference similarity (img stream), averaged over (step, block)
    mains = [n for n in roots if n in pooled_mean]
    keys = sorted(k for k in pooled_mean[mains[0]] if k[2] == "img"
                  and (args.step is None or k[0] == args.step)) if mains else []

    def cross(names, vec_of):
        mats = [matrix(names, lambda n, k=k: vec_of(n, k)) for k in keys]
        return np.nanmean(mats, axis=0)

    if len(mains) > 1:
        m_pool = cross(mains, lambda n, k: pooled_mean[n][k])
        common = {k: torch.stack([pooled_mean[n][k] for n in mains]).mean(0) for k in keys}
        m_spec = cross(mains, lambda n, k: pooled_mean[n][k] - common[k])
        offd = ~np.eye(len(mains), dtype=bool)
        report["cross"]["pooled_offdiag_by_block"] = {}
        for layer in sorted({k[1] for k in keys}):
            ks = [k for k in keys if k[1] == layer]
            mm = np.nanmean([matrix(mains, lambda n, k=k: pooled_mean[n][k]) for k in ks], axis=0)
            report["cross"]["pooled_offdiag_by_block"][f"block_{layer + 1}"] = float(mm[offd].mean())
        report["cross"]["names"] = mains
        report["cross"]["pooled"] = m_pool.tolist()
        report["cross"]["pooled_specific"] = m_spec.tolist()
        off = ~np.eye(len(mains), dtype=bool)
        report["cross"]["pooled_offdiag_mean"] = float(m_pool[off].mean())
        report["cross"]["specific_offdiag_mean"] = float(m_spec[off].mean())

        # full per-token fields, streamed per (step, block) to bound memory
        fields = {n: mean_vectors(roots[n]) for n in mains}
        if all(f is not None for f in fields.values()):
            mats = []
            for (step, layer, _s) in keys:
                F = {n: fields[n][step][f"layer_{layer}"]["img"].double() for n in mains}
                mats.append(matrix(mains, lambda n: F[n]))
            m_field = np.nanmean(mats, axis=0)
            per_layer = {}
            for (step, layer, _s), mm in zip(keys, mats):
                per_layer.setdefault(layer, []).append(float(mm[off].mean()))
            report["cross"]["field_offdiag_by_block"] = {f"block_{l + 1}": float(np.mean(v))
                                                        for l, v in sorted(per_layer.items())}
            report["cross"]["field"] = m_field.tolist()
            report["cross"]["field_offdiag_mean"] = float(m_field[off].mean())

        # text-SHIFT vs reference vector of the same reference
        for n in mains:
            tv = mean_vectors(roots[n], "text")
            if tv is None or fields.get(n) is None:
                continue
            vals = [cos(tv[st][f"layer_{ly}"]["img"], fields[n][st][f"layer_{ly}"]["img"]) for st, ly, _ in keys]
            report["text_vs_ref"][n] = float(np.mean(vals))

    # ---- twins: own reference vs others
    for t in twins:
        if t not in pooled_mean or TWIN_OF.get(t) not in pooled_mean:
            continue
        sims = {n: float(np.mean([cos(pooled_mean[t][k], pooled_mean[n][k]) for k in keys])) for n in mains}
        common = {k: torch.stack([pooled_mean[n][k] for n in mains]).mean(0) for k in keys}
        spec = {n: float(np.mean([cos(pooled_mean[t][k] - common[k], pooled_mean[n][k] - common[k])
                                  for k in keys])) for n in mains}
        own = TWIN_OF[t]
        report["twins"][t] = {"own": own, "cos_to": sims, "specific_cos_to": spec,
                              "own_rank_specific": 1 + sorted(spec.values(), reverse=True).index(spec[own])}

    (out / "stability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- markdown
    md = ["# Steering-vector stability (scripts/vector_stability.py)", "",
          "## Within reference (pooled per-prompt differences, mean over steps and blocks)", "",
          "| reference | stream | split-half cos [2.5, 97.5] | LOO cos | signal ratio |", "|---|---|---|---|---|"]
    for n, w in report["within"].items():
        for s in STREAMS:
            a = w[s]["all"]
            md.append(f"| {n} | {s} | {a['split_half']:.3f} | {a['loo_cos']:.3f} | {a['signal_ratio']:.3f} |")
    md += ["", "### By block group (img stream)", "", "| reference | " + " | ".join(
        next(iter(report["within"].values()))["img"]["by_block"].keys()) + " |",
        "|---" * (1 + len(next(iter(report["within"].values()))["img"]["by_block"])) + "|"] \
        if report["within"] else []
    for n, w in report["within"].items():
        md.append(f"| {n} | " + " | ".join(f"{v['split_half']:.3f} / {v['loo_cos']:.3f}"
                                         for v in w["img"]["by_block"].values()) + " |")
    md.append("\n(cells: split-half / LOO cosine)")
    if report["within"]:
        layers = list(next(iter(report["within"].values()))["img"]["by_layer"].keys())
        md += ["", "### By single block (img stream; split-half / LOO / signal ratio)", "",
               "| reference | " + " | ".join(layers) + " |", "|---" * (len(layers) + 1) + "|"]
        for n, w in report["within"].items():
            md.append(f"| {n} | " + " | ".join(f"{v['split_half']:.3f} / {v['loo_cos']:.3f} / {v['signal_ratio']:.2f}"
                                             for v in w["img"]["by_layer"].values()) + " |")
    if report["cross"]:
        names = report["cross"]["names"]
        for key, title in (("pooled", "pooled mean difference"), ("pooled_specific", "after removing the common "
                           "component"), ("field", "full per-token field")):
            if key not in report["cross"]:
                continue
            md += ["", f"## Cross-reference cosine, {title} (img)", "", "| | " + " | ".join(names) + " |",
                   "|---" * (len(names) + 1) + "|"]
            for i, n in enumerate(names):
                md.append(f"| {n} | " + " | ".join(f"{x:.2f}" for x in report["cross"][key][i]) + " |")
        md += ["", f"Off-diagonal mean: pooled {report['cross']['pooled_offdiag_mean']:.3f}, "
               f"specific {report['cross']['specific_offdiag_mean']:.3f}"
               + (f", field {report['cross']['field_offdiag_mean']:.3f}" if "field_offdiag_mean" in report["cross"]
                  else "")]
    for key, title in (("pooled_offdiag_by_block", "pooled"), ("field_offdiag_by_block", "per-token field")):
        if report["cross"].get(key):
            md += ["", f"## Mean cross-reference cosine by block ({title}, img)", "",
                   "| " + " | ".join(report["cross"][key]) + " |", "|---" * len(report["cross"][key]) + "|",
                   "| " + " | ".join(f"{v:.3f}" for v in report["cross"][key].values()) + " |"]
    if report["text_vs_ref"]:
        md += ["", "## Text-SHIFT vs reference vector (img field cosine)", ""]
        md += [f"- {n}: {v:.3f}" for n, v in report["text_vs_ref"].items()]
    if report["twins"]:
        md += ["", "## Twins (second image of the same style)", ""]
        for t, v in report["twins"].items():
            md.append(f"- {t} (own: {v['own']}): rank of own reference by specific cosine = "
                      f"{v['own_rank_specific']}/{len(v['specific_cos_to'])}; "
                      + ", ".join(f"{n} {c:.2f}" for n, c in sorted(v['specific_cos_to'].items(),
                                                                       key=lambda kv: -kv[1])))
    (out / "stability.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if report["cross"]:
            names = report["cross"]["names"]
            short = [n.split("_")[0].capitalize() for n in names]
            panels = [k for k in ("field", "pooled_specific") if k in report["cross"]]
            fig, axs = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.0))
            axs = np.atleast_1d(axs)
            for ax, k in zip(axs, panels):
                m = np.array(report["cross"][k])
                im = ax.imshow(m, vmin=-1 if k != "field" else 0, vmax=1, cmap="RdBu_r" if k != "field" else "Greys")
                ax.set_xticks(range(len(short)), short, rotation=60, fontsize=7)
                ax.set_yticks(range(len(short)), short, fontsize=7)
                for i in range(len(m)):
                    for j in range(len(m)):
                        ax.text(j, i, f"{m[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=5.5,
                                color="white" if abs(m[i, j]) > 0.6 else "black")
                ax.set_title({"field": "(а) поле вектора", "pooled_specific": "(б) без общей составляющей"}[k],
                             fontsize=8, loc="left")
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out / "figs" / "vector_similarity.pdf", bbox_inches="tight")
            fig.savefig(out / "figs" / "vector_similarity.png", dpi=150, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print(f"figure skipped: {exc}")
    print(f"wrote {out}/stability.json, stability.md")


if __name__ == "__main__":
    main()
