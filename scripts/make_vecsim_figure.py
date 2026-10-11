"""Fig. 9 (cross-reference cosine of steering vectors) from stability.json, Russian labels,
decimal commas, references in the order of Fig. 1 (CPU, no GPU run needed).

    python scripts/make_vecsim_figure.py --stability paper_results_revision2/stability.json \
        --out article_tex/figs_v6/vector_similarity.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RU = {"monet": "Моне", "vangogh": "Ван Гог", "ciurlionis": "Чюрлёнис", "pirosmani": "Пиросмани"}
ORDER = ["monet", "vangogh", "ciurlionis", "pirosmani"]          # order of Fig. 1
PANELS = [("field", "(а) поле вектора", 0, 1, "Greys"),
          ("pooled_specific", "(б) без общей составляющей", -1, 1, "RdBu_r")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stability", default="paper_results_revision2/stability.json")
    ap.add_argument("--out", default="article_tex/figs_v6/vector_similarity.pdf")
    a = ap.parse_args()
    cross = json.loads(Path(a.stability).read_text(encoding="utf-8"))["cross"]
    keys = [n.split("_")[0] for n in cross["names"]]
    idx = [keys.index(k) for k in ORDER if k in keys]
    labels = [RU[keys[i]] for i in idx]
    comma = mticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",").replace("-", "−"))

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, (key, title, vmin, vmax, cmap) in zip(axs, PANELS):
        m = np.array(cross[key])[np.ix_(idx, idx)]
        im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        for i in range(len(m)):
            for j in range(len(m)):
                ax.text(j, i, f"{m[i, j]:.2f}".replace(".", ",").replace("-", "−"), ha="center", va="center",
                        fontsize=6.5, color="white" if abs(m[i, j]) > 0.6 else "black")
        ax.set_title(title, fontsize=8, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.ax.yaxis.set_major_formatter(comma)
        cb.ax.tick_params(labelsize=6)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
