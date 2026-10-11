"""Qualitative figure of limitations (revision 3, review item 6).

Rows: (reference, test prompt) pairs that show (i) the teacher copying content of the reference
and (ii) the absence of the spatial organisation of the style in the steered output.
Columns: reference | T2I | ours (alpha=6) | SHIFT (alpha=6) | teacher (I2I).

Two sources:
  --examples DIR   per-image files from scripts/run_revision3.sh (stage `examples`):
                   DIR/<ref>/<idx>__{t2i,ours,shift,teacher}.png and DIR/<ref>/reference.*  (with SHIFT)
  --grids DIR      revision-2 test grids DIR/figs/<ref>_test_grid.jpg (T2I | ours | teacher; no SHIFT)

    python scripts/make_limits_figure.py --grids paper_results_revision2 --out article_tex/figs_v5
"""
import argparse, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROWS = [("ciurlionis_sonata_sun", 20, "Чюрлёнис\n«a baby sleeping»"),
        ("ciurlionis_sonata_sun", 60, "Чюрлёнис\n«a rice terrace»"),
        ("pirosmani_giraffe", 60, "Пиросмани\n«a rice terrace»"),
        ("vangogh_self_portrait", 20, "Ван Гог\n«a baby sleeping»"),
        ("vangogh_self_portrait", 40, "Ван Гог\n«a pocket watch»")]
GRID_ROW = {0: 0, 10: 1, 20: 2, 40: 3, 60: 4, 80: 5}   # rows of the revision-2 test grid
GRID_COL = {"t2i": 0, "ours": 1, "teacher": 2}


def from_grid(res, ref, idx, kind):
    g = Image.open(os.path.join(res, "figs", f"{ref}_test_grid.jpg")).convert("RGB")
    x, y = GRID_COL[kind] * 326, 22 + GRID_ROW[idx] * 326
    return g.crop((x, y, x + 320, y + 320))


def from_examples(d, ref, idx, kind):
    f = glob.glob(os.path.join(d, ref, f"{idx:02d}__{kind}.png"))
    return Image.open(f[0]).convert("RGB") if f else None


def reference(src, ref, grids):
    pat = os.path.join(src, "figs", f"{ref}_reference.jpg") if grids else os.path.join(src, ref, "reference.*")
    f = glob.glob(pat)
    return Image.open(f[0]).convert("RGB") if f else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids"); ap.add_argument("--examples")
    ap.add_argument("--out", default="article_tex/figs_v5")
    a = ap.parse_args()
    grids = a.grids is not None
    src = a.grids or a.examples
    cols = [("ref", "Референс"), ("t2i", "T2I"), ("ours", "Наш метод"), ("shift", "SHIFT"), ("teacher", "Учитель (I2I)")]
    if grids:
        cols = [c for c in cols if c[0] != "shift"]
    fig, ax = plt.subplots(len(ROWS), len(cols), figsize=(2.0 * len(cols) + 1.1, 2.0 * len(ROWS)))
    for i, (ref, idx, label) in enumerate(ROWS):
        for j, (kind, title) in enumerate(cols):
            A = ax[i, j]; A.axis("off")
            if kind == "ref":
                im = reference(src, ref, grids)
            else:
                im = from_grid(src, ref, idx, kind) if grids else from_examples(src, ref, idx, kind)
            if im is not None:
                if kind != "ref":
                    im = im.resize((512, 512), Image.LANCZOS)
                A.imshow(im)
            if i == 0:
                A.set_title(title, fontsize=11)
            if j == 0:
                A.text(-0.08, 0.5, label, transform=A.transAxes, ha="right", va="center", fontsize=9)
    plt.subplots_adjust(left=0.17, right=0.995, top=0.96, bottom=0.005, wspace=0.03, hspace=0.04)
    os.makedirs(a.out, exist_ok=True)
    name = "limits_grid.jpg" if grids else "limits_grid_shift.jpg"
    fig.savefig(os.path.join(a.out, name), dpi=170, pil_kwargs={"quality": 90})
    print(os.path.join(a.out, name))


if __name__ == "__main__":
    main()
