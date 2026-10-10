"""Pre-registered alpha selection rule of the paper (Section 3.4), applied to validation scores.

Rule: among the swept alpha_img values, take the one with the smallest Gram distance to the
reference, subject to mean DINOv2 similarity to the unsteered T2I image >= tau (default 0.85).
With --metric csd the style score is CSD similarity (larger is better) instead of Gram.

    python scripts/select_alpha.py experiments/klein_9b/paper/monet_water_lilies/scores
    python scripts/select_alpha.py <scores_dir> --verbose      # table of all alphas

Prints only the chosen alpha on stdout (used by paper_klein_experiments.sh, stage test_rule).
"""
import argparse
import json
import sys
from pathlib import Path


def mean(d, metric):
    vals = list(d["per_image"].get(metric, {}).values())
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir")
    ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--metric", choices=["gram", "csd"], default="gram")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fallback", action="store_true",
                    help="If no strength meets tau, take the most content-preserving one (max DINOv2-to-origin) "
                         "instead of failing; a WARNING goes to stderr and to --note")
    ap.add_argument("--note", default=None, help="Write the selection outcome (rule / fallback) to this file")
    ap.add_argument("--prefix", default="val_alpha_", help="score files <prefix><strength>.json (baselines: val_act_ ...)")
    a = ap.parse_args()
    rows = []
    for f in Path(a.scores_dir).glob(f"{a.prefix}*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        try:
            alpha = float(f.stem[len(a.prefix):])
        except ValueError:
            continue
        rows.append((alpha, mean(d, "gram_to_reference"), mean(d, "csd_to_reference"),
                     mean(d, "dinov2_to_origin")))
    if not rows:
        sys.exit(f"no {a.prefix}*.json in {a.scores_dir}")
    rows.sort()
    if a.verbose:
        for r in rows:
            print(f"alpha={r[0]:g} gram={r[1]} csd={r[2]} dinov2_origin={r[3]}", file=sys.stderr)
    ok = [r for r in rows if r[3] is not None and r[3] >= a.tau]
    if not ok:
        if not a.fallback:
            sys.exit(f"no alpha satisfies DINOv2-to-origin >= {a.tau}")
        best = max((r for r in rows if r[3] is not None), key=lambda r: r[3])
        msg = (f"fallback: no strength meets DINOv2-to-origin >= {a.tau} "
               f"(max {best[3]:.3f} at {best[0]:g}); chose the most content-preserving one")
        print(f"WARNING {a.scores_dir} {a.prefix}: {msg}", file=sys.stderr)
        if a.note:
            Path(a.note).write_text(f"{best[0]:g}\t{msg}\n", encoding="utf-8")
        print(f"{best[0]:g}")
        return
    if a.metric == "gram":
        best = min(ok, key=lambda r: r[1])
    else:
        if any(r[2] is None for r in ok):
            sys.exit("CSD scores missing; run the score stage with CSD_CKPT set")
        best = max(ok, key=lambda r: r[2])
    if a.note:
        Path(a.note).write_text(f"{best[0]:g}\trule (DINOv2-to-origin {best[3]:.3f} >= {a.tau})\n", encoding="utf-8")
    print(f"{best[0]:g}")


if __name__ == "__main__":
    main()
