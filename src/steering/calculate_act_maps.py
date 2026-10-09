"""Linear Activation Transport (Linear-AcT; Rodriguez et al., ICLR 2025, arXiv:2410.23054) maps
from the activations already recorded by get_vector_klein.py.

Same contrastive pairs as the mean-difference vector: source = negative pass (prompt only),
target = positive pass (prompt + reference, reference tokens dropped). As in AcT for diffusion
models, every unit is described by its spatially average-pooled value per sample (the "pooled"
(N, D) entry), and an independent affine map is fitted per unit by least squares on the
*sorted* source and target samples (1-D optimal transport for the squared cost):

    a_(i), b_(i) sorted;  omega = sum (a_(i)-m_a)(b_(i)-m_b) / sum (a_(i)-m_a)^2,  beta = m_b - omega*m_a

At inference (apply_steering_klein.py --act_path) every token h of the stream becomes
    h' = (1 - lam) h + lam (omega * h + beta),   lam in [0, 1]
with unbounded support (Q_inf, used by AcT for induction). Maps are fitted per grid node
(step), double-stream block and stream. Simplification w.r.t. the paper: all layers are fitted
simultaneously from one set of recorded activations, not causally layer by layer.

    python src/steering/calculate_act_maps.py --data_dir <exp>/ref/data_vectors --out <exp>/act/act_maps.pt
"""
import argparse
import glob
import os

import torch


def fit(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a, b = a.double().sort(0).values, b.double().sort(0).values  # (N, D) each, sorted per unit
    ma, mb = a.mean(0), b.mean(0)
    ac, bc = a - ma, b - mb
    omega = (ac * bc).sum(0) / (ac.pow(2).sum(0) + eps)
    beta = mb - omega * ma
    return omega.float(), beta.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pos_p = sorted(glob.glob(os.path.join(a.data_dir, "*_pos_block.pt")))[0]
    neg_p = sorted(glob.glob(os.path.join(a.data_dir, "*_neg_block.pt")))[0]
    pos = torch.load(pos_p, map_location="cpu", weights_only=False)["pooled"]
    neg = torch.load(neg_p, map_location="cpu", weights_only=False)["pooled"]
    maps, summary = {}, []
    for step in sorted(pos):
        maps[int(step)] = {}
        for layer in sorted(pos[step], key=lambda s: int(s.split("_")[1])):
            maps[int(step)][layer] = {}
            for s in ("img", "txt"):
                if s in pos[step][layer] and s in neg[step][layer]:
                    om, be = fit(neg[step][layer][s], pos[step][layer][s])
                    maps[int(step)][layer][s] = {"omega": om, "beta": be}
                    summary.append(f"step {step} {layer} {s}: omega mean {om.mean():.3f} "
                                   f"[{om.min():.2f}, {om.max():.2f}], |beta| mean {be.abs().mean():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"maps": maps, "method": "linear_act_sorted_ls", "n_samples": int(pos[0]["layer_0"]["img"].shape[0])},
               a.out)
    print("\n".join(summary[:8] + ["..."]))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
