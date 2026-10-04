"""Extended evaluation for Klein reference-image style steering (paper revision protocol).

Two sub-commands.

score   -- per-image metrics for one generated set, saved as JSON:
    style:     CSD cosine to the reference (optional, needs the CSD repo),
               VGG-19 Gram-matrix distance to the reference (Gatys et al.)
    reference: DINOv2 / DINOv3 cosine to the reference
    content:   DINOv2 cosine, LPIPS, DISTS to the unsteered origin image
    prompt:    CLIP text-image cosine
    set-level: FID and KID between the generated set and a target set (e.g. I2I teacher)
  Every per-image metric is reported as mean and a 95% bootstrap CI over prompts.

compare -- paired Wilcoxon signed-rank tests between several scored runs (paired by
  image index), Holm-corrected, written as JSON + printed table.

Images are paired by their leading index ("00_", "01_", ...), as in dino_klein.py.

Examples (run from the repository root):

  python metrics/eval_klein_extended.py score \
      --gen_dir experiments/klein_9b/style/picasso_style/generated_images/steered \
      --origin_dir experiments/klein_9b/style/picasso_style/generated_images/origin \
      --reference data/reference_images/picasso_style.jpg \
      --prompts prompts_collection/klein_style/test_prompts.txt \
      --fid_target_dir experiments/klein_9b/style/picasso_style/dataset_images/i2i \
      --out results/picasso_ours.json

  python metrics/eval_klein_extended.py compare \
      --runs results/picasso_ours.json results/picasso_shift_text.json results/picasso_t2i.json \
      --out results/picasso_compare.json

Optional dependencies: lpips, piq (DISTS), torchmetrics[image] (FID/KID),
transformers (CLIP, DINOv3), scipy (Wilcoxon). Missing ones are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
INDEX_RE = re.compile(r"^(\d+)_")
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# ----------------------------------------------------------------------------- io

def map_by_index(directory: Optional[str]) -> Dict[int, Path]:
    if not directory:
        return {}
    root = Path(directory)
    mapping: Dict[int, Path] = {}
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        match = INDEX_RE.match(path.stem)
        if match:
            mapping[int(match.group(1))] = path
    return mapping


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


# ------------------------------------------------------------------- statistics

def bootstrap_ci(values: List[float], n_boot: int = 10000, seed: int = 0) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n": int(arr.size),
    }


def holm(pvalues: List[float]) -> List[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------- metrics

class Dinov2:
    name = "dinov2"

    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
        self.tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                             T.CenterCrop(224), T.ToTensor(), IMAGENET_NORM])

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        feat = self.model(self.tf(img).unsqueeze(0).to(device)).flatten()
        return feat / feat.norm()


class Dinov3:
    """DINOv3 via transformers. The checkpoint is gated on the Hub: accept the license first."""

    name = "dinov3"

    def __init__(self, model_id: str):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        out = self.model(**inputs)
        feat = out.pooler_output.flatten() if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0].flatten()
        return feat / feat.norm()


class Clip:
    def __init__(self, model_id: str = "openai/clip-vit-large-patch14"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def score(self, img: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=img, return_tensors="pt", padding=True,
                                truncation=True).to(device)
        img_f = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_f = self.model.get_text_features(input_ids=inputs["input_ids"],
                                             attention_mask=inputs["attention_mask"])
        return float(F.cosine_similarity(img_f, txt_f).item())


class GramStyle:
    """Gatys et al. style distance: mean squared difference of normalised Gram matrices
    of VGG-19 relu1_1..relu5_1 features. Lower = closer style; content-agnostic by construction."""

    LAYERS = {1: "relu1_1", 6: "relu2_1", 11: "relu3_1", 20: "relu4_1", 29: "relu5_1"}

    def __init__(self, size: int = 512):
        from torchvision.models import VGG19_Weights, vgg19

        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30].to(device).eval()
        self.tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), IMAGENET_NORM])

    @torch.no_grad()
    def grams(self, img: Image.Image) -> List[torch.Tensor]:
        x = self.tf(img).unsqueeze(0).to(device)
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.LAYERS:
                _, c, h, w = x.shape
                f = x.view(c, h * w)
                out.append(f @ f.t() / (c * h * w))
        return out

    def distance(self, g1: List[torch.Tensor], g2: List[torch.Tensor]) -> float:
        return float(sum(F.mse_loss(a, b).item() for a, b in zip(g1, g2)) / len(g1))


class Csd:
    """CSD style descriptor (Somepalli et al., 2024). Requires https://github.com/learn2phoenix/CSD
    on PYTHONPATH and its checkpoint. Adjust the import if the repository layout changes."""

    def __init__(self, ckpt: str):
        from CSD.model import CSD_CLIP  # type: ignore
        from CSD.utils import convert_state_dict  # type: ignore

        self.model = CSD_CLIP("vit_large", "default")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model.load_state_dict(convert_state_dict(state["model_state_dict"]), strict=False)
        self.model = self.model.to(device).eval()
        self.tf = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        _, _, style = self.model(self.tf(img).unsqueeze(0).to(device))
        style = style.flatten()
        return style / style.norm()


def try_build(builder: Callable, label: str):
    try:
        return builder()
    except Exception as exc:  # noqa: BLE001 - optional metric, report and continue
        print(f"WARNING: {label} disabled ({type(exc).__name__}: {exc})")
        return None


def to_tensor_01(img: Image.Image, size: int = 512) -> torch.Tensor:
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])(img).unsqueeze(0).to(device)


# ------------------------------------------------------------------------ score

def cmd_score(args) -> None:
    gen = map_by_index(args.gen_dir)
    origin = map_by_index(args.origin_dir)
    prompts = load_prompts(args.prompts)
    if not gen:
        sys.exit(f"No indexed images in {args.gen_dir}")
    ref_img = load_rgb(Path(args.reference)) if args.reference else None

    dinov2 = try_build(Dinov2, "DINOv2")
    dinov3 = try_build(lambda: Dinov3(args.dinov3_id), "DINOv3") if args.dinov3_id else None
    clip = try_build(lambda: Clip(args.clip_id), "CLIP") if prompts else None
    gram = try_build(GramStyle, "Gram/VGG-19") if ref_img is not None else None
    csd = try_build(lambda: Csd(args.csd_ckpt), "CSD") if args.csd_ckpt and ref_img is not None else None
    lpips_fn = try_build(lambda: __import__("lpips").LPIPS(net="alex").to(device), "LPIPS") if origin else None
    dists_fn = try_build(lambda: __import__("piq").DISTS().to(device), "DISTS") if origin else None

    ref_cache = {}
    if ref_img is not None:
        if dinov2: ref_cache["dinov2"] = dinov2.embed(ref_img)
        if dinov3: ref_cache["dinov3"] = dinov3.embed(ref_img)
        if gram: ref_cache["gram"] = gram.grams(ref_img)
        if csd: ref_cache["csd"] = csd.embed(ref_img)

    per_image: Dict[str, Dict[int, float]] = {}

    def put(metric: str, idx: int, value: float):
        per_image.setdefault(metric, {})[idx] = float(value)

    for idx in tqdm(sorted(gen), desc="score"):
        img = load_rgb(gen[idx])
        if dinov2:
            e = dinov2.embed(img)
            if "dinov2" in ref_cache:
                put("dinov2_to_reference", idx, torch.dot(e, ref_cache["dinov2"]))
            if idx in origin:
                put("dinov2_to_origin", idx, torch.dot(e, dinov2.embed(load_rgb(origin[idx]))))
        if dinov3 and "dinov3" in ref_cache:
            put("dinov3_to_reference", idx, torch.dot(dinov3.embed(img), ref_cache["dinov3"]))
        if gram:
            put("gram_to_reference", idx, gram.distance(gram.grams(img), ref_cache["gram"]))
        if csd:
            put("csd_to_reference", idx, torch.dot(csd.embed(img), ref_cache["csd"]))
        if clip and idx < len(prompts):
            put("clip_text", idx, clip.score(img, prompts[idx]))
        if idx in origin and (lpips_fn or dists_fn):
            a, b = to_tensor_01(img), to_tensor_01(load_rgb(origin[idx]))
            with torch.no_grad():
                if lpips_fn:
                    put("lpips_to_origin", idx, lpips_fn(a * 2 - 1, b * 2 - 1).item())
                if dists_fn:
                    put("dists_to_origin", idx, dists_fn(a, b).item())

    summary = {metric: bootstrap_ci(list(vals.values()), args.n_boot) for metric, vals in per_image.items()}

    set_level = {}
    if args.fid_target_dir:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchmetrics.image.kid import KernelInceptionDistance

            target = map_by_index(args.fid_target_dir)
            n_min = min(len(gen), len(target))
            fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
            kid = KernelInceptionDistance(subset_size=max(2, min(50, n_min // 2)), normalize=True).to(device)
            for path in target.values():
                x = to_tensor_01(load_rgb(path), 299)
                fid.update(x, real=True); kid.update(x, real=True)
            for path in gen.values():
                x = to_tensor_01(load_rgb(path), 299)
                fid.update(x, real=False); kid.update(x, real=False)
            kid_mean, kid_std = kid.compute()
            set_level = {"fid": float(fid.compute()), "kid_mean": float(kid_mean), "kid_std": float(kid_std),
                         "n_gen": len(gen), "n_target": len(target),
                         "note": "FID is strongly biased for small sets; prefer KID below ~1000 images."}
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: FID/KID disabled ({type(exc).__name__}: {exc})")

    result = {
        "gen_dir": args.gen_dir, "origin_dir": args.origin_dir, "reference": args.reference,
        "prompts": args.prompts, "summary": summary, "set_level": set_level,
        "per_image": {m: {str(k): v for k, v in vals.items()} for m, vals in per_image.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\n{'metric':<22} {'mean':>9}  95% CI")
    for metric, stats in sorted(summary.items()):
        ci = stats["ci95"]
        print(f"{metric:<22} {stats['mean']:>9.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  n={stats['n']}")
    if set_level:
        print(f"FID {set_level['fid']:.2f}  KID {set_level['kid_mean']:.4f}±{set_level['kid_std']:.4f}")
    print(f"saved {args.out}")


# ---------------------------------------------------------------------- compare

def cmd_compare(args) -> None:
    from scipy.stats import wilcoxon

    runs = [json.load(open(path, encoding="utf-8")) for path in args.runs]
    names = args.names or [Path(p).stem for p in args.runs]
    base, base_name = runs[0], names[0]
    rows, pvals = [], []
    for run, name in zip(runs[1:], names[1:]):
        for metric, base_vals in base["per_image"].items():
            other = run["per_image"].get(metric)
            if not other:
                continue
            common = sorted(set(base_vals) & set(other), key=int)
            if len(common) < 5:
                continue
            a = np.array([base_vals[k] for k in common]); b = np.array([other[k] for k in common])
            if np.allclose(a, b):
                p = 1.0
            else:
                p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            rows.append({"baseline": base_name, "other": name, "metric": metric, "n": len(common),
                         "mean_diff": float((a - b).mean()), "p": p})
            pvals.append(p)
    for row, p_adj in zip(rows, holm(pvals) if pvals else []):
        row["p_holm"] = p_adj
        row["significant_0.05"] = p_adj < 0.05
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"{'vs':<18} {'metric':<22} {'n':>4} {'mean diff':>10} {'p_holm':>9}")
    for row in rows:
        star = "*" if row["significant_0.05"] else ""
        print(f"{row['other']:<18} {row['metric']:<22} {row['n']:>4} {row['mean_diff']:>10.4f} "
              f"{row['p_holm']:>9.4f}{star}")
    print(f"saved {args.out}  (mean diff = {base_name} - other)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--gen_dir", required=True)
    s.add_argument("--origin_dir", default=None)
    s.add_argument("--reference", default=None)
    s.add_argument("--prompts", default=None, help="Prompt file in the same index order as images")
    s.add_argument("--fid_target_dir", default=None)
    s.add_argument("--dinov3_id", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                   help="Empty string disables DINOv3")
    s.add_argument("--clip_id", default="openai/clip-vit-large-patch14")
    s.add_argument("--csd_ckpt", default=None)
    s.add_argument("--n_boot", type=int, default=10000)
    s.add_argument("--out", required=True)

    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True, help="First run is the method under test")
    c.add_argument("--names", nargs="*", default=None)
    c.add_argument("--out", required=True)

    args = parser.parse_args()
    cmd_score(args) if args.cmd == "score" else cmd_compare(args)


if __name__ == "__main__":
    main()
