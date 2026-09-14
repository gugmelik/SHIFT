"""DINOv2 metrics for Klein reference-image style steering.

Pairs images by leading prompt index (00_, 01_, ...) so Klein filenames work.

  content DINO          origin T2I  <->  steered T2I
  style DINO (steered)  style photo <->  steered T2I
  style DINO (I2I)      style photo <->  positive I2I teacher

Run from the repository root:

  python metrics/dino_klein.py picasso_style
  python metrics/dino_klein.py --all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parents[1]
STYLES_ROOT = REPO_ROOT / "experiments" / "klein_9b" / "style"
STEERED_RE = re.compile(
    r"^(?P<idx>\d+)_.+_s_(?P<strength>[-+]?\d+(?:\.\d+)?)_simg_"
    r"(?P<strength_img>[-+]?\d+(?:\.\d+)?)_v_(?P<vector_type>[^.]+)$",
    re.IGNORECASE,
)
INDEX_RE = re.compile(r"^(\d+)_")

device = "cuda" if torch.cuda.is_available() else "cpu"
dinov2 = None
dinov2_transform = T.Compose(
    [
        T.Resize(224, interpolation=Image.Resampling.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_dinov2():
    global dinov2
    if dinov2 is not None:
        return dinov2
    print("Initializing DINOv2 ViT-L/14...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)
    dinov2.eval()
    return dinov2


def get_dinov2_embedding(img: Image.Image) -> torch.Tensor:
    model = load_dinov2()
    with torch.no_grad():
        img_t = dinov2_transform(img).unsqueeze(0).to(device)
        feats = model(img_t).flatten()
        return feats / feats.norm()


def cosine(feat1: torch.Tensor, feat2: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        feat1.unsqueeze(0), feat2.unsqueeze(0)
    ).item()


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def index_of(path: Path) -> Optional[int]:
    match = INDEX_RE.match(path.stem)
    return int(match.group(1)) if match else None


def map_by_index(directory: Path) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for path in list_images(directory):
        idx = index_of(path)
        if idx is None:
            continue
        if idx in mapping:
            print(f"WARNING: multiple files for index {idx:02d} in {directory}; using {path.name}")
        mapping[idx] = path
    return mapping


def steered_runs(steered_dir: Path) -> Dict[str, Dict[int, Path]]:
    runs: Dict[str, Dict[int, Path]] = {}
    for path in list_images(steered_dir):
        match = STEERED_RE.match(path.stem)
        if match:
            run_id = (
                f"s_{match.group('strength')}_simg_{match.group('strength_img')}"
                f"_v_{match.group('vector_type')}"
            )
            idx = int(match.group("idx"))
        else:
            idx = index_of(path)
            if idx is None:
                continue
            run_id = "steered"
        runs.setdefault(run_id, {})[idx] = path
    return runs


def find_reference(exp_dir: Path) -> Optional[Path]:
    for path in sorted(exp_dir.iterdir()):
        if path.is_file() and path.stem.lower() == "reference" and path.suffix.lower() in IMAGE_EXTS:
            return path
    return None


def embed_file(path: Path, cache: Dict[Path, torch.Tensor]) -> torch.Tensor:
    if path not in cache:
        cache[path] = get_dinov2_embedding(Image.open(path).convert("RGB"))
    return cache[path]


def fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if value is not None else "  n/a"


def evaluate_style(exp_dir: Path) -> dict:
    origin_dir = exp_dir / "generated_images" / "origin"
    steered_dir = exp_dir / "generated_images" / "steered"
    i2i_dir = exp_dir / "dataset_images" / "i2i"
    reference = find_reference(exp_dir)

    origin = map_by_index(origin_dir)
    i2i = map_by_index(i2i_dir)
    runs = steered_runs(steered_dir)
    cache: Dict[Path, torch.Tensor] = {}

    if reference is None:
        print(f"WARNING: no reference.* in {exp_dir}")
        ref_feat = None
    else:
        ref_feat = embed_file(reference, cache)

    style_dino_i2i_scores: List[float] = []
    if ref_feat is not None:
        for idx in sorted(i2i):
            style_dino_i2i_scores.append(cosine(ref_feat, embed_file(i2i[idx], cache)))
    style_dino_i2i = mean_or_none(style_dino_i2i_scores)

    run_metrics = {}
    for run_id, steered in sorted(runs.items()):
        content_scores: List[float] = []
        style_steered_scores: List[float] = []
        desc = f"{exp_dir.name}/{run_id}"
        for idx in tqdm(sorted(steered), desc=desc):
            steered_feat = embed_file(steered[idx], cache)
            if idx in origin:
                content_scores.append(cosine(embed_file(origin[idx], cache), steered_feat))
            if ref_feat is not None:
                style_steered_scores.append(cosine(ref_feat, steered_feat))
        run_metrics[run_id] = {
            "n_steered": len(steered),
            "n_content": len(content_scores),
            "n_style_steered": len(style_steered_scores),
            "content_dino": mean_or_none(content_scores),
            "style_dino_steered": mean_or_none(style_steered_scores),
        }

    result = {
        "exp_dir": str(exp_dir),
        "reference": str(reference) if reference else None,
        "n_origin": len(origin),
        "n_i2i": len(i2i),
        "style_dino_i2i": style_dino_i2i,
        "n_style_i2i": len(style_dino_i2i_scores),
        "runs": run_metrics,
        "missing": {
            "origin": not bool(origin),
            "steered": not bool(runs),
            "i2i": not bool(i2i),
            "reference": reference is None,
        },
    }
    return result


def print_style_report(result: dict) -> None:
    print(f"\n=== {Path(result['exp_dir']).name} ===")
    print(f"dir: {result['exp_dir']}")
    print(f"reference: {result['reference']}")
    missing = [name for name, is_missing in result["missing"].items() if is_missing]
    if missing:
        print(f"missing: {', '.join(missing)}")
    print(
        f"style DINO I2I (teacher): {fmt(result['style_dino_i2i'])}  "
        f"(n={result['n_style_i2i']})"
    )
    if not result["runs"]:
        print("no steered images")
        return
    print(f"{'run':<36} {'n':>4}  {'content DINO':>12}  {'style DINO steered':>18}")
    for run_id, metrics in result["runs"].items():
        print(
            f"{run_id:<36} {metrics['n_steered']:>4}  "
            f"{fmt(metrics['content_dino']):>12}  "
            f"{fmt(metrics['style_dino_steered']):>18}"
        )


def as_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_exp_dir(style: str, styles_root: Path) -> Path:
    candidate = Path(style)
    if candidate.is_dir() and (
        (candidate / "generated_images").exists()
        or (candidate / "dataset_images").exists()
    ):
        return candidate.resolve()
    nested = styles_root / style
    if nested.is_dir():
        return nested.resolve()
    raise FileNotFoundError(
        f"No experiment folder for '{style}'. Expected {nested} "
        f"(run extract/apply first)."
    )


def discover_styles(styles_root: Path) -> List[Path]:
    if not styles_root.is_dir():
        return []
    dirs = []
    for path in sorted(styles_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "generated_images").is_dir() or (path / "dataset_images").is_dir():
            dirs.append(path)
    return dirs


def save_result(result: dict, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "metrics_dinov2_klein.json"
    pt_path = save_dir / "metrics_dinov2_klein.pt"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save(result, pt_path)
    print(f"saved {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 content + style scores for Klein style steering."
    )
    parser.add_argument(
        "style",
        nargs="?",
        help="Experiment stem (picasso_style) or path to experiments/klein_9b/style/<stem>.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every style folder under --styles_root.",
    )
    parser.add_argument(
        "--styles_root",
        type=str,
        default=str(STYLES_ROOT),
        help="Root that contains one folder per style.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Explicit experiment directory (overrides the style positional).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Where to write metrics_dinov2_klein.json (default: <exp>/metrics).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    styles_root = as_path(args.styles_root)
    if args.all:
        exp_dirs = discover_styles(styles_root)
        if not exp_dirs:
            print(f"No style experiment folders under {styles_root}")
            sys.exit(1)
    elif args.exp_dir:
        exp_dirs = [as_path(args.exp_dir)]
    elif args.style:
        exp_dirs = [resolve_exp_dir(args.style, styles_root)]
    else:
        print("Pass a style stem, --exp_dir, or --all")
        sys.exit(1)

    combined = {}
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            print(f"ERROR: not a directory: {exp_dir}")
            sys.exit(1)
        result = evaluate_style(exp_dir)
        print_style_report(result)
        save_dir = Path(args.save_dir) if args.save_dir else exp_dir / "metrics"
        save_result(result, save_dir)
        combined[exp_dir.name] = result

    if len(combined) > 1:
        print("\n=== all styles ===")
        print(
            f"{'style':<22} {'run':<32} {'content':>8}  {'style steered':>13}  {'style I2I':>9}"
        )
        for style, result in combined.items():
            i2i = fmt(result["style_dino_i2i"])
            if not result["runs"]:
                print(f"{style:<22} {'(no steered)':<32} {'n/a':>8}  {'n/a':>13}  {i2i:>9}")
                continue
            for run_id, metrics in result["runs"].items():
                print(
                    f"{style:<22} {run_id:<32} "
                    f"{fmt(metrics['content_dino']):>8}  "
                    f"{fmt(metrics['style_dino_steered']):>13}  "
                    f"{i2i:>9}"
                )
        summary_dir = styles_root / "metrics"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "metrics_dinov2_klein_all.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(combined, handle, indent=2)
        print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
