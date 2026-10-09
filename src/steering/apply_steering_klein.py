"""
Apply dual-stream mean-diff steering on FLUX.2 Klein 9B at text-to-image time.

Never pass a reference image here: token lengths must match the sliced
vectors from get_vector_klein.py (generated img tokens + text tokens only).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

STEERING_ROOT = Path(__file__).resolve().parents[2]
if str(STEERING_ROOT) not in sys.path:
    sys.path.insert(0, str(STEERING_ROOT))


def _import_klein_pipeline():
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline requires diffusers>=0.37.0. "
            "Install with: pip install -U 'diffusers>=0.37.0'"
        ) from exc
    return Flux2KleinPipeline


def load_prompts(prompt_path: str, num_prompts: Optional[int] = None):
    with open(prompt_path, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def find_vector_file(data_dir: str, vector_type: str) -> str:
    suffix_candidates = [f"_{vector_type}.pt", f"{vector_type}.pt"]
    skip_substrings = ("text_", "svm_models", "scores", "normals")
    candidates = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".pt")
        and not any(skip in name for skip in skip_substrings)
        and any(name.endswith(suffix) for suffix in suffix_candidates)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No vector file in '{data_dir}' ending with '{vector_type}.pt'. "
            "Run scripts/steering_calculate_klein.sh first."
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple vector candidates: {candidates}")
    return candidates[0]


def find_aux_file(data_dir: str, suffix: str) -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(data_dir, f"*{suffix}")))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: multiple {suffix} files, using {matches[0]}")
    return matches[0]


def _ensemble_for_branch(layer_models, branch: str):
    if layer_models is None:
        return None
    if isinstance(layer_models, dict):
        return layer_models.get(branch) or layer_models.get("txt")
    return layer_models


def cls_scale(mean_act: torch.Tensor, model, cls_min: float, task: str) -> float:
    """Scale steering by SVM P(class). Does not use calculate_cls_score (add-concept asserts there)."""
    vec = torch.nan_to_num(mean_act.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    if vec.dim() == 1:
        vec = vec.unsqueeze(0)
    if not torch.isfinite(vec).all() or vec.norm() < 1e-8:
        return 1.0
    vec = vec / vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    arr = np.nan_to_num(vec.numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
    if not np.isfinite(arr).all():
        return 1.0
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(arr)
            if task == "add concept":
                p_neg = float(proba[0][0])
                return min(cls_min, 1.0 / ((1.0 - p_neg) + 1e-8) - 1.0)
            p_pos = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            return min(cls_min, max(0.0, 1.0 / ((1.0 - p_pos) + 1e-8) - 1.0))
        if hasattr(model, "decision_function"):
            dist = float(model.decision_function(arr)[0])
            if task == "add concept":
                dist = -dist
            return min(cls_min, max(0.0, dist))
    except ValueError:
        return 1.0
    return 1.0


def split_vector(vector: dict) -> Tuple[dict, dict]:
    """Split dual-stream {step: {layer: {img, txt}}} into two dicts."""
    img_vec: Dict = {}
    txt_vec: Dict = {}
    is_dual = False
    for step in vector:
        if not isinstance(step, int):
            continue
        for layer_key in vector[step]:
            val = vector[step][layer_key]
            if isinstance(val, dict) and ("img" in val or "txt" in val):
                is_dual = True
            break
        break

    if not is_dual:
        return {}, vector

    for step in vector:
        if not isinstance(step, int):
            continue
        img_vec[step] = {}
        txt_vec[step] = {}
        for layer_key in vector[step]:
            val = vector[step][layer_key]
            if isinstance(val, dict):
                if "img" in val:
                    tensor = val["img"]
                    while tensor.dim() > 2:
                        tensor = tensor.squeeze(0)
                    img_vec[step][layer_key] = tensor
                if "txt" in val:
                    tensor = val["txt"]
                    while tensor.dim() > 2:
                        tensor = tensor.squeeze(0)
                    txt_vec[step][layer_key] = tensor
            else:
                txt_vec[step][layer_key] = val
    return img_vec, txt_vec


def apply_steering(
    activations: torch.Tensor,
    steering_vec: torch.Tensor,
    strength: float,
    task: str,
    score_val: float = 1.0,
) -> torch.Tensor:
    dtype = activations.dtype
    act_f32 = torch.nan_to_num(activations.float(), nan=0.0, posinf=0.0, neginf=0.0)
    orig_norm = torch.norm(act_f32, dim=-1, keepdim=True).clamp(min=1e-6)
    v_clean = torch.nan_to_num(steering_vec.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v_unit = v_clean / torch.norm(v_clean, dim=-1, keepdim=True).clamp(min=1e-6)
    score = torch.nan_to_num(
        torch.as_tensor(score_val, device=act_f32.device, dtype=act_f32.dtype),
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    adjustment = strength * v_unit * score
    if task == "remove":
        steered = act_f32 - adjustment
    else:
        steered = act_f32 + adjustment
    steered = torch.nan_to_num(steered, nan=0.0, posinf=0.0, neginf=0.0)
    steered_unit = steered / torch.norm(steered, dim=-1, keepdim=True).clamp(min=1e-6)
    return (steered_unit * orig_norm).to(dtype)


def apply_act_steering(pipe, args, norm_log=None):
    """Linear-AcT baseline: h' = (1 - lam) h + lam (omega * h + beta) per unit, on every token of
    the stream; lam = --strength_img (image stream) and --strength (text stream).
    Maps from src/steering/calculate_act_maps.py. No renormalisation (as in AcT)."""
    maps = torch.load(args.act_path, map_location="cpu", weights_only=False)["maps"]
    state = {"step": 0}
    cache = {}

    def _map(step, layer_idx, branch, ref):
        key = (step, layer_idx, branch)
        if key not in cache:
            m = maps.get(step, {}).get(f"layer_{layer_idx}", {}).get(branch)
            cache[key] = None if m is None else (m["omega"].to(ref.device, torch.float32),
                                                 m["beta"].to(ref.device, torch.float32))
        return cache[key]

    def _transport(h, m, lam):
        if m is None or lam == 0:
            return h
        omega, beta = m
        hf = h.float()
        return ((1 - lam) * hf + lam * (hf * omega + beta)).to(h.dtype)

    def hook_for(layer_idx):
        def hook(module, input, output):
            step = state["step"]
            if not (isinstance(output, tuple) and len(output) == 2):
                return output
            if norm_log is not None:
                for branch, tensor in (("txt", output[0]), ("img", output[1])):
                    key = f"{step}|{layer_idx}|{branch}"
                    total, count = norm_log.get(key, (0.0, 0))
                    norm_log[key] = (total + float(tensor.detach().float().norm(dim=-1).mean()), count + 1)
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output
            txt, img = output
            return (_transport(txt, _map(step, layer_idx, "txt", txt), args.strength),
                    _transport(img, _map(step, layer_idx, "img", img), args.strength_img))
        return hook

    blocks = pipe.transformer.transformer_blocks
    handles = [blocks[i].register_forward_hook(hook_for(i)) for i in range(min(args.num_layers, len(blocks)))]
    print(f"  Linear-AcT maps: {args.act_path} (lam_img={args.strength_img}, lam_txt={args.strength})")
    return state, lambda: [h.remove() for h in handles]


def apply_attention_steering(pipe, args, vector, norm_log=None):
    if getattr(args, "act_path", None):
        return apply_act_steering(pipe, args, norm_log=norm_log)
    """norm_log: optional dict filled with running sums of mean token L2 norms of the
    *unsteered* block outputs, keyed by "step|layer|stream" (used for the alpha / angle
    analysis in the paper: theta ~= arctan(alpha / ||h||))."""
    img_vector, txt_vector = split_vector(vector)
    print(f"  Steering branches: img={'YES' if img_vector else 'NO'}, txt={'YES' if txt_vector else 'NO'}")
    if img_vector:
        for step in img_vector:
            for layer_key in img_vector[step]:
                print(f"  img vector shape: {tuple(img_vector[step][layer_key].shape)}")
                break
            break
    if txt_vector:
        for step in txt_vector:
            for layer_key in txt_vector[step]:
                print(f"  txt vector shape: {tuple(txt_vector[step][layer_key].shape)}")
                break
            break

    state = {"step": 0}
    handles = []

    svm_path = find_aux_file(args.data_dir, "_svm_models.pt")
    scores_path = find_aux_file(args.data_dir, "_scores.pt")
    models = None
    scores_all = None
    if args.use_cls:
        if svm_path is None:
            raise FileNotFoundError(
                f"No *_svm_models.pt in {args.data_dir}. "
                "Run scripts/steering_calculate_klein.sh (SVM step) first."
            )
        models = torch.load(svm_path, map_location="cpu", weights_only=False)
        print(f"  use_cls models: {svm_path}")
        if scores_path:
            scores_all = torch.load(scores_path, map_location="cpu", weights_only=False)
            if torch.is_tensor(scores_all):
                scores_all = scores_all.numpy()
            print(f"  use_cls scores: {scores_path}")

    def _get_score_val(step: int, layer_idx: int, to_modify: torch.Tensor, branch: str) -> float:
        if not args.use_cls or models is None:
            return 1.0
        layer_key = f"layer_{layer_idx}"
        ensemble = _ensemble_for_branch(models.get(step, {}).get(layer_key), branch)
        if not ensemble:
            return 1.0
        current_signal = np.ones(len(ensemble), dtype=np.float32)
        if scores_all is not None:
            try:
                current_signal = np.asarray(scores_all[:, step, layer_idx], dtype=np.float32)
            except Exception:
                current_signal = np.ones(len(ensemble), dtype=np.float32)
        mean_act = torch.nan_to_num(
            to_modify.float(), nan=0.0, posinf=0.0, neginf=0.0
        ).mean(dim=tuple(range(to_modify.dim() - 1)))
        votes = []
        for i, model in enumerate(ensemble):
            signal = float(current_signal[i]) if i < len(current_signal) else 1.0
            if signal <= args.min_signal_threshold:
                votes.append(1.0)
                continue
            votes.append(cls_scale(mean_act, model, args.cls_min, args.task))
        return float(np.mean(votes)) if votes else 1.0

    def _prepare_vec(sv_raw, activations):
        if args.steering_type == "mean":
            steering_vec = sv_raw.mean(0, keepdim=True)
        else:
            steering_vec = sv_raw
        return steering_vec.to(activations.device, activations.dtype)

    def _get_layer_vectors(step: int, layer_idx: int):
        layer_key = f"layer_{layer_idx}"
        sv_txt = txt_vector.get(step, {}).get(layer_key)
        sv_img = img_vector.get(step, {}).get(layer_key)
        return sv_img, sv_txt

    def steering_hook(layer_idx: int):
        def hook(module, input, output):
            step = state["step"]
            if norm_log is not None and isinstance(output, tuple) and len(output) == 2:
                for branch, tensor in (("txt", output[0]), ("img", output[1])):
                    key = f"{step}|{layer_idx}|{branch}"
                    mean_norm = float(tensor.detach().float().norm(dim=-1).mean())
                    total, count = norm_log.get(key, (0.0, 0))
                    norm_log[key] = (total + mean_norm, count + 1)
            if args.block_steering != "all" and layer_idx not in args.block_steering:
                return output
            if args.t_steering != "all" and step not in args.t_steering:
                return output
            if not (isinstance(output, tuple) and len(output) == 2):
                return output

            sv_img, sv_txt = _get_layer_vectors(step, layer_idx)
            if sv_img is None and sv_txt is None:
                return output

            txt_hidden, img_hidden = output[0], output[1]
            txt_new, img_new = txt_hidden, img_hidden

            if sv_txt is not None and args.strength != 0:
                txt_new = apply_steering(
                    txt_hidden,
                    _prepare_vec(sv_txt, txt_hidden),
                    args.strength,
                    args.task,
                    score_val=_get_score_val(step, layer_idx, txt_hidden, "txt"),
                )
            if sv_img is not None and args.strength_img != 0:
                img_new = apply_steering(
                    img_hidden,
                    _prepare_vec(sv_img, img_hidden),
                    args.strength_img,
                    args.task,
                    score_val=_get_score_val(step, layer_idx, img_hidden, "img"),
                )
            return (txt_new, img_new)

        return hook

    blocks = pipe.transformer.transformer_blocks
    n_hook = min(args.num_layers, len(blocks))
    for layer_id in range(n_hook):
        handles.append(blocks[layer_id].register_forward_hook(steering_hook(layer_id)))
    return state, lambda: [handle.remove() for handle in handles]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply Klein-9B dual-stream steering at T2I time (no reference image)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="black-forest-labs/FLUX.2-klein-9B",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--act_path", type=str, default=None,
                        help="Linear-AcT maps (calculate_act_maps.py); strengths become lambda in [0, 1]")
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="prompts_collection/dataset_creation/dataset_prompts_style.txt",
    )
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--vector_type", type=str, default="diff")
    parser.add_argument("--task", type=str, default="add concept", choices=["add concept", "remove"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--strength", type=float, default=10.0, help="Text-branch strength")
    parser.add_argument("--strength_img", type=float, default=0.0, help="Image-branch strength")
    parser.add_argument(
        "--strength_txt",
        type=float,
        default=0.0,
        help="Ignored on Klein (no T5/CLIP text-encoder steering).",
    )
    parser.add_argument(
        "--steer_txt",
        action="store_true",
        help="Ignored on Klein (Qwen3 text-encoder steering is not implemented).",
    )
    parser.add_argument("--steering_type", type=str, default="separate", choices=["mean", "separate"])
    parser.add_argument(
        "--injection_point",
        type=str,
        default="block",
        choices=["block"],
        help="Klein apply hooks double-stream block residuals (same as extraction).",
    )
    parser.add_argument("--top_k_percent", type=float, default=0.95)
    parser.add_argument("--min_signal_threshold", type=float, default=0.05)
    parser.add_argument("--use_cls", action="store_true", help="Scale steering by SVM classifier score.")
    parser.add_argument("--cls_min", type=float, default=20.0)
    parser.add_argument("--cls_type", type=str, default="tanh")
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--block_steering", type=str, default="all")
    parser.add_argument("--t_steering", type=str, default="all")
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="experiments/klein_9b/style/generated_images")
    parser.add_argument(
        "--stats_path",
        type=str,
        default=None,
        help="Write JSON with per-image wall time (origin and steered) and mean token norms "
        "per (step, block, stream) of unsteered activations.",
    )
    parser.add_argument(
        "--save_origin",
        action="store_true",
        help="Also generate unsteered T2I images for visual comparison.",
    )
    return parser.parse_args()


def generate_one(pipe, prompt, args, device, seed, callback=None):
    kwargs = dict(
        prompt=prompt,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        width=args.width,
        height=args.height,
        generator=torch.Generator(device).manual_seed(int(seed)),
    )
    if callback is not None:
        kwargs["callback_on_step_end"] = callback
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    return pipe(**kwargs).images[0]


def main():
    args = parse_args()
    if args.block_steering != "all":
        args.block_steering = [int(x) for x in args.block_steering.split(",")]
    if args.t_steering != "all":
        args.t_steering = [int(x) for x in args.t_steering.split(",")]

    Flux2KleinPipeline = _import_klein_pipeline()
    print(f"Loading {args.model_name}...")
    pipe = Flux2KleinPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        pipe.to("cuda")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(
            f"data_dir not found: {args.data_dir}. Run scripts/steering_calculate_klein.sh first."
        )
    print(f"Vector dir contents: {sorted(os.listdir(args.data_dir))}")
    if args.act_path:
        vector = None  # Linear-AcT baseline uses its own maps
    else:
        vector_path = find_vector_file(args.data_dir, args.vector_type)
        vector = torch.load(vector_path, map_location="cpu", weights_only=False)
        print(f"Loaded: {vector_path}")
    if args.steer_txt or args.strength_txt:
        print("WARNING: --steer_txt / --strength_txt are ignored on Klein (no T5/CLIP encoder).")

    prompts = load_prompts(args.prompts_path, args.num_prompts)
    steered_dir = os.path.join(args.results_dir, "steered")
    origin_dir = os.path.join(args.results_dir, "origin")
    os.makedirs(steered_dir, exist_ok=True)
    if args.save_origin:
        os.makedirs(origin_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    stats = {"origin_seconds": [], "steered_seconds": []}
    if args.save_origin:
        for idx, prompt in enumerate(tqdm(prompts, desc="Origin T2I")):
            sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
            origin_path = os.path.join(origin_dir, f"{idx:02d}_{sanitized}_origin.png")
            if os.path.exists(origin_path):
                continue
            _sync()
            t0 = time.perf_counter()
            image = generate_one(pipe, prompt, args, device, int(args.seed) + idx)
            _sync()
            stats["origin_seconds"].append(time.perf_counter() - t0)
            image.save(origin_path)
        print(f"Saved origin images to {origin_dir}")

    norm_log = {} if args.stats_path else None
    hook_state, remove_hooks = apply_attention_steering(pipe, args, vector, norm_log=norm_log)

    def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        hook_state["step"] = step_index + 1
        return callback_kwargs

    try:
        for idx, prompt in enumerate(tqdm(prompts, desc="Steered T2I")):
            sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
            suffix = f"s_{args.strength}_simg_{args.strength_img}_v_{args.vector_type}"
            out_path = os.path.join(steered_dir, f"{idx:02d}_{sanitized}_{suffix}.png")
            if os.path.exists(out_path):
                continue
            hook_state["step"] = 0
            _sync()
            t0 = time.perf_counter()
            image = generate_one(
                pipe, prompt, args, device, int(args.seed) + idx, callback=_on_step_end
            )
            _sync()
            stats["steered_seconds"].append(time.perf_counter() - t0)
            image.save(out_path)
    finally:
        remove_hooks()

    if args.stats_path:
        stats["mean_token_norm"] = {
            key: total / max(count, 1) for key, (total, count) in sorted(norm_log.items())
        }
        stats["config"] = {
            "strength": args.strength,
            "strength_img": args.strength_img,
            "block_steering": args.block_steering,
            "t_steering": args.t_steering,
            "steering_type": args.steering_type,
            "use_cls": args.use_cls,
            "act_path": args.act_path,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)), exist_ok=True)
        with open(args.stats_path, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"Saved stats to {args.stats_path}")

    print(f"Saved steered images to {steered_dir}")
    print(
        f"Check: compare {steered_dir} vs "
        f"{origin_dir if args.save_origin else 'unsteered T2I (rerun with --save_origin)'}."
    )


if __name__ == "__main__":
    main()
