"""
Apply dual-stream mean-diff steering on FLUX.2 Klein 9B at text-to-image time.

Never pass a reference image here: token lengths must match the sliced
vectors from get_vector_klein.py (generated img tokens + text tokens only).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import torch
from tqdm import tqdm


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
    candidates = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith(".pt")
        and "text_" not in name
        and any(name.endswith(suffix) for suffix in suffix_candidates)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No vector file in '{data_dir}' ending with '{vector_type}.pt'"
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple vector candidates: {candidates}")
    return candidates[0]


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
) -> torch.Tensor:
    dtype = activations.dtype
    act_f32 = activations.float()
    orig_norm = torch.norm(act_f32, dim=-1, keepdim=True) + 1e-6
    v_unit = steering_vec.float() / (torch.norm(steering_vec.float(), dim=-1, keepdim=True) + 1e-6)
    adjustment = strength * v_unit.to(activations.dtype)
    if task == "remove":
        steered = activations - adjustment
    else:
        steered = activations + adjustment
    steered_unit = steered.float() / (torch.norm(steered.float(), dim=-1, keepdim=True) + 1e-6)
    return (steered_unit * orig_norm).to(dtype)


def apply_attention_steering(pipe, args, vector):
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
                )
            if sv_img is not None and args.strength_img != 0:
                img_new = apply_steering(
                    img_hidden,
                    _prepare_vec(sv_img, img_hidden),
                    args.strength_img,
                    args.task,
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
    parser.add_argument("--strength_img", type=float, default=10.0, help="Image-branch strength")
    parser.add_argument("--steering_type", type=str, default="mean", choices=["mean", "separate"])
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--block_steering", type=str, default="all")
    parser.add_argument("--t_steering", type=str, default="all")
    parser.add_argument("--inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="experiments/klein_9b/style/generated_images")
    return parser.parse_args()


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

    vector_path = find_vector_file(args.data_dir, args.vector_type)
    vector = torch.load(vector_path, map_location="cpu", weights_only=False)
    print(f"Loaded: {vector_path}")

    prompts = load_prompts(args.prompts_path, args.num_prompts)
    steered_dir = os.path.join(args.results_dir, "steered")
    os.makedirs(steered_dir, exist_ok=True)

    hook_state, remove_hooks = apply_attention_steering(pipe, args, vector)

    def _on_step_end(_pipe, step_index, _timestep, callback_kwargs):
        hook_state["step"] = step_index + 1
        return callback_kwargs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        for idx, prompt in enumerate(tqdm(prompts, desc="Steered T2I")):
            sanitized = prompt.replace(" ", "_").replace("/", "").replace(",", "")[:50]
            suffix = f"s_{args.strength}_simg_{args.strength_img}_v_{args.vector_type}"
            out_path = os.path.join(steered_dir, f"{idx:02d}_{sanitized}_{suffix}.png")
            if os.path.exists(out_path):
                continue

            hook_state["step"] = 0
            generator = torch.Generator(device).manual_seed(int(args.seed) + idx)
            images = pipe(
                prompt,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                width=args.width,
                height=args.height,
                generator=generator,
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images
            images[0].save(out_path)
    finally:
        remove_hooks()

    print(f"Saved steered images to {steered_dir}")


if __name__ == "__main__":
    main()
