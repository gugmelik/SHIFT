"""
FLUX.2 Klein 9B activation extraction for reference-image style steering.

Positive pass: same text prompt + one shared reference image (Klein I2I).
Negative pass: same text prompt, no image (Klein T2I).

Reference tokens are concatenated after generated image tokens:
  hidden_states = [gen_img | ref_img]
and are sliced off before saving so pos/neg dumps share token lengths.

Output format (compatible with calculate_steering_vectors.py):
  {
    step: {
      "layer_N": {
        "img": tensor(n_samples, n_img_tokens, hidden_dim),
        "txt": tensor(n_samples, n_txt_tokens, hidden_dim),
      },
      ...
    },
    ...
  }
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_REF_DIR = "data/reference_images"


def _import_klein_pipeline():
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline requires diffusers>=0.37.0. "
            "Install with: pip install -U 'diffusers>=0.37.0'"
        ) from exc
    return Flux2KleinPipeline


def load_prompts(prompt_path: str, num_prompts: Optional[int] = None) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_path}")
    return prompts


def resolve_reference_image(path_or_dir: str) -> Path:
    path = Path(path_or_dir)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(
            f"Reference image path not found: {path}. "
            f"Add one image under {DEFAULT_REF_DIR}/"
        )
    images = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(
            f"No image found in {path}. Drop one .png/.jpg/.webp into that folder."
        )
    if len(images) > 1:
        print(f"Multiple reference images in {path}; using {images[0].name}")
    return images[0]


def count_generated_img_tokens(pipe, height: int, width: int) -> int:
    """Packed generated-image token count used by Flux2KleinPipeline.prepare_latents."""
    vae_scale = pipe.vae_scale_factor
    packed_h = 2 * (int(height) // (vae_scale * 2))
    packed_w = 2 * (int(width) // (vae_scale * 2))
    return (packed_h // 2) * (packed_w // 2)


def generator_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class DualStreamHookManager:
    """Collects Klein double-stream block outputs and drops reference tokens."""

    def __init__(
        self,
        num_blocks: int,
        save_timesteps: int,
        stream: str = "both",
        n_gen_tokens: Optional[int] = None,
    ):
        self.num_blocks = num_blocks
        self.save_timesteps = save_timesteps
        self.stream = stream
        self.n_gen_tokens = n_gen_tokens
        self.data: Dict = {}
        self.reset_state()

    def reset_state(self):
        self.current_step = 0
        self.hooks_fired_this_step = 0

    def _advance_step_counter(self):
        self.hooks_fired_this_step += 1
        if self.hooks_fired_this_step >= self.num_blocks:
            self.current_step += 1
            self.hooks_fired_this_step = 0

    def _should_save(self, block_idx: int) -> bool:
        return block_idx < self.num_blocks and self.current_step < self.save_timesteps

    def _wants_branch(self, branch: str) -> bool:
        if self.stream == "both":
            return branch in ("img", "txt")
        return branch == self.stream

    def _store(self, step: int, block_idx: int, branch: str, tensor: torch.Tensor):
        if not self._wants_branch(branch):
            return
        if step not in self.data:
            self.data[step] = {}
        layer_key = f"layer_{block_idx}"
        if layer_key not in self.data[step]:
            self.data[step][layer_key] = {"img": [], "txt": []}
        self.data[step][layer_key][branch].append(tensor.detach().cpu())

    def _slice_generated_img(self, img: torch.Tensor) -> torch.Tensor:
        if self.n_gen_tokens is None:
            return img
        if img.shape[1] < self.n_gen_tokens:
            raise ValueError(
                f"Image stream has {img.shape[1]} tokens, expected at least "
                f"{self.n_gen_tokens} generated tokens"
            )
        if img.shape[1] > self.n_gen_tokens:
            return img[:, : self.n_gen_tokens]
        return img

    def make_block_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            if self._should_save(block_idx):
                txt, img = output[0], output[1]
                img = self._slice_generated_img(img)
                self._store(self.current_step, block_idx, "txt", txt)
                self._store(self.current_step, block_idx, "img", img)
            self._advance_step_counter()

        return hook_fn

    def aggregate(self) -> Dict:
        result = {}
        for step, layers in self.data.items():
            result[step] = {}
            for layer_key, branches in layers.items():
                result[step][layer_key] = {}
                for branch, tensor_list in branches.items():
                    if tensor_list:
                        result[step][layer_key][branch] = torch.stack(tensor_list)
        return result


def register_hooks(pipe, manager: DualStreamHookManager, num_blocks: int) -> List:
    blocks = pipe.transformer.transformer_blocks
    handles = []
    for idx in range(min(num_blocks, len(blocks))):
        handles.append(blocks[idx].register_forward_hook(manager.make_block_hook(idx)))
    return handles


def run_extraction(pipe, prompts, args, reference_image=None) -> Tuple[Dict, List]:
    manager = DualStreamHookManager(
        num_blocks=args.num_layers,
        save_timesteps=args.save_timesteps,
        stream=args.token_stream,
        n_gen_tokens=args.n_gen_tokens,
    )
    handles = register_hooks(pipe, manager, args.num_layers)
    all_images = []
    device = generator_device()

    try:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Extracting"):
            batch = prompts[i : i + args.batch_size]
            manager.reset_state()
            generators = [
                torch.Generator(device).manual_seed(42000 + i * 10 + j)
                for j in range(len(batch))
            ]
            call_kwargs = dict(
                prompt=batch,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.gs,
                height=args.height,
                width=args.width,
                generator=generators,
            )
            if reference_image is not None:
                call_kwargs["image"] = reference_image
            result = pipe(**call_kwargs)
            all_images.extend(result.images)
    finally:
        for handle in handles:
            handle.remove()

    vectors = manager.aggregate()
    for step in sorted(k for k in vectors if isinstance(k, int)):
        for layer_key in sorted(vectors[step].keys()):
            shapes = {branch: tuple(tensor.shape) for branch, tensor in vectors[step][layer_key].items()}
            print(f"  step {step} {layer_key}: {shapes}")
        break
    return vectors, all_images


def save_grid(images, path: str):
    if not images:
        return
    to_tensor = ToTensor()
    nrow = max(1, int(len(images) ** 0.5))
    tensors = [to_tensor(img) for img in images]
    grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    array = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FLUX.2 Klein dual-stream activation extraction (reference-image style)"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts_collection/dataset_creation/dataset_prompts_style.txt",
    )
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--exp_type", type=str, default="style_ref")
    parser.add_argument(
        "--reference_image",
        type=str,
        default=DEFAULT_REF_DIR,
        help="Path to one image, or a folder containing one image",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="black-forest-labs/FLUX.2-klein-9B",
    )
    parser.add_argument("--gs", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument(
        "--token_stream",
        type=str,
        default="both",
        choices=["txt", "img", "both"],
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=8,
        help="Double-stream blocks to hook (Klein 9B has 8)",
    )
    parser.add_argument("--save_timesteps", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="experiments/klein_9b/style/data_vectors")
    parser.add_argument("--save_image_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    Flux2KleinPipeline = _import_klein_pipeline()

    prompts = load_prompts(args.prompt_path, args.num_prompts)
    ref_path = resolve_reference_image(args.reference_image)
    reference_image = Image.open(ref_path).convert("RGB")
    print(f"Reference image: {ref_path}")
    print(f"Prompts: {len(prompts)} (same list for pos and neg)")

    print(f"Loading {args.model_name}...")
    pipe = Flux2KleinPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        pipe.to("cuda")

    n_double = len(pipe.transformer.transformer_blocks)
    args.num_layers = min(args.num_layers, n_double)
    args.n_gen_tokens = count_generated_img_tokens(pipe, args.height, args.width)
    print(
        f"Double-stream blocks: {args.num_layers}/{n_double}, "
        f"generated img tokens: {args.n_gen_tokens}, "
        f"timesteps saved: {args.save_timesteps}"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    n_prompts = len(prompts)
    file_template = (
        f"{args.exp_type}_gs_{args.gs}_prompts_{n_prompts}_{{}}_block.pt"
    )
    metadata = {
        "img_latent_hw": (
            2 * (args.height // (pipe.vae_scale_factor * 2)) // 2,
            2 * (args.width // (pipe.vae_scale_factor * 2)) // 2,
        ),
        "img_resolution": (args.height, args.width),
        "n_gen_tokens": args.n_gen_tokens,
        "reference_image": str(ref_path),
        "model_name": args.model_name,
    }

    print("\nRunning Positive Pass (prompt + reference image)...")
    pos_vecs, pos_imgs = run_extraction(pipe, prompts, args, reference_image=reference_image)
    pos_vecs.update(metadata)
    pos_path = os.path.join(args.save_dir, file_template.format("pos"))
    torch.save(pos_vecs, pos_path)
    print(f"Saved {pos_path}")
    del pos_vecs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nRunning Negative Pass (prompt only)...")
    neg_vecs, neg_imgs = run_extraction(pipe, prompts, args, reference_image=None)
    neg_vecs.update(metadata)
    neg_path = os.path.join(args.save_dir, file_template.format("neg"))
    torch.save(neg_vecs, neg_path)
    print(f"Saved {neg_path}")
    del neg_vecs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.save_image_dir:
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_grid(pos_imgs, os.path.join(args.save_image_dir, f"positive_{args.exp_type}_{n_prompts}_grid.png"))
        save_grid(neg_imgs, os.path.join(args.save_image_dir, f"negative_{args.exp_type}_{n_prompts}_grid.png"))
        print(f"Saved image grids to {args.save_image_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
