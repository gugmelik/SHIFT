"""External training-free baselines named in the review that need at most one image.

  ipadapter  IP-Adapter (Ye et al., 2023) for FLUX.1-dev, XLabs-AI/flux-ip-adapter, one reference
             image per generation. There is no IP-Adapter for FLUX.2 Klein, so the base model differs;
             content preservation is measured against FLUX.1-dev T2I with the same prompt and seed
             (the same pipeline with adapter scale 0).
  casteer    CASteer (Gaintseva et al., ICLR 2026) steering vectors on SDXL-base-1.0: per cross-attention
             layer, patch-averaged attn2 outputs of positive (prompt + text style tag, the tag of the
             text-SHIFT baseline) minus negative (prompt) runs with a shared seed, averaged over the
             50 extraction prompts and all denoising steps (CASteer Eq. 1, 2, 7; their SDXL vectors are
             step-independent). The public CASteer code implements erasure/translation only, so the
             vector is *added* with the same norm-preserving rule as in the paper (Eq. 11), with a
             relative strength: ca' = ||ca|| (ca + s ||ca|| v^) / ||ca + s ||ca|| v^||, i.e. a rotation
             by about arctan(s), applied to all attn2 layers and steps (both CFG branches).

Usage
  python src/baselines/external_baselines.py casteer_extract --prompts TRAIN --tag ", as ..." --out V.pt
  python src/baselines/external_baselines.py generate --method casteer --vectors V.pt \\
      --prompts VAL --strengths 0.05 0.1 0.2 0.4 --seed 42 --out DIR
  python src/baselines/external_baselines.py generate --method ipadapter --reference REF.jpg \\
      --prompts VAL --strengths 0.5 0.8 1.0 --seed 42 --out DIR
Layout: DIR/origin/NN_<prompt>.png (strength 0) and DIR/s_<strength>/NN_<prompt>.png; timing.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_prompts(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]


def fname(idx, prompt):
    return f"{idx:02d}_{prompt.replace(' ', '_').replace('/', '').replace(',', '')[:50]}.png"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ------------------------------------------------------------------ SDXL / CASteer
def sdxl(args):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(args.sdxl_id, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def attn2_modules(pipe):
    return {n: m for n, m in pipe.unet.named_modules() if n.endswith("attn2")}


def sdxl_call(pipe, prompt, seed, args, cb=None):
    kw = dict(prompt=prompt, num_inference_steps=args.steps or 30, guidance_scale=args.guidance or 5.0,
              height=args.size, width=args.size, generator=torch.Generator(device).manual_seed(seed))
    if cb:
        kw.update(callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    return pipe(**kw).images[0]


def casteer_extract(args):
    pipe = sdxl(args)
    mods = attn2_modules(pipe)
    prompts = load_prompts(args.prompts)
    sums = {"pos": {n: None for n in mods}, "neg": {n: None for n in mods}}
    counts = {"pos": 0, "neg": 0}
    current = {"kind": None}

    def mk(name):
        def hook(m, inp, out):
            o = out[out.shape[0] // 2:] if out.shape[0] > 1 else out  # conditional CFG branch
            v = o.float().mean(1).sum(0).cpu()                         # patch average, summed over batch
            k = current["kind"]
            sums[k][name] = v if sums[k][name] is None else sums[k][name] + v
        return hook

    handles = [m.register_forward_hook(mk(n)) for n, m in mods.items()]
    for i, p in enumerate(tqdm(prompts, desc="CASteer pairs")):
        for kind, text in (("pos", p + args.tag), ("neg", p)):
            current["kind"] = kind
            sdxl_call(pipe, text, 42000 + i, args)   # shared seed within the pair
            counts[kind] += 1
    for h in handles:
        h.remove()
    vec = {n: (sums["pos"][n] - sums["neg"][n]) / counts["pos"] for n in mods}  # same #steps in both
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": vec, "tag": args.tag, "n_pairs": len(prompts), "model": args.sdxl_id}, args.out)
    print(f"saved {args.out} ({len(vec)} attn2 layers)")


def casteer_hooks(pipe, vectors, state):
    handles = []
    for n, m in attn2_modules(pipe).items():
        v = vectors[n].to(device, torch.float32)
        v = v / v.norm().clamp(min=1e-8)

        def hook(mod, inp, out, v=v):
            s = state["s"]
            if s == 0:
                return out
            o = out.float()
            nrm = o.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            st = o + s * nrm * v
            return (st / st.norm(dim=-1, keepdim=True).clamp(min=1e-6) * nrm).to(out.dtype)
        handles.append(m.register_forward_hook(hook))
    return handles


# ------------------------------------------------------------------ FLUX.1-dev / IP-Adapter
def flux_ip(args):
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(args.flux_id, torch_dtype=torch.bfloat16)
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.load_ip_adapter(args.ip_id, weight_name="ip_adapter.safetensors",
                         image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def flux_call(pipe, prompt, seed, args, ref):
    kw = dict(prompt=prompt, num_inference_steps=args.steps or 28, guidance_scale=args.guidance or 3.5,
              height=args.size, width=args.size, ip_adapter_image=ref,
              generator=torch.Generator(device).manual_seed(seed))
    if args.true_cfg > 1:
        kw.update(true_cfg_scale=args.true_cfg, negative_prompt="")
    return pipe(**kw).images[0]


# ------------------------------------------------------------------ generate
def generate(args):
    prompts = load_prompts(args.prompts)
    out = Path(args.out)
    strengths = [0.0] + [s for s in args.strengths if s != 0]
    timing = {}
    if args.method == "casteer":
        pipe = sdxl(args)
        state = {"s": 0.0}
        handles = casteer_hooks(pipe, torch.load(args.vectors, map_location="cpu")["vectors"], state)
        def run(p, seed, s):
            state["s"] = s
            return sdxl_call(pipe, p, seed, args)
    else:
        pipe = flux_ip(args)
        ref = Image.open(args.reference).convert("RGB")
        def run(p, seed, s):
            pipe.set_ip_adapter_scale(s)
            return flux_call(pipe, p, seed, args, ref)
    for s in strengths:
        d = out / ("origin" if s == 0 else f"s_{s:g}")
        d.mkdir(parents=True, exist_ok=True)
        times = []
        for i, p in enumerate(tqdm(prompts, desc=f"{args.method} s={s:g}")):
            f = d / fname(i, p)
            if f.exists():
                continue
            sync(); t0 = time.perf_counter()
            img = run(p, args.seed + i, s)
            sync(); times.append(time.perf_counter() - t0)
            img.save(f)
        timing[f"{s:g}"] = times
    old = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
    for k, v in timing.items():
        old[k] = old.get(k, []) + v
    (out / "timing.json").write_text(json.dumps(old, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("casteer_extract", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--prompts", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--size", type=int, default=1024)
        p.add_argument("--steps", type=int, default=None, help="default: 30 (SDXL) / 28 (FLUX.1-dev)")
        p.add_argument("--guidance", type=float, default=None, help="default: 5.0 (SDXL) / 3.5 (FLUX.1-dev)")
        p.add_argument("--sdxl_id", default="stabilityai/stable-diffusion-xl-base-1.0")
        p.add_argument("--flux_id", default="black-forest-labs/FLUX.1-dev")
        p.add_argument("--ip_id", default="XLabs-AI/flux-ip-adapter")
        p.add_argument("--true_cfg", type=float, default=1.0, help=">1 doubles the cost (diffusers example: 4.0)")
        p.add_argument("--offload", action="store_true", help="CPU offload for GPUs < 40 GB")
    sub.choices["casteer_extract"].add_argument("--tag", required=True)
    g = sub.choices["generate"]
    g.add_argument("--method", choices=["casteer", "ipadapter"], required=True)
    g.add_argument("--vectors", default=None)
    g.add_argument("--reference", default=None)
    g.add_argument("--strengths", type=float, nargs="+", required=True)
    g.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    casteer_extract(a) if a.cmd == "casteer_extract" else generate(a)


if __name__ == "__main__":
    main()
