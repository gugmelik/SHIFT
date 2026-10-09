# Revision 2 — full run from scratch

## GPU
* Needs >= 48 GB VRAM: FLUX.2 Klein 9B + Qwen3-8B encoder ~35 GB in bf16; FLUX.1-dev + T5 ~33 GB.
* Cheapest per result on RunPod: L40S 48 GB (~16 h, ~$18) or A40 / RTX A6000 (~32 h, ~$19). H100: ~11 h, ~$32.
* 4 references (Monet, Van Gogh, Ciurlionis, Pirosmani) + 2 twins. Several GPUs: one reference per GPU (step 3b).
* Use ONE GPU type for all runs: Table 2 (timing) is measured during the run; report that GPU in the paper.
* Disk: ~300 GB free (models ~90 GB, activations/vectors ~45 GB, images ~60 GB). RAM >= 64 GB.

## 1. Environment (once)
    git clone -b flux-klein https://github.com/gugmelik/SHIFT && cd SHIFT
    conda create -n shift python=3.11 -y && conda activate shift
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt "diffusers>=0.37" lpips piq "torchmetrics[image]" scipy matplotlib \
                ftfy regex huggingface_hub sentencepiece protobuf
    git clone https://github.com/learn2phoenix/CSD third_party/CSD

## 2. Model access (once)
Accept the licences on huggingface.co for: black-forest-labs/FLUX.2-klein-9B, black-forest-labs/FLUX.1-dev,
facebook/dinov3-vitl16-pretrain-lvd1689m. Then:
    huggingface-cli login
    huggingface-cli download tomg-group-umd/CSD-ViT-L pytorch_model.bin --local-dir weights/csd
    python scripts/download_open_references.py              # 4 references (2 AIC CC0 + 2 Wikimedia Commons PD)
    # look at data/reference_images/open/*.jpg; if a Commons hit is wrong, pin it:
    #   python scripts/download_open_references.py --only pirosmani_giraffe --commons_file pirosmani_giraffe=File:<exact name>.jpg
    python scripts/download_open_references.py --set twins  # 2 same-style references

## 3a. One GPU: everything
    tmux new -s rev2
    bash scripts/run_revision2_all.sh
Restartable: finished work is skipped. Progress in logs/revision2_*.log.

## 3b. Several GPUs: split references, then finish on one
    R=data/reference_images/open
    CUDA_VISIBLE_DEVICES=0 SKIP_FINAL=1 REFS="$R/monet_water_lilies.jpg" bash scripts/run_revision2_all.sh &
    CUDA_VISIBLE_DEVICES=1 SKIP_FINAL=1 REFS="$R/vangogh_self_portrait.jpg" bash scripts/run_revision2_all.sh &
    CUDA_VISIBLE_DEVICES=2 SKIP_FINAL=1 REFS="$R/ciurlionis_sonata_sun.jpg" bash scripts/run_revision2_all.sh &
    CUDA_VISIBLE_DEVICES=3 SKIP_FINAL=1 REFS="$R/pirosmani_giraffe.jpg" bash scripts/run_revision2_all.sh &
    wait
    bash scripts/run_revision2_all.sh      # nothing is regenerated; runs twins + analysis + zip
(Different machines: copy experiments/klein_9b/paper/<ref>/ folders into one repo before the last command.)

## What runs (order handled by the script)
1. extract (pos/neg activations, vectors) · textbase (text-SHIFT) · sweep (alpha 2..12, val) ·
   nameprobe (prompt + 'in the style of <artist>': does the model know the style by name?) · ablation ·
   blocks (only_k / drop_k / first_k for all 8 blocks) · test (ours + text SHIFT, 3 seeds; paired I2I teacher)
2. score (CSD, Gram, DINOv2/3, LPIPS, DISTS, CLIP, FID/KID)
3. test_rule (alpha from the paper's rule) · baselines: Linear-AcT (Klein), CASteer (SDXL), IP-Adapter (FLUX.1-dev)
4. score · twins (Monet Water Lily Pond, Van Gogh Bedroom) · CPU analysis

## Send back
    paper_results_revision2.zip   (created in the repository root at the end)
    logs/revision2_*.log          only if something failed
