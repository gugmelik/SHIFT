# Reference-image style steering for FLUX.2 Klein

This branch extends **SHIFT** (activation steering for flow transformers) to
**reference-image style steering** on the unified model
[FLUX.2 Klein 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).
The style is defined by one picture instead of a text tag.
The picture is used only once, to extract a steering vector.
Generation afterwards is plain text-to-image, with no reference image.

The original SHIFT code for FLUX.1 / SD3.5 is unchanged; see [`README.md`](README.md).

---

## 1. Method in one page

**Reference method: SHIFT** [1]. SHIFT runs contrastive prompt pairs (positive = subject + style tag,
negative = subject only) through an MM-DiT. It records block activations per denoising step and
takes the difference of means (or an SVM normal) as a steering vector. That vector is added at
inference time.

**What changes here** is only how the positive class is defined. FLUX.2 Klein accepts the same prompt
with or without a reference image, so each prompt is run twice from the same initial noise:

| Class | Klein call | Meaning |
| --- | --- | --- |
| positive | `prompt` + shared reference image `I` | teacher: what Klein does when it sees the reference |
| negative | `prompt` only | student: ordinary text-to-image |

Pipeline (all three steps are SHIFT's, with Klein-specific token handling):

1. **Extract** (`src/steering/get_vector_klein.py`). Forward hooks are placed on all 8 double-stream
   blocks and record the outputs of both streams at all 4 denoising steps. On the positive pass the image
   stream is `[generated tokens | reference tokens]`. The reference tokens (750 for a 405×493 image)
   are **cropped**, so that positive and negative tensors have the same length (4096 image tokens at
   1024², 512 text tokens). The reference still influences the generated tokens through joint attention;
   only its own token states are discarded. Only prompt means are stored.
2. **Calculate** (`src/steering/calculate_steering_vectors.py --method diff`). For every step t,
   block ℓ, stream s and token position j:
   `v[t,ℓ,s][j] = mean_k(h⁺) − mean_k(h⁻)`.
   The vector is therefore *per token* (a 64×64 field for the image stream).
3. **Apply** (`src/steering/apply_steering_klein.py`). Text-to-image only. Each token is updated as
   `h ← ‖h‖ · (h + α v̂) / ‖h + α v̂‖`.
   This is a rotation of h towards v̂ by an angle θ with `tan θ ≈ α / ‖h‖`.
   - Separate strengths are used for the text stream (`--strength`) and the image stream
     (`--strength_img`), because the two streams have different token norms.
   - At very large α, every token collapses onto v̂, and the output turns into the reference's content.

Four-step time grid used by the distilled model: t ≈ (1.000, 0.967, 0.908, 0.767, 0). Vectors are
tied to this grid. If you change the resolution or the number of steps, extract the vector again.

---

## 2. Setup

Requirements:
- a CUDA GPU that runs FLUX.2 Klein 9B in bf16;
- Python ≥ 3.10.

```bash
git clone <this-repo> && cd SHIFT
# install a CUDA build of torch first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt               # includes diffusers>=0.37 (Flux2KleinPipeline)
pip install lpips piq scipy "torchmetrics[image]"   # extended evaluation
huggingface-cli login                          # see the access notes below
```

Access and licences:
- **FLUX.2 Klein 9B** is distributed under the *FLUX Non-Commercial License*. Accept it on the model
  page before the first download.
- **DINOv3** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is gated on the Hub. Accept its licence, or pass
  `--dinov3_id ""` to skip it.
- **CSD** (style descriptor, optional): clone <https://github.com/learn2phoenix/CSD>, put it on
  `PYTHONPATH` and pass `--csd_ckpt path/to/checkpoint.pth`. Without it the CSD column is skipped.
  The VGG Gram distance is always computed and is the fallback style metric.

---

## 3. Reference images (free to use)

Only openly licensed images are used. The downloader fetches public-domain artworks from the
**Art Institute of Chicago Open Access** collection (CC0 1.0). It refuses any image that AIC does not
mark as public domain, and writes the credits and citations next to the images:

```bash
python scripts/download_open_references.py          # → data/reference_images/open/
```

| File | Artwork (AIC id) | Style |
| --- | --- | --- |
| `hartley_movement.jpg` | Marsden Hartley, *Movement*, 1913 (65916) | cubist-influenced abstraction |
| `kandinsky_improvisation30.jpg` | Vasily Kandinsky, *Improvisation No. 30 (Cannons)*, 1913 (8991) | expressionist abstraction |
| `vangogh_self_portrait.jpg` | Vincent van Gogh, *Self-Portrait*, 1887 (80607) | post-impressionism |
| `seurat_grande_jatte.jpg` | Georges Seurat, *A Sunday on La Grande Jatte — 1884*, 1884–86 (27992) | pointillism |
| `monet_water_lilies.jpg` | Claude Monet, *Water Lilies*, 1906 (16568) | impressionism |
| `hokusai_great_wave.jpg` | Katsushika Hokusai, *Under the Wave off Kanagawa*, 1830/33 (24645) | ukiyo-e woodblock |
| `degas_jockey.jpg` | Edgar Degas, *Jockey*, 1866–68 (7157) | drawing |

Files written next to the images:

| File | Contents |
| --- | --- |
| `SOURCES.md` | credit table: artist, title, date, medium, AIC reference number, URL, licence |
| `references.bib` | one `@misc` entry per artwork, for the paper |
| `sources.json` | the same metadata in machine-readable form |
| `style_tags.tsv` | text tags for the text-only SHIFT baseline |

Credit line for figure captions:
*Artist, Title, date. The Art Institute of Chicago, ref. no. — CC0 (AIC Open Access).*
CC0 does not legally require attribution, but cite the artworks anyway.

The older images in `data/reference_images/` (`picasso_style.jpg`, `sketch.jpg`,
`luca-illustration.jpg`) are **not** openly licensed. Do not use them in new figures or releases.

---

## 4. Quick run on one reference

```bash
bash scripts/run_klein_style.sh data/reference_images/open/hokusai_great_wave.jpg    # extract + calculate
bash scripts/apply_steering_klein.sh hokusai_great_wave                               # apply (T2I)
bash scripts/eval_dino_klein.sh hokusai_great_wave                                    # DINOv2 scores
```

Outputs go to `experiments/klein_9b/style/<stem>/`.

> ⚠️ `scripts/apply_steering_klein.sh` keeps its historical defaults: `--strength 45` on the text
> stream and `--use_cls`. The paper configuration uses image-stream steering only, with no classifier.
> For paper numbers use the full protocol below.

---

## 5. Full experimental protocol (paper revision)

`scripts/paper_klein_experiments.sh` runs everything requested in the review. It loops over every
image in `data/reference_images/open/`.

```bash
python scripts/download_open_references.py

# 1) extraction, text baseline, alpha sweep on the validation prompts
STAGES="extract textbase sweep score" bash scripts/paper_klein_experiments.sh

# 2) pick alpha from the sweep (rule below), then ablations + test runs + scoring
ALPHA=6 STAGES="ablation test score" bash scripts/paper_klein_experiments.sh

# 3) gather everything for the paper
python scripts/collect_paper_results.py --alpha 6
```

Prompts are split into disjoint sets in `prompts_collection/klein_style/`:

| Set | File | Prompts | Used for |
| --- | --- | --- | --- |
| train | `train_prompts.txt` | 50 | vector extraction |
| val | `val_prompts.txt` | 25 | α sweep and ablations |
| test | `test_prompts.txt` | 100 | final numbers, 3 seeds |

| Stage | What it does | Main outputs (`experiments/klein_9b/paper/<stem>/`) |
| --- | --- | --- |
| `extract` | pos/neg activations on train prompts, mean-diff + SVM files, timing | `ref/vectors/`, `ref/extract_timing.json` |
| `textbase` | text-only SHIFT on Klein (`--pos_suffix` from `style_tags.tsv`) | `text/vectors/` |
| `sweep` | α_img ∈ {2,4,6,7,8,10,12} on val, with token norms and timing | `val/alpha_*/{origin,steered,stats.json}` |
| `ablation` | blocks 1–4 / 5–8, steps, token-pooled vector, + text stream, + SVM | `val/ablation/*` |
| `test` | ours and text-SHIFT on test with 3 seeds; I2I teacher images | `test/{ours,shift_text}/seed_*`, `test/teacher/i2i` |
| `score` | extended metrics for val and test, Wilcoxon comparison | `scores/*.json` |

Useful environment variables:

| Variable | Meaning |
| --- | --- |
| `REFS` | list of reference image paths (overrides the folder loop) |
| `REF_DIR` | folder of reference images |
| `ALPHA`, `ALPHAS` | test strength, and sweep grid |
| `SEEDS` | seeds for the test runs |
| `TRAIN`, `VAL`, `TEST` | prompt files |
| `MODEL_NAME`, `CUDA_VISIBLE_DEVICES`, `PYTHON` | model id and runtime settings |

**α selection rule** (fix it before looking at test numbers): on the val set, pick the α with the highest
style score (CSD, or the Gram distance if CSD is unavailable), subject to:
- DINOv2 similarity to the unsteered image ≥ 0.85;
- CLIP text score at most 0.02 below T2I.

---

## 6. Evaluation

`metrics/eval_klein_extended.py` has two sub-commands.

```bash
# per-image metrics + 95% bootstrap CI for one generated set
python metrics/eval_klein_extended.py score \
    --gen_dir   <run>/steered  --origin_dir <run>/origin \
    --reference data/reference_images/open/hokusai_great_wave.jpg \
    --prompts   prompts_collection/klein_style/test_prompts.txt \
    --fid_target_dir <exp>/test/teacher/i2i  --out scores/ours.json

# paired Wilcoxon signed-rank vs other runs (first run = method under test), Holm-corrected
python metrics/eval_klein_extended.py compare --runs scores/ours.json scores/t2i.json scores/teacher.json \
    --out scores/compare.json
```

| Metric | Measures | Reference |
| --- | --- | --- |
| CSD cosine to reference | style similarity, trained to ignore content | Somepalli et al., 2024 [5] |
| VGG-19 Gram distance to reference | texture/colour statistics | Gatys et al., 2016 [6] |
| DINOv2 / DINOv3 cosine to reference | overall visual similarity (content-sensitive) | [7], [8] |
| DINOv2 cosine, LPIPS, DISTS to the unsteered image | content preservation | [7], [9], [10] |
| CLIP text–image cosine | prompt alignment | [11] |
| FID / KID vs teacher set | distribution distance; prefer KID for small sets | [12] |

Images are paired by their leading index (`00_…`, `01_…`), so the steered, origin and teacher folders
must come from the same prompt file.

---

## 7. Results folder for the paper

`scripts/collect_paper_results.py` reads `experiments/klein_9b/paper/` and writes **`paper_results/`**.
That folder is everything needed to update the paper text; see [`paper_results/README.md`](paper_results/README.md).

---

## 8. Repository map (Klein part)

```text
scripts/download_open_references.py   CC0 reference images + citations
scripts/paper_klein_experiments.sh    full protocol (extract → score)
scripts/collect_paper_results.py      experiments → paper_results/
scripts/run_klein_style.sh            quick extract + calculate for one image
scripts/apply_steering_klein.sh       quick apply (legacy defaults, see warning)
src/steering/get_vector_klein.py      extraction (+ --pos_suffix text baseline, --i2i_only teacher)
src/steering/apply_steering_klein.py  apply (+ --stats_path: token norms, timing)
src/steering/calculate_steering_vectors.py  shared SHIFT calculator
metrics/eval_klein_extended.py        extended metrics and significance tests
metrics/dino_klein.py                 original DINOv2-only evaluation
prompts_collection/klein_style/       train / val / test prompts
data/reference_images/open/           CC0 reference images (after download)
article_tex/                          paper sources
```

---

## References

1. N. Konovalova, A. Kuznetsov, A. Alanov. *SHIFT: Steering hidden intermediates in flow transformers.* arXiv:2604.09213, 2026.
2. Black Forest Labs. *FLUX.2 [klein] 9B.* https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
3. P. Rodriguez et al. *Controlling language and diffusion models by transporting activations.* arXiv:2410.23054, 2024.
4. P. Esser et al. *Scaling rectified flow transformers for high-resolution image synthesis.* ICML 2024.
5. G. Somepalli et al. *Measuring style similarity in diffusion models.* ECCV 2024, arXiv:2404.01292.
6. L. A. Gatys, A. S. Ecker, M. Bethge. *Image style transfer using convolutional neural networks.* CVPR 2016.
7. M. Oquab et al. *DINOv2: learning robust visual features without supervision.* TMLR 2024.
8. O. Siméoni et al. *DINOv3.* arXiv:2508.10104, 2025.
9. R. Zhang et al. *The unreasonable effectiveness of deep features as a perceptual metric.* CVPR 2018.
10. K. Ding et al. *Image quality assessment: unifying structure and texture similarity.* IEEE TPAMI 2022.
11. A. Radford et al. *Learning transferable visual models from natural language supervision.* ICML 2021.
12. M. Heusel et al. *GANs trained by a two time-scale update rule converge to a local Nash equilibrium.* NeurIPS 2017.
13. The Art Institute of Chicago. *Open Access images (CC0)* and *Public API.* https://api.artic.edu/docs/

If you use this code, please cite SHIFT [1] and the reference artworks as listed in
`data/reference_images/open/references.bib`.
