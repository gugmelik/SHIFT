# paper_results/ — input for the next paper revision

Fill this folder after the GPU runs. `article_tex/paper.tex` is then updated from these files.
Do not edit the `.tex` by hand before that: every red `\TODO` in the paper maps to a file below.

## How to fill it

```bash
python scripts/download_open_references.py
STAGES="extract textbase sweep score" bash scripts/paper_klein_experiments.sh
ALPHA=<chosen> STAGES="ablation test score" bash scripts/paper_klein_experiments.sh
python scripts/collect_paper_results.py --alpha <chosen>
```

If you ran the experiments elsewhere, either:
- copy `experiments/klein_9b/paper/` back and run the collector, or
- paste the files by hand using the layout below.

## Expected layout

```text
paper_results/
  summary.json                 all numbers (generated)
  tables.md                    readable tables (generated)
  MISSING.md                   inputs the collector could not find (generated)
  sources/
    SOURCES.md                 image credits (from data/reference_images/open/)
    references.bib             artwork citations
    sources.json
  figs/
    <stem>_reference.jpg       reference image as used
    <stem>_test_grid.jpg       T2I | ours | teacher, selected test prompts
    <stem>_val_alpha_sweep.jpg T2I | a=2 | a=4 | ... on validation prompts
  notes.md                     ← write by hand (template below)
```

## Where each file goes in the paper

| Paper element | Source in this folder |
| --- | --- |
| Table 1 (token norms) | `summary.json → references.<stem>.token_norms` |
| Table 2 (timing) | `summary.json → references.<stem>.timing` |
| Table 4 (DINO, first reference) | `summary.json → references.<stem>.test.{ours,teacher}` (`dinov2_to_*`) |
| Table 5 (method comparison) | `summary.json → references.<stem>.test`, `wilcoxon` |
| Table 6 (ablations) | `summary.json → references.<stem>.val_ablation` |
| α choice (Sec. 3.4) | `summary.json → references.<stem>.val_sweep` |
| Figures 1–3 | `figs/` (captions use `sources/SOURCES.md`) |
| Bibliography (artworks) | `sources/references.bib` |

## notes.md template (fill by hand)

```markdown
GPU / driver / torch / diffusers versions:
Chosen ALPHA and why (sweep table row):
Reference used as "reference 1" in the paper (main figure):
Reference used for the alpha-sweep figure:
Prompt indices that should appear in the qualitative figure:
Anything that failed or was skipped (e.g. CSD, DINOv3, LoRA, AcT):
Observations you want reflected in the text:
```
