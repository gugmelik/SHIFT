# Revision 3 — GPU run of the NEW experiments only

What is run (nothing from revision 2 is repeated — no sweep, ablations, VAL block analysis, name probe,
Linear-AcT / CASteer / IP-Adapter, twins, rule runs):

| stage         | what                                                                      | L40S, 4 refs |
|---------------|---------------------------------------------------------------------------|--------------|
| `angle`       | angle-matched control: only_k and all blocks at θ* = 2.6° / 3.0° (VAL), 2.6° (TEST) | ~6 h |
| `blocks_test` | held-out per-block check, TEST, 3 seeds: only_1, drop_1, drop_8, 1–4, 5–8, steps 3–4 | ~4 h |
| `shift_sweep` | SHIFT with the selection rule (VAL sweep, TEST 3 seeds)                    | ~1.5 h |
| `score`, `examples`, `analysis` | metrics, figure images, CPU analysis → `paper_results_revision3.zip` | ~1.5 h |
| `prereq`      | **only on a new pod**: ref/text vectors, VAL T2I origin, main TEST run "ours" (3 seeds), SHIFT α=6 + teacher on TEST seed 42 for the figures | ~4.5 h; 0 on the old volume |

## 0. On your computer: publish the code (once)
The revision-3 code is not in git yet.

    git add scripts/run_revision3.sh scripts/revision3_analysis.py scripts/make_limits_figure.py \
            scripts/make_revision3_tables.py scripts/make_vecsim_figure.py \
            src/steering/apply_steering_klein.py REVISION3_RUNBOOK.md
    git commit -m "Revision 3: angle-matched control, held-out block check, SHIFT rule, prereq stage"
    git push origin flux-klein

## 1a. The old RunPod volume still exists (experiments/klein_9b/paper/ is there)

    cd SHIFT && git pull
    ls experiments/klein_9b/paper/*/ref/vectors | head      # must list *_diff.pt
    tmux new -s rev3
    bash scripts/run_revision3.sh                           # prereq finds everything and does nothing

## 1b. New pod (volume lost)
GPU ≥ 48 GB (L40S as in Table 2), ~200 GB disk, RAM ≥ 64 GB.

    git clone -b flux-klein https://github.com/gugmelik/SHIFT && cd SHIFT
    conda create -n shift python=3.11 -y && conda activate shift
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt "diffusers>=0.37" lpips piq "torchmetrics[image]" scipy matplotlib \
                ftfy regex huggingface_hub sentencepiece protobuf
    git clone https://github.com/learn2phoenix/CSD third_party/CSD
    huggingface-cli login        # licences accepted: FLUX.2-klein-9B, dinov3-vitl16-pretrain-lvd1689m
    huggingface-cli download tomg-group-umd/CSD-ViT-L pytorch_model.bin --local-dir weights/csd
    # the same 4 references as in the paper; the two Commons files are pinned so the lookup cannot pick another image
    python scripts/download_open_references.py --commons_file \
        pirosmani_giraffe=File:Pirosmani_-_Giraffe.jpg ciurlionis_sonata_sun=File:Saules_sonata.Andante.jpg
    ls data/reference_images/open/    # exactly: ciurlionis_sonata_sun, monet_water_lilies, pirosmani_giraffe, vangogh_self_portrait
    tmux new -s rev3
    bash scripts/run_revision3.sh

Several GPUs (one reference each), then the analysis once:

    R=data/reference_images/open
    CUDA_VISIBLE_DEVICES=0 REFS="$R/monet_water_lilies.jpg"    STAGES="prereq angle blocks_test shift_sweep score examples" bash scripts/run_revision3.sh > logs/g0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 REFS="$R/vangogh_self_portrait.jpg" STAGES="prereq angle blocks_test shift_sweep score examples" bash scripts/run_revision3.sh > logs/g1.log 2>&1 &
    CUDA_VISIBLE_DEVICES=2 REFS="$R/ciurlionis_sonata_sun.jpg" STAGES="prereq angle blocks_test shift_sweep score examples" bash scripts/run_revision3.sh > logs/g2.log 2>&1 &
    CUDA_VISIBLE_DEVICES=3 REFS="$R/pirosmani_giraffe.jpg"     STAGES="prereq angle blocks_test shift_sweep score examples" bash scripts/run_revision3.sh > logs/g3.log 2>&1 &
    wait
    STAGES=analysis bash scripts/run_revision3.sh

On a new pod the vectors are re-extracted with the same prompts and seeds; the held-out block
configurations and the angle control are compared with the "ours"/T2I runs of the SAME pod, so the
comparisons stay paired. The tables of revision 2 (Tables 4–8) are not recomputed and stay as they are.

## 2. Check and send back

    tail -n 3 logs/*.log
    ls experiments/klein_9b/paper/*/scores/test_angle_2.6_all_seed_42.json   # 4 files
    ls paper_results_revision3.zip

Restart after a crash with the same command: finished images and scores are skipped.
Send back `paper_results_revision3.zip`. Then in the paper: replace `figs_v5/limits_grid.jpg` by
`limits_grid_shift.jpg`, add the angle-control and held-out results to Sec. 5.2, and drop the corresponding
items from Sec. 6 «Ограничения эксперимента».
