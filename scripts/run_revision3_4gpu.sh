#!/bin/bash
# Revision 3 on several GPUs of one machine: one reference per GPU, then the CPU analysis once.
#   bash scripts/run_revision3_4gpu.sh            # GPUS="0 1 2 3" by default
# Restartable: rerun the same command after a crash, finished work is skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
GPUS=(${GPUS:-0 1 2 3})
R="${REF_DIR:-data/reference_images/open}"
REFS=("$R/monet_water_lilies.jpg" "$R/vangogh_self_portrait.jpg" "$R/ciurlionis_sonata_sun.jpg" "$R/pirosmani_giraffe.jpg")
for f in "${REFS[@]}"; do [[ -f "$f" ]] || { echo "missing $f: run scripts/download_open_references.py"; exit 1; }; done
(( ${#GPUS[@]} >= ${#REFS[@]} )) || { echo "need ${#REFS[@]} GPUs, got ${#GPUS[@]}"; exit 1; }

# Fill the Hugging Face cache once, so four processes do not download the same weights at the same time.
huggingface-cli download "${MODEL_NAME:-black-forest-labs/FLUX.2-klein-9B}" >/dev/null
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m >/dev/null || true

pids=()
for i in "${!REFS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" REFS="${REFS[$i]}" \
    STAGES="prereq angle blocks_test shift_sweep score examples" \
    bash scripts/run_revision3.sh > "logs/rev3_gpu${GPUS[$i]}.log" 2>&1 &
  pids+=($!)
  echo "GPU ${GPUS[$i]}: $(basename "${REFS[$i]}")  pid $!  log logs/rev3_gpu${GPUS[$i]}.log"
  sleep "${STAGGER:-180}"   # stagger model loading: four parallel loads need ~4x35 GB of CPU RAM
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail )) && { echo "a GPU job failed: see logs/rev3_gpu*.log, then rerun this script"; exit 1; }
STAGES=analysis bash scripts/run_revision3.sh 2>&1 | tee logs/rev3_analysis.log
