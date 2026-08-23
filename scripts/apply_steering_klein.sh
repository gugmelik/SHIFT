#!/bin/bash
# Step 3/3 (Klein 9B): apply the style vector at T2I time (no reference image).
# Run after scripts/steering_calculate_klein.sh
# Run from repository root: bash scripts/apply_steering_klein.sh
set -euo pipefail

DATA_DIR="experiments/klein_9b/style/final_steering/block_steering"
RESULTS_DIR="experiments/klein_9b/style/generated_images"
PROMPTS_PATH="prompts_collection/dataset_creation/dataset_prompts_style.txt"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: ${DATA_DIR} not found. Run scripts/steering_calculate_klein.sh first."
  exit 1
fi
if ! ls "${DATA_DIR}"/*diff.pt >/dev/null 2>&1; then
  echo "ERROR: no *_diff.pt in ${DATA_DIR}. Run scripts/steering_calculate_klein.sh first."
  ls -la "${DATA_DIR}" || true
  exit 1
fi
echo "Using vectors in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"/*.pt

mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES=0 python3 ./src/steering/apply_steering_klein.py \
    --model_name "black-forest-labs/FLUX.2-klein-9B" \
    --data_dir "${DATA_DIR}" \
    --task "add concept" \
    --strength 45.0 \
    --strength_txt 6.0 \
    --strength_img 0.0 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --injection_point block \
    --results_dir "${RESULTS_DIR}" \
    --inference_steps 4 \
    --seed 42 \
    --vector_type diff \
    --steering_type separate \
    --guidance_scale 0.0 \
    --width 1024 \
    --height 1024 \
    --steer_txt \
    --save_origin \
    --prompts_path "${PROMPTS_PATH}"

echo "Check origin vs steered:"
echo "  ${RESULTS_DIR}/origin"
echo "  ${RESULTS_DIR}/steered"
