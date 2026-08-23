#!/bin/bash
# Step 3/3 (Klein 9B): apply the style vector at T2I time (no reference image).
# Run after scripts/steering_calculate_klein.sh
# Run from repository root: bash scripts/apply_steering_klein.sh
set -euo pipefail

DATA_DIR="experiments/klein_9b/style/final_steering/block_steering"
RESULTS_DIR="experiments/klein_9b/style/generated_images"
PROMPTS_PATH="prompts_collection/dataset_creation/dataset_prompts_style.txt"

mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES=0 python ./src/steering/apply_steering_klein.py \
    --model_name "black-forest-labs/FLUX.2-klein-9B" \
    --data_dir "${DATA_DIR}" \
    --prompts_path "${PROMPTS_PATH}" \
    --num_prompts 25 \
    --task "add concept" \
    --strength 10.0 \
    --strength_img 10.0 \
    --vector_type diff \
    --steering_type mean \
    --num_layers 8 \
    --guidance_scale 1.0 \
    --inference_steps 4 \
    --width 1024 \
    --height 1024 \
    --seed 42 \
    --results_dir "${RESULTS_DIR}"
