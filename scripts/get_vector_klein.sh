#!/bin/bash
# Step 1/3 (Klein 9B): extract double-stream activations.
# Positive pass uses one shared reference image; negative pass is T2I only.
# Drop the reference image in data/reference_images/ first.
# Run from repository root: bash scripts/get_vector_klein.sh
set -euo pipefail

MODEL_NAME="black-forest-labs/FLUX.2-klein-9B"
EXP_TYPE="style_ref"
NUM_PROMPTS=25
SAVE_DIR="experiments/klein_9b/style/data_vectors"
SAVE_IMAGE_DIR="experiments/klein_9b/style/dataset_images"
PROMPT_PATH="prompts_collection/dataset_creation/dataset_prompts_style.txt"
REFERENCE_IMAGE="data/reference_images"

mkdir -p "${SAVE_DIR}" "${SAVE_IMAGE_DIR}"

CUDA_VISIBLE_DEVICES=0 python ./src/steering/get_vector_klein.py \
    --model_name "${MODEL_NAME}" \
    --exp_type "${EXP_TYPE}" \
    --prompt_path "${PROMPT_PATH}" \
    --num_prompts "${NUM_PROMPTS}" \
    --reference_image "${REFERENCE_IMAGE}" \
    --token_stream both \
    --num_layers 8 \
    --save_timesteps 4 \
    --height 1024 \
    --width 1024 \
    --gs 1.0 \
    --num_inference_steps 4 \
    --batch_size 1 \
    --save_dir "${SAVE_DIR}" \
    --save_image_dir "${SAVE_IMAGE_DIR}"
