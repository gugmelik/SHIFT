#!/bin/bash
# Step 2/3 (Klein 9B): compute steering vectors from extracted activations.
# Edit paths to match files from scripts/get_vector_klein.sh.
# Run from repository root: bash scripts/steering_calculate_klein.sh
set -euo pipefail

POS_PATH="experiments/klein_9b/style/data_vectors/style_ref_gs_1.0_prompts_25_pos_block.pt"
NEG_PATH="experiments/klein_9b/style/data_vectors/style_ref_gs_1.0_prompts_25_neg_block.pt"
SAVE_DIR="experiments/klein_9b/style/final_steering/block_steering"

mkdir -p "${SAVE_DIR}"

# Classifier / scores (for --use_cls). Needs the compact 'pooled' (N, D)
# entry written by get_vector_klein.py — re-extract if dumps are prompt-means only.
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --save_svm \
    --timesteps 4 \
    --blocks 8 \
    --n_samples 25 \
    --token_stream both \
    --classifier none

# Mean-difference steering vector
python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method diff \
    --timesteps 4 \
    --blocks 8 \
    --n_samples 25 \
    --token_stream both \
    --classifier none
