#!/bin/bash
# Step 2/3 (Klein 9B): mean-diff steering vectors from extracted activations.
# Edit paths to match files from scripts/get_vector_klein.sh.
# Run from repository root: bash scripts/steering_calculate_klein.sh
set -euo pipefail

POS_PATH="experiments/klein_9b/style/data_vectors/style_ref_gs_1.0_prompts_25_pos_block.pt"
NEG_PATH="experiments/klein_9b/style/data_vectors/style_ref_gs_1.0_prompts_25_neg_block.pt"
SAVE_DIR="experiments/klein_9b/style/final_steering/block_steering"

mkdir -p "${SAVE_DIR}"

python ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --method diff \
    --timesteps 4 \
    --blocks 8 \
    --n_samples 25 \
    --classifier none
