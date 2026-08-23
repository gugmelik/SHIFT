#!/bin/bash
# Step 2/3 (Klein 9B): compute steering vectors from extracted activations.
# Run from repository root:
#   bash scripts/steering_calculate_klein.sh path/to/style.jpg
#   bash scripts/steering_calculate_klein.sh style
set -euo pipefail

SCRIPT_NAME="steering_calculate_klein.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${ROOT}"
# shellcheck source=scripts/_klein_ref_env.sh
source "${ROOT}/scripts/_klein_ref_env.sh" "${1:-}"

if [[ ! -f "${POS_PATH}" || ! -f "${NEG_PATH}" ]]; then
  echo "ERROR: missing activation dumps. Run extract first:"
  echo "  bash scripts/get_vector_klein.sh <reference_image>"
  echo "Expected:"
  echo "  ${POS_PATH}"
  echo "  ${NEG_PATH}"
  exit 1
fi

mkdir -p "${VECTOR_DIR}"

echo "Pos: ${POS_PATH}"
echo "Neg: ${NEG_PATH}"
echo "Out: ${VECTOR_DIR}"

# Classifier / scores (for --use_cls). Needs the compact 'pooled' (N, D)
# entry written by get_vector_klein.py — re-extract if dumps are prompt-means only.
"${PYTHON}" ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${VECTOR_DIR}" \
    --save_svm \
    --timesteps 4 \
    --blocks 8 \
    --n_samples "${NUM_PROMPTS}" \
    --token_stream both \
    --classifier none

# Mean-difference steering vector
"${PYTHON}" ./src/steering/calculate_steering_vectors.py \
    --pos_path "${POS_PATH}" \
    --neg_path "${NEG_PATH}" \
    --save_dir "${VECTOR_DIR}" \
    --method diff \
    --timesteps 4 \
    --blocks 8 \
    --n_samples "${NUM_PROMPTS}" \
    --token_stream both \
    --classifier none

echo "Wrote vectors in ${VECTOR_DIR}:"
ls -lh "${VECTOR_DIR}"/*.pt
echo "Next: bash scripts/apply_steering_klein.sh ${1}"
