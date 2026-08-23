#!/bin/bash
# Step 1/3 (Klein 9B): extract double-stream activations.
# Positive pass uses the given reference image; negative pass is T2I only.
# Outputs go to experiments/klein_9b/style/<image_stem>/
# Run from repository root:
#   bash scripts/get_vector_klein.sh path/to/style.jpg
set -euo pipefail

SCRIPT_NAME="get_vector_klein.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${ROOT}"
# shellcheck source=scripts/_klein_ref_env.sh
source "${ROOT}/scripts/_klein_ref_env.sh" "${1:-}"

if [[ -z "${REF_IMAGE}" || ! -f "${REF_IMAGE}" ]]; then
  echo "ERROR: extract needs a real image file, not just a folder stem."
  echo "Usage: bash scripts/get_vector_klein.sh path/to/style.jpg"
  exit 1
fi

mkdir -p "${SAVE_DIR}" "${SAVE_IMAGE_DIR}" "${EXP_ROOT}"
_ext="${REF_IMAGE##*.}"
cp -f "${REF_IMAGE}" "${EXP_ROOT}/reference.${_ext}"

echo "Reference image: ${REF_IMAGE}"
echo "Experiment dir:  ${EXP_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" ./src/steering/get_vector_klein.py \
    --model_name "${MODEL_NAME}" \
    --exp_type "${EXP_TYPE}" \
    --prompt_path "${PROMPT_PATH}" \
    --num_prompts "${NUM_PROMPTS}" \
    --reference_image "${REF_IMAGE}" \
    --token_stream both \
    --num_layers 8 \
    --save_timesteps 4 \
    --height 1024 \
    --width 1024 \
    --gs "${GS}" \
    --num_inference_steps 4 \
    --batch_size 1 \
    --save_dir "${SAVE_DIR}" \
    --save_image_dir "${SAVE_IMAGE_DIR}"

echo "Extracted:"
echo "  ${POS_PATH}"
echo "  ${NEG_PATH}"
echo "Next: bash scripts/steering_calculate_klein.sh ${REF_IMAGE}"
