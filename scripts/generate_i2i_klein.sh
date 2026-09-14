#!/bin/bash
# Generate per-prompt Klein I2I teacher images (no activation dumps).
# Use this if extract already ran before individual I2I files were saved.
# Run from repository root:
#   bash scripts/generate_i2i_klein.sh data/reference_images/picasso_style.jpg
set -euo pipefail

SCRIPT_NAME="generate_i2i_klein.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${ROOT}"
# shellcheck source=scripts/_klein_ref_env.sh
source "${ROOT}/scripts/_klein_ref_env.sh" "${1:-}"

if [[ -z "${REF_IMAGE}" || ! -f "${REF_IMAGE}" ]]; then
  echo "ERROR: needs a real image file, not just a folder stem."
  echo "Usage: bash scripts/generate_i2i_klein.sh path/to/style.jpg"
  exit 1
fi

mkdir -p "${SAVE_IMAGE_DIR}" "${EXP_ROOT}"
_ext="${REF_IMAGE##*.}"
cp -f "${REF_IMAGE}" "${EXP_ROOT}/reference.${_ext}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" ./src/steering/get_vector_klein.py \
    --model_name "${MODEL_NAME}" \
    --exp_type "${EXP_TYPE}" \
    --prompt_path "${PROMPT_PATH}" \
    --num_prompts "${NUM_PROMPTS}" \
    --reference_image "${REF_IMAGE}" \
    --height 1024 \
    --width 1024 \
    --gs "${GS}" \
    --num_inference_steps 4 \
    --batch_size 1 \
    --save_image_dir "${SAVE_IMAGE_DIR}" \
    --i2i_only

echo "I2I teacher images: ${SAVE_IMAGE_DIR}/i2i"
echo "Next: bash scripts/eval_dino_klein.sh ${REF_STEM}"
