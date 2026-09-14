#!/bin/bash
# Step 3/3 (Klein 9B): apply the style vector at T2I time (no reference image).
# Run from repository root:
#   bash scripts/apply_steering_klein.sh path/to/style.jpg
#   bash scripts/apply_steering_klein.sh style
set -euo pipefail

SCRIPT_NAME="apply_steering_klein.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${ROOT}"
# shellcheck source=scripts/_klein_ref_env.sh
source "${ROOT}/scripts/_klein_ref_env.sh" "${1:-}"

if [[ ! -d "${VECTOR_DIR}" ]]; then
  echo "ERROR: ${VECTOR_DIR} not found. Run scripts/steering_calculate_klein.sh first."
  exit 1
fi
if ! ls "${VECTOR_DIR}"/*diff.pt >/dev/null 2>&1; then
  echo "ERROR: no *_diff.pt in ${VECTOR_DIR}. Run scripts/steering_calculate_klein.sh first."
  ls -la "${VECTOR_DIR}" || true
  exit 1
fi
if ! ls "${VECTOR_DIR}"/*svm_models.pt >/dev/null 2>&1; then
  echo "ERROR: no *_svm_models.pt in ${VECTOR_DIR}. Run scripts/steering_calculate_klein.sh first (SVM step)."
  ls -la "${VECTOR_DIR}" || true
  exit 1
fi
echo "Using vectors in ${VECTOR_DIR}:"
ls -lh "${VECTOR_DIR}"/*.pt

mkdir -p "${RESULTS_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" ./src/steering/apply_steering_klein.py \
    --model_name "${MODEL_NAME}" \
    --data_dir "${VECTOR_DIR}" \
    --task "add concept" \
    --strength 45.0 \
    --strength_txt 6.0 \
    --strength_img 0.0 \
    --top_k_percent 0.95 \
    --min_signal_threshold 0.05 \
    --cls_min 20.0 \
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
    --use_cls \
    --save_origin \
    --prompts_path "${PROMPT_PATH}"

echo "Check origin vs steered:"
echo "  ${RESULTS_DIR}/origin"
echo "  ${RESULTS_DIR}/steered"
echo "Score DINOv2 with:"
echo "  bash scripts/eval_dino_klein.sh ${REF_STEM}"
