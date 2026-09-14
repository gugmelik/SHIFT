#!/bin/bash
# DINOv2 content + style scores for Klein style experiments.
# Run from repository root after origin / steered / I2I images exist:
#   bash scripts/eval_dino_klein.sh picasso_style
#   bash scripts/eval_dino_klein.sh --all
set -euo pipefail

SCRIPT_NAME="eval_dino_klein.sh"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    PYTHON=python3
  fi
fi

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: bash scripts/eval_dino_klein.sh <style_stem_or_image>"
  echo "       bash scripts/eval_dino_klein.sh --all"
  echo "  style: picasso_style | luca-illustration | sketch | path/to/style.jpg"
  exit 1
fi

if [[ "${1}" == "--all" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" ./metrics/dino_klein.py --all
  exit 0
fi

# shellcheck source=scripts/_klein_ref_env.sh
source "${ROOT}/scripts/_klein_ref_env.sh" "${1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON}" ./metrics/dino_klein.py --exp_dir "${EXP_ROOT}"
