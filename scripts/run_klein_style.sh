#!/bin/bash
# Extract activations and compute Klein style vectors for one reference image.
# Creates experiments/klein_9b/style/<image_stem>/ with all files needed to apply.
# Run from repository root:
#   bash scripts/run_klein_style.sh path/to/style.jpg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${ROOT}"

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: bash scripts/run_klein_style.sh <reference_image>"
  echo "  reference_image: path to a .png/.jpg/.jpeg/.webp"
  echo "  Writes: experiments/klein_9b/style/<image_stem>/"
  exit 1
fi

_arg="${1//\\//}"
if [[ ! -e "${_arg}" && -e "${CALLER_PWD}/${_arg}" ]]; then
  _arg="${CALLER_PWD}/${_arg}"
fi
if [[ -f "${_arg}" ]]; then
  _arg="$(cd "$(dirname "${_arg}")" && pwd)/$(basename "${_arg}")"
fi

bash "${ROOT}/scripts/get_vector_klein.sh" "${_arg}"
bash "${ROOT}/scripts/steering_calculate_klein.sh" "${_arg}"

# shellcheck source=scripts/_klein_ref_env.sh
source "${ROOT}/scripts/_klein_ref_env.sh" "${_arg}"

echo
echo "Done. Vectors are in ${VECTOR_DIR}"
echo "Apply with:"
echo "  bash scripts/apply_steering_klein.sh ${_arg}"
