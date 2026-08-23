# Shared Klein style-steering paths. Source from the repository root:
#   source scripts/_klein_ref_env.sh <reference_image_or_stem>
#
# <reference_image_or_stem> is a path to one image (.png/.jpg/.jpeg/.webp/.bmp),
# or the already-created experiment folder name (image stem) for calculate/apply.
#
# Sets: REF_IMAGE, REF_STEM, EXP_ROOT, SAVE_DIR, SAVE_IMAGE_DIR, VECTOR_DIR,
#       RESULTS_DIR, POS_PATH, NEG_PATH, and default extract settings.

_klein_usage() {
  echo "Usage: bash scripts/${SCRIPT_NAME:-<klein_script>.sh} <reference_image>"
  echo "  reference_image: path to a .png/.jpg/.jpeg/.webp (folder name = image stem)"
  echo "  Outputs: experiments/klein_9b/style/<image_stem>/"
  echo "  Calculate/apply can also take that stem if the folder already exists."
}

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  _klein_usage
  exit 1
fi

_arg="${1//\\//}"

if [[ -n "${CALLER_PWD:-}" && ! -e "${_arg}" && -e "${CALLER_PWD}/${_arg}" ]]; then
  _arg="${CALLER_PWD}/${_arg}"
fi

NUM_PROMPTS="${NUM_PROMPTS:-25}"
GS="${GS:-1.0}"
EXP_TYPE="${EXP_TYPE:-style_ref}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-9B}"
PROMPT_PATH="${PROMPT_PATH:-prompts_collection/dataset_creation/dataset_prompts_style.txt}"

_klein_sanitize() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g; s/__*/_/g; s/^_//; s/_$//'
}

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    PYTHON=python3
  fi
fi

REF_IMAGE=""
if [[ -f "${_arg}" ]]; then
  REF_IMAGE="${_arg}"
  _base="$(basename "${_arg}")"
  REF_STEM="$(_klein_sanitize "${_base%.*}")"
elif [[ -d "experiments/klein_9b/style/${_arg}" ]]; then
  REF_STEM="$(_klein_sanitize "$(basename "${_arg}")")"
elif [[ -d "experiments/klein_9b/style/$(_klein_sanitize "${_arg}")" ]]; then
  REF_STEM="$(_klein_sanitize "${_arg}")"
elif [[ -d "${_arg}" ]]; then
  _maybe_stem="$(_klein_sanitize "$(basename "${_arg}")")"
  if [[ -d "experiments/klein_9b/style/${_maybe_stem}" ]]; then
    REF_STEM="${_maybe_stem}"
  else
    echo "ERROR: pass a single image file, not a directory: ${_arg}"
    _klein_usage
    exit 1
  fi
else
  echo "ERROR: reference image not found: ${_arg}"
  echo "Pass a real image file, or an existing experiment stem under experiments/klein_9b/style/"
  _klein_usage
  exit 1
fi

if [[ -z "${REF_STEM}" ]]; then
  echo "ERROR: could not derive a folder name from ${_arg}"
  exit 1
fi

EXP_ROOT="experiments/klein_9b/style/${REF_STEM}"
SAVE_DIR="${EXP_ROOT}/data_vectors"
SAVE_IMAGE_DIR="${EXP_ROOT}/dataset_images"
VECTOR_DIR="${EXP_ROOT}/final_steering/block_steering"
RESULTS_DIR="${EXP_ROOT}/generated_images"
POS_PATH="${SAVE_DIR}/${EXP_TYPE}_gs_${GS}_prompts_${NUM_PROMPTS}_pos_block.pt"
NEG_PATH="${SAVE_DIR}/${EXP_TYPE}_gs_${GS}_prompts_${NUM_PROMPTS}_neg_block.pt"
