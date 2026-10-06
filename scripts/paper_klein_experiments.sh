#!/bin/bash
# Experiments requested in the review of the Klein reference-image steering paper.
# Run from the repository root on a GPU machine:
#
#   bash scripts/paper_klein_experiments.sh                 # every image in data/reference_images
#   REFS="data/reference_images/picasso_style.jpg" bash scripts/paper_klein_experiments.sh
#   STAGES="extract sweep" bash scripts/paper_klein_experiments.sh
#
# Stages (STAGES, default: all of them, in this order):
#   extract   pos/neg activations on the TRAIN prompts + mean-diff vectors (+ SVM files)
#   textbase  text-only SHIFT baseline on Klein (needs a style tag in STYLE_TAGS)
#   sweep     alpha_img grid on the VAL prompts, with token-norm and timing stats
#   ablation  block / step / pooled-vector / text-stream / SVM ablations on VAL (ALPHA)
#   test      final run on the TEST prompts with ALPHA, 3 seeds, + I2I teacher images
#   score     extended metrics (metrics/eval_klein_extended.py) and Wilcoxon comparison
#
# Rerunning is safe: finished work is detected per reference and per run and skipped
# (vectors, every alpha/ablation/test folder, teacher images, score files). FORCE=1 redoes it.
#
# Outputs: experiments/klein_9b/paper/<reference_stem>/...
# Then: python scripts/collect_paper_results.py  -> paper_results/ (tables, figures, summary)
# ALPHA must be chosen from the sweep (see the selection rule in the paper, Section 3.4)
# before running the ablation and test stages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-9B}"
TRAIN="${TRAIN:-prompts_collection/klein_style/train_prompts.txt}"
VAL="${VAL:-prompts_collection/klein_style/val_prompts.txt}"
TEST="${TEST:-prompts_collection/klein_style/test_prompts.txt}"
ALPHAS="${ALPHAS:-2 4 6 7 8 10 12}"
ALPHA="${ALPHA:-6}"
SEEDS="${SEEDS:-42 1042 2042}"
STAGES="${STAGES:-extract textbase sweep ablation test score}"
# Open-access (CC0) references: run `python scripts/download_open_references.py` first.
REF_DIR="${REF_DIR:-data/reference_images/open}"
# Tab-separated "<reference_stem>\t<suffix>" for the text-SHIFT baseline (written by the downloader)
STYLE_TAGS="${STYLE_TAGS:-${REF_DIR}/style_tags.tsv}"
REFS="${REFS:-$(ls "${REF_DIR}"/*.{jpg,jpeg,png,webp} 2>/dev/null || true)}"
if [[ -z "${REFS}" ]]; then
  echo "No reference images in ${REF_DIR}. Run: python scripts/download_open_references.py"
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

count_lines() { grep -c . "$1"; }
has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

# ---- resume support -------------------------------------------------------------
# Every unit of work (extraction, each apply run, teacher images) is skipped when its
# outputs are already complete, so rerunning the script never repeats finished work
# for a reference. Set FORCE=1 to redo everything that the selected STAGES cover.
n_images() { compgen -G "$1/[0-9]*_*.png" 2>/dev/null | wc -l; }
images_complete() {  # <dir> <prompt_file>
  [[ "${FORCE:-0}" != "1" ]] && [[ -d "$1" ]] && (( $(n_images "$1") >= $(count_lines "$2") ))
}
vectors_complete() {  # <vector_dir>
  [[ "${FORCE:-0}" != "1" ]] && compgen -G "$1/*_diff.pt" >/dev/null && compgen -G "$1/*_svm_models.pt" >/dev/null
}
activations_complete() {  # <data_vectors_dir>
  [[ "${FORCE:-0}" != "1" ]] && compgen -G "$1/*_pos_block.pt" >/dev/null && compgen -G "$1/*_neg_block.pt" >/dev/null
}
skip() { echo "  skip (done): $*"; }

extract_and_calculate() {  # <tag: ref|text> [extra extract args...]
  local kind="$1"; shift
  local base="${EXP}/${kind}"
  if vectors_complete "${base}/vectors"; then
    skip "${kind} vectors"
    return 0
  fi
  if activations_complete "${base}/data_vectors"; then
    skip "${kind} activations (recomputing vectors only)"
  else
    local t0; t0=$(date +%s)
    mkdir -p "${base}"
    extract "${REF}" "${base}/data_vectors" "${base}/dataset_images" "$@"
    echo "{\"seconds\": $(( $(date +%s) - t0 )), \"n_prompts\": $(count_lines "${TRAIN}")}" \
        > "${base}/extract_timing.json"
  fi
  calculate "${base}/data_vectors" "${base}/vectors"
}

extract() {  # <ref_or_empty> <save_dir> <img_dir> [extra args...]
  local ref="$1" save="$2" imgs="$3"; shift 3
  "${PYTHON}" ./src/steering/get_vector_klein.py \
      --model_name "${MODEL_NAME}" --exp_type style_ref --prompt_path "${TRAIN}" \
      --reference_image "${ref}" --token_stream both --num_layers 8 --save_timesteps 4 \
      --height 1024 --width 1024 --gs 1.0 --num_inference_steps 4 --batch_size 1 \
      --save_dir "${save}" --save_image_dir "${imgs}" "$@"
}

calculate() {  # <data_vectors_dir> <out_dir>
  local pos neg n
  n="$(count_lines "${TRAIN}")"
  pos="$(ls "$1"/*_pos_block.pt)"; neg="$(ls "$1"/*_neg_block.pt)"
  "${PYTHON}" ./src/steering/calculate_steering_vectors.py --pos_path "${pos}" --neg_path "${neg}" \
      --save_dir "$2" --save_svm --timesteps 4 --blocks 8 --n_samples "${n}" --token_stream both --classifier none
  "${PYTHON}" ./src/steering/calculate_steering_vectors.py --pos_path "${pos}" --neg_path "${neg}" \
      --save_dir "$2" --method diff --timesteps 4 --blocks 8 --n_samples "${n}" --token_stream both --classifier none
}

apply() {  # <vector_dir> <prompts> <results_dir> <stats_json> [extra args...]
  local vec="$1" prompts="$2" out="$3" stats="$4"; shift 4
  if images_complete "${out}/steered" "${prompts}" && images_complete "${out}/origin" "${prompts}"; then
    skip "${out}"
    [[ -f "${stats}" ]] || echo "    note: ${stats} missing (norm/timing stats); use FORCE=1 on this run if you need them"
    return 0
  fi
  "${PYTHON}" ./src/steering/apply_steering_klein.py \
      --model_name "${MODEL_NAME}" --data_dir "${vec}" --prompts_path "${prompts}" \
      --task "add concept" --vector_type diff --inference_steps 4 --guidance_scale 1.0 \
      --width 1024 --height 1024 --strength 0 --injection_point block \
      --results_dir "${out}" --stats_path "${stats}" --save_origin "$@"
}

for REF in ${REFS}; do
  STEM="$(basename "${REF%.*}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"
  EXP="experiments/klein_9b/paper/${STEM}"
  mkdir -p "${EXP}"
  cp -f "${REF}" "${EXP}/reference.${REF##*.}"
  echo "=== ${STEM} ==="

  if has_stage extract; then
    extract_and_calculate ref
  fi

  if has_stage textbase; then
    TAG="$(awk -F'\t' -v s="${STEM}" '$1==s {print $2}' "${STYLE_TAGS}" 2>/dev/null || true)"
    if [[ -z "${TAG}" ]]; then
      echo "  textbase: no style tag for ${STEM} in ${STYLE_TAGS}; skipped"
    else
      extract_and_calculate text --pos_suffix "${TAG}"
    fi
  fi

  if has_stage sweep; then
    for A in ${ALPHAS}; do
      apply "${EXP}/ref/vectors" "${VAL}" "${EXP}/val/alpha_${A}" "${EXP}/val/alpha_${A}/stats.json" \
          --strength_img "${A}" --steering_type separate --seed 42
    done
  fi

  if has_stage ablation; then
    V="${EXP}/ref/vectors"; O="${EXP}/val/ablation"
    apply "$V" "${VAL}" "$O/blocks_0-3"  "$O/blocks_0-3/stats.json"  --strength_img "${ALPHA}" --block_steering 0,1,2,3 --seed 42
    apply "$V" "${VAL}" "$O/blocks_4-7"  "$O/blocks_4-7/stats.json"  --strength_img "${ALPHA}" --block_steering 4,5,6,7 --seed 42
    apply "$V" "${VAL}" "$O/steps_0"     "$O/steps_0/stats.json"     --strength_img "${ALPHA}" --t_steering 0 --seed 42
    apply "$V" "${VAL}" "$O/steps_0-1"   "$O/steps_0-1/stats.json"   --strength_img "${ALPHA}" --t_steering 0,1 --seed 42
    apply "$V" "${VAL}" "$O/steps_2-3"   "$O/steps_2-3/stats.json"   --strength_img "${ALPHA}" --t_steering 2,3 --seed 42
    apply "$V" "${VAL}" "$O/pooled"      "$O/pooled/stats.json"      --strength_img "${ALPHA}" --steering_type mean --seed 42
    apply "$V" "${VAL}" "$O/with_txt"    "$O/with_txt/stats.json"    --strength_img "${ALPHA}" --strength "${ALPHA_TXT:-6}" --seed 42
    apply "$V" "${VAL}" "$O/svm"         "$O/svm/stats.json"         --strength_img "${ALPHA}" --use_cls --seed 42
  fi

  if has_stage test; then
    for SEED in ${SEEDS}; do
      apply "${EXP}/ref/vectors" "${TEST}" "${EXP}/test/ours/seed_${SEED}" "${EXP}/test/ours/seed_${SEED}/stats.json" \
          --strength_img "${ALPHA}" --steering_type separate --seed "${SEED}"
      if [[ -d "${EXP}/text/vectors" ]]; then
        apply "${EXP}/text/vectors" "${TEST}" "${EXP}/test/shift_text/seed_${SEED}" \
            "${EXP}/test/shift_text/seed_${SEED}/stats.json" \
            --strength_img "${ALPHA_TEXTBASE:-${ALPHA}}" --steering_type separate --seed "${SEED}"
      fi
    done
    if images_complete "${EXP}/test/teacher/i2i" "${TEST}"; then
      skip "${EXP}/test/teacher"
    else
      "${PYTHON}" ./src/steering/get_vector_klein.py --model_name "${MODEL_NAME}" --exp_type style_ref \
          --prompt_path "${TEST}" --reference_image "${REF}" --height 1024 --width 1024 --gs 1.0 \
          --num_inference_steps 4 --batch_size 1 --save_image_dir "${EXP}/test/teacher" --i2i_only
    fi
  fi

  if has_stage score; then
    # Scoring only reads images that already exist; it never generates or steers.
    # Existing score files are kept (set FORCE_SCORE=1 to recompute them).
    S="${EXP}/scores"; mkdir -p "$S"
    # CSD_CKPT: path to the CSD checkpoint or the Hub id tomg-group-umd/CSD-ViT-L (see README, section 6).
    CSD_ARGS=()
    [[ -n "${CSD_CKPT:-}" ]] && CSD_ARGS=(--csd_ckpt "${CSD_CKPT}")
    score() {  # <out_json> <args...>
      local out="$1"; shift
      if [[ -f "${out}" && "${FORCE_SCORE:-0}" != "1" ]]; then
        if [[ -n "${CSD_CKPT:-}" ]] && ! grep -q '"csd_to_reference"' "${out}"; then
          echo "  add CSD to ${out}"
          "${PYTHON}" metrics/eval_klein_extended.py score --out "${out}" --merge "${CSD_ARGS[@]}" "$@"
        else
          echo "  keep ${out}"
        fi
        return 0
      fi
      "${PYTHON}" metrics/eval_klein_extended.py score --out "${out}" "${CSD_ARGS[@]}" "$@"
    }
    has_images() { [[ -d "$1" ]] && compgen -G "$1/[0-9]*_*" >/dev/null; }

    # validation runs: alpha sweep and ablations (used for the alpha selection and Table 6)
    for D in "${EXP}"/val/alpha_* "${EXP}"/val/ablation/*; do
      has_images "$D/steered" || continue
      score "$S/val_$(basename "$D").json" --gen_dir "$D/steered" --origin_dir "$D/origin" \
          --reference "${REF}" --prompts "${VAL}"
    done

    TEACHER="${EXP}/test/teacher/i2i"
    FID_ARGS=()
    has_images "${TEACHER}" && FID_ARGS=(--fid_target_dir "${TEACHER}")
    for SEED in ${SEEDS}; do
      for M in ours shift_text; do
        D="${EXP}/test/${M}/seed_${SEED}"
        has_images "$D/steered" || continue
        score "$S/${M}_seed_${SEED}.json" --gen_dir "$D/steered" --origin_dir "$D/origin" \
            --reference "${REF}" --prompts "${TEST}" "${FID_ARGS[@]}"
      done
      D="${EXP}/test/ours/seed_${SEED}/origin"
      has_images "$D" && score "$S/t2i_seed_${SEED}.json" --gen_dir "$D" \
          --reference "${REF}" --prompts "${TEST}" "${FID_ARGS[@]}"
    done
    has_images "${TEACHER}" && score "$S/teacher.json" --gen_dir "${TEACHER}" \
        --reference "${REF}" --prompts "${TEST}"

    if [[ -f "$S/ours_seed_42.json" && -f "$S/t2i_seed_42.json" ]]; then
      RUNS=("$S/ours_seed_42.json" "$S/t2i_seed_42.json")
      [[ -f "$S/teacher.json" ]] && RUNS+=("$S/teacher.json")
      [[ -f "$S/shift_text_seed_42.json" ]] && RUNS+=("$S/shift_text_seed_42.json")
      "${PYTHON}" metrics/eval_klein_extended.py compare --runs "${RUNS[@]}" --out "$S/compare_seed_42.json"
    fi
  fi
done

echo "Done. Results under experiments/klein_9b/paper/"
