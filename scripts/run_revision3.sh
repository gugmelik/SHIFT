#!/bin/bash
# Revision 3: experiments requested by the second review (run AFTER run_revision2_all.sh,
# in the same repository: it reuses vectors, T2I origins and scores of revision 2).
#
#   bash scripts/run_revision3.sh                       # all four references
#   REFS="data/reference_images/open/monet_water_lilies.jpg" CUDA_VISIBLE_DEVICES=1 bash scripts/run_revision3.sh
#
# Stages (STAGES, default all, in this order):
#   prereq       only on a NEW pod (revision-2 volume lost): rebuilds the minimum that the stages below
#                compare against, and nothing else -- reference vectors, text-SHIFT vectors, the VAL T2I
#                origin (seed 42), the main TEST run "ours" (alpha=ALPHA, SEEDS, with T2I origins) and, for
#                the figures only, SHIFT alpha=6 and the I2I teacher on TEST seed 42. Everything that
#                already exists is skipped, so on the old volume this stage does nothing.
#                No sweep, ablations, VAL blocks, name probe, baselines, twins or rule runs are repeated.
#   angle        angle-matched control (review item 1): every steered image-stream token is rotated by
#                exactly THETA degrees towards the steering direction (--angle_deg), so the angle is the
#                same in every block. only_k (k=1..8) and all blocks, VAL (seed 42) and TEST (seed 42).
#                THETAS default "2.6 3.0": 2.6 = mean angle of blocks 1-7 at alpha=6, 3.0 = block 1.
#   blocks_test  held-out confirmation of the central per-block results (review item 5): TEST prompts,
#                seeds 42 1042 2042, alpha=ALPHA: only_1, drop_1, drop_8, blocks 1-4, blocks 5-8, steps 3-4.
#   shift_sweep  SHIFT with the selection rule (review item 4): alpha sweep of the text vector on VAL,
#                the paper's rule, then TEST with 3 seeds -> test/shift_rule/seed_<S>.
#   score        metrics for everything above (eval_klein_extended.py), existing files kept.
#   examples     images for the qualitative figure with SHIFT and the limitation examples
#                -> paper_results_revision3/examples/
#   analysis     CPU: scripts/revision3_analysis.py -> paper_results_revision3/
# Every unit skips finished work; restart after a crash. Logs: logs/revision3_<stage>.log
# Approximate cost on one L40S: ~2.7 h per reference (~11 h for four), scoring ~1.5 h.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.2-klein-9B}"
VAL="${VAL:-prompts_collection/klein_style/val_prompts.txt}"
TEST="${TEST:-prompts_collection/klein_style/test_prompts.txt}"
ALPHA="${ALPHA:-6}"
ALPHAS="${ALPHAS:-2 4 6 7 8 10 12}"
SEEDS="${SEEDS:-42 1042 2042}"
THETAS="${THETAS:-2.6 3.0}"
ANGLE_TEST_THETAS="${ANGLE_TEST_THETAS:-2.6}"
EXAMPLE_IDX="${EXAMPLE_IDX:-0 10 20 40 60 80}"   # tiger, koala, baby, pocket watch, rice terrace, skyscraper
STAGES="${STAGES:-prereq angle blocks_test shift_sweep score examples analysis}"
EXP_ROOT="${EXP_ROOT:-experiments/klein_9b/paper}"
REF_DIR="${REF_DIR:-data/reference_images/open}"
REFS="${REFS:-$(ls "${REF_DIR}"/*.{jpg,jpeg,png,webp} 2>/dev/null || true)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${CSD_CKPT:-}" && -f weights/csd/pytorch_model.bin ]]; then export CSD_CKPT=weights/csd/pytorch_model.bin; fi
export PYTHONPATH="$PWD/third_party/CSD:$PWD/third_party/CSD/models:${PYTHONPATH:-}"

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }
count_lines() { grep -c . "$1"; }
n_images() { compgen -G "$1/[0-9]*_*.png" 2>/dev/null | wc -l; }
images_complete() { [[ "${FORCE:-0}" != "1" && -d "$1" ]] && (( $(n_images "$1") >= $(count_lines "$2") )); }
has_images() { [[ -d "$1" ]] && compgen -G "$1/[0-9]*_*" >/dev/null; }
copy_origin() { [[ -d "$2/origin" ]] || { mkdir -p "$2"; cp -r "$1" "$2/origin"; }; }
log() { echo "=== $(date '+%F %T') $*"; }

apply() {  # <vector_dir> <prompts> <out> [extra...]  (origin must already be in <out>/origin)
  local vec="$1" prompts="$2" out="$3"; shift 3
  if images_complete "${out}/steered" "${prompts}"; then echo "  skip ${out}"; return 0; fi
  "${PYTHON}" ./src/steering/apply_steering_klein.py \
      --model_name "${MODEL_NAME}" --data_dir "${vec}" --prompts_path "${prompts}" \
      --task "add concept" --vector_type diff --inference_steps 4 --guidance_scale 1.0 \
      --width 1024 --height 1024 --strength 0 --injection_point block \
      --results_dir "${out}" --stats_path "${out}/stats.json" "$@"
}
all_but() { seq 0 7 | grep -vx "$1" | paste -sd, -; }

score() {  # <out_json> <gen_dir> <origin_dir|-> <prompts>
  local out="$1" gen="$2" org="$3" pr="$4" extra=()
  has_images "${gen}" || return 0
  [[ "${org}" != "-" ]] && extra+=(--origin_dir "${org}")
  [[ -n "${CSD_CKPT:-}" ]] && extra+=(--csd_ckpt "${CSD_CKPT}")
  if [[ -f "${out}" && "${FORCE_SCORE:-0}" != "1" ]]; then echo "  keep ${out}"; return 0; fi
  "${PYTHON}" metrics/eval_klein_extended.py score --out "${out}" --gen_dir "${gen}" \
      --reference "${REF}" --prompts "${pr}" "${extra[@]}"
}

for REF in ${REFS}; do
  STEM="$(basename "${REF%.*}" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/_/g')"
  EXP="${EXP_ROOT}/${STEM}"; V="${EXP}/ref/vectors"; S="${EXP}/scores"
  log "${STEM}"

  if has_stage prereq; then
    PK=(env REFS="${REF}" REF_DIR="${REF_DIR}" EXP_ROOT="${EXP_ROOT}" MODEL_NAME="${MODEL_NAME}" PYTHON="${PYTHON}")
    mkdir -p "${EXP}"
    # vectors: ~13 min per reference and per kind (ref / text) on an L40S
    compgen -G "${V}/*_diff.pt" >/dev/null || "${PK[@]}" STAGES="extract" bash scripts/paper_klein_experiments.sh
    if has_stage shift_sweep || has_stage examples; then
      compgen -G "${EXP}/text/vectors/*_diff.pt" >/dev/null \
        || "${PK[@]}" STAGES="textbase" bash scripts/paper_klein_experiments.sh
    fi
    # VAL T2I origin (seed 42) = val/alpha_2/origin, used by the angle control
    images_complete "${EXP}/val/alpha_2/origin" "${VAL}" \
      || "${PK[@]}" ALPHAS="2" STAGES="sweep" bash scripts/paper_klein_experiments.sh
    # main TEST configuration (the "full" run the held-out block configurations are compared with)
    for SEED in ${SEEDS}; do
      D="${EXP}/test/ours/seed_${SEED}"
      images_complete "$D/origin" "${TEST}" && images_complete "$D/steered" "${TEST}" && continue
      rm -rf "$D/steered"
      apply "$V" "${TEST}" "$D" --strength_img "${ALPHA}" --steering_type separate --seed "${SEED}" --save_origin
    done
    if has_stage examples; then   # figure columns only (Figs. 6 and 7), seed 42
      D="${EXP}/test/shift_text/seed_42"
      if [[ -d "${EXP}/text/vectors" ]]; then
        copy_origin "${EXP}/test/ours/seed_42/origin" "$D"
        apply "${EXP}/text/vectors" "${TEST}" "$D" --strength_img "${ALPHA_TEXTBASE:-6}" --steering_type separate --seed 42
      fi
      TD="${EXP}/test/teacher/seed_42"
      images_complete "${TD}/i2i" "${TEST}" || "${PYTHON}" ./src/steering/get_vector_klein.py \
          --model_name "${MODEL_NAME}" --exp_type style_ref --prompt_path "${TEST}" --reference_image "${REF}" \
          --height 1024 --width 1024 --gs 1.0 --num_inference_steps 4 --batch_size 1 \
          --save_image_dir "${TD}" --i2i_only --seed_base 42
    fi
    # scores of these runs (kept if present)
    mkdir -p "$S"
    score "$S/val_t2i.json" "${EXP}/val/alpha_2/origin" - "${VAL}"
    for SEED in ${SEEDS}; do
      D="${EXP}/test/ours/seed_${SEED}"
      score "$S/t2i_seed_${SEED}.json" "$D/origin" - "${TEST}"
      score "$S/ours_seed_${SEED}.json" "$D/steered" "$D/origin" "${TEST}"
    done
  fi
  [[ -d "${V}" ]] || { echo "${STEM}: no vectors (add the prereq stage)"; continue; }

  if has_stage angle; then
    for TH in ${THETAS}; do
      O="${EXP}/val/angle_${TH}"
      for K in 0 1 2 3 4 5 6 7; do
        copy_origin "${EXP}/val/alpha_2/origin" "$O/only_$((K + 1))"
        apply "$V" "${VAL}" "$O/only_$((K + 1))" --angle_deg "${TH}" --block_steering "$K" --seed 42
      done
      copy_origin "${EXP}/val/alpha_2/origin" "$O/all"
      apply "$V" "${VAL}" "$O/all" --angle_deg "${TH}" --seed 42
    done
    for TH in ${ANGLE_TEST_THETAS}; do
      O="${EXP}/test/angle_${TH}"
      for K in 0 1 2 3 4 5 6 7; do
        copy_origin "${EXP}/test/ours/seed_42/origin" "$O/only_$((K + 1))"
        apply "$V" "${TEST}" "$O/only_$((K + 1))" --angle_deg "${TH}" --block_steering "$K" --seed 42
      done
      copy_origin "${EXP}/test/ours/seed_42/origin" "$O/all"
      apply "$V" "${TEST}" "$O/all" --angle_deg "${TH}" --seed 42
    done
  fi

  if has_stage blocks_test; then
    for SEED in ${SEEDS}; do
      for C in only_1:0 drop_1:$(all_but 0) drop_8:$(all_but 7) blocks_1-4:0,1,2,3 blocks_5-8:4,5,6,7; do
        D="${EXP}/test/blocks/${C%%:*}/seed_${SEED}"
        copy_origin "${EXP}/test/ours/seed_${SEED}/origin" "$D"
        apply "$V" "${TEST}" "$D" --strength_img "${ALPHA}" --block_steering "${C#*:}" --seed "${SEED}"
      done
      D="${EXP}/test/blocks/steps_3-4/seed_${SEED}"
      copy_origin "${EXP}/test/ours/seed_${SEED}/origin" "$D"
      apply "$V" "${TEST}" "$D" --strength_img "${ALPHA}" --t_steering 2,3 --seed "${SEED}"
    done
  fi

  if has_stage shift_sweep && [[ -d "${EXP}/text/vectors" ]]; then
    for A in ${ALPHAS}; do
      D="${EXP}/val/shift_alpha_${A}"
      copy_origin "${EXP}/val/alpha_2/origin" "$D"
      apply "${EXP}/text/vectors" "${VAL}" "$D" --strength_img "${A}" --steering_type separate --seed 42
      score "$S/val_shift_alpha_${A}.json" "$D/steered" "$D/origin" "${VAL}"
    done
    A_RULE="$("${PYTHON}" scripts/select_alpha.py "$S" --prefix val_shift_alpha_ --tau "${TAU:-0.85}" \
        --fallback --note "${EXP}/shift_rule_alpha_note.txt")"
    echo "${A_RULE}" > "${EXP}/shift_rule_alpha.txt"; echo "  SHIFT rule alpha ${STEM}: ${A_RULE}"
    for SEED in ${SEEDS}; do
      D="${EXP}/test/shift_rule/seed_${SEED}"
      copy_origin "${EXP}/test/ours/seed_${SEED}/origin" "$D"
      apply "${EXP}/text/vectors" "${TEST}" "$D" --strength_img "${A_RULE}" --steering_type separate --seed "${SEED}"
    done
  fi

  if has_stage score; then
    for D in "${EXP}"/val/angle_*/*; do
      score "$S/val_$(basename "$(dirname "$D")")_$(basename "$D").json" "$D/steered" "$D/origin" "${VAL}"
    done
    for D in "${EXP}"/test/angle_*/*; do
      score "$S/test_$(basename "$(dirname "$D")")_$(basename "$D")_seed_42.json" "$D/steered" "$D/origin" "${TEST}"
    done
    for D in "${EXP}"/test/blocks/*/seed_*; do
      C="$(basename "$(dirname "$D")")"; SEED="${D##*seed_}"
      score "$S/test_blk_${C}_seed_${SEED}.json" "$D/steered" "$D/origin" "${TEST}"
    done
    for D in "${EXP}"/test/shift_rule/seed_*; do
      score "$S/shift_rule_seed_${D##*seed_}.json" "$D/steered" "$D/origin" "${TEST}"
    done
  fi

  if has_stage examples; then
    X="paper_results_revision3/examples/${STEM}"; mkdir -p "$X"
    for I in ${EXAMPLE_IDX}; do
      P="$(printf '%02d' "$I")"
      for pair in t2i:test/ours/seed_42/origin ours:test/ours/seed_42/steered shift:test/shift_text/seed_42/steered \
                  shift_rule:test/shift_rule/seed_42/steered teacher:test/teacher/seed_42/i2i \
                  ours_rule:test/ours_rule/seed_42/steered; do
        f="$(ls "${EXP}/${pair#*:}/${I}_"*.png "${EXP}/${pair#*:}/${P}_"*.png 2>/dev/null | head -1 || true)"
        [[ -n "$f" ]] && cp "$f" "$X/${P}__${pair%%:*}.png"
      done
    done
    cp -f "${REF}" "$X/reference.${REF##*.}"
  fi
done

if has_stage analysis; then
  log analysis
  "${PYTHON}" scripts/revision3_analysis.py --exp_root "${EXP_ROOT}" --alpha "${ALPHA}" --out paper_results_revision3
  "${PYTHON}" scripts/make_limits_figure.py --examples paper_results_revision3/examples \
      --out paper_results_revision3/figs || true
  "${PYTHON}" -c "import shutil; shutil.make_archive('paper_results_revision3', 'zip', 'paper_results_revision3')"
  echo "DONE: send paper_results_revision3.zip"
fi
