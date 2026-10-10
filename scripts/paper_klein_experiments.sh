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
#   nameprobe T2I of the VAL prompts + ", in the style of <artist>" (no reference, no steering): does the
#             model know the style by name? (name_tags.tsv from the downloader) -> val/name_probe
#   blocks    per-block analysis on VAL at ALPHA (BLOCK_MODES, default "only drop prefix"):
#             only_k = steer block k alone, drop_k = all blocks except k, first_k = blocks 1..k
#             -> val/blocks/<cfg>; analysed by scripts/block_analysis.py
#   test      final run on the TEST prompts with ALPHA, 3 seeds, + I2I teacher images
#             (teacher per seed in test/teacher/seed_<S>/i2i, same initial noise as the T2I
#              origin of test/ours/seed_<S>; the old unpaired test/teacher/i2i is kept)
#   test_rule test run with the alpha chosen by the paper's rule (scripts/select_alpha.py) on
#             the val sweep -> test/ours_rule/seed_<S>  (RULE_SEEDS, default SEEDS)
#   score     extended metrics (metrics/eval_klein_extended.py) and Wilcoxon comparison
#   baselines training-free baselines that need at most one image (BASELINES, default all three):
#             act        Linear-AcT on Klein from the same pairs (lambda sweep on VAL, test on SEEDS)
#             casteer    CASteer vectors on SDXL (text pairs; strength sweep on VAL, test seed 42)
#             ipadapter  IP-Adapter on FLUX.1-dev (one image; scale sweep on VAL, test seed 42)
#             Each strength is chosen by the same rule as alpha (scripts/select_alpha.py).
#
# Same-style stability check (second image of the same artist/style, see the downloader):
#   python scripts/download_open_references.py --set twins
#   REF_DIR=data/reference_images/twins EXP_ROOT=experiments/klein_9b/twins SEEDS=42 \
#     STAGES="extract test score" bash scripts/paper_klein_experiments.sh
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
STAGES="${STAGES:-extract textbase sweep nameprobe ablation blocks test test_rule score}"
TEACHER_SEEDS="${TEACHER_SEEDS:-${SEEDS}}"
RULE_SEEDS="${RULE_SEEDS:-${SEEDS}}"
BASELINES="${BASELINES:-act casteer ipadapter}"
ACT_LAMS="${ACT_LAMS:-0.25 0.5 0.75 1}"
CASTEER_S="${CASTEER_S:-0.05 0.1 0.2 0.4}"
IP_SCALES="${IP_SCALES:-0.1 0.2 0.3 0.5 0.8 1.0}"
EXT_SEEDS="${EXT_SEEDS:-42}"
BLOCK_MODES="${BLOCK_MODES:-only drop prefix}"
EXP_ROOT="${EXP_ROOT:-experiments/klein_9b/paper}"
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

# The unsteered T2I image depends only on (prompt, seed), not on the vector or the strength, so
# it is generated once per (prompt set, seed) and copied into the other run folders.
reuse_origin() {  # <src_run_dir> <dst_run_dir>
  if [[ -d "$1/origin" && ! -d "$2/origin" ]]; then mkdir -p "$2"; cp -r "$1/origin" "$2/origin"; fi
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
  EXP="${EXP_ROOT}/${STEM}"
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
    FIRST_A=""
    for A in ${ALPHAS}; do
      [[ -n "${FIRST_A}" ]] && reuse_origin "${EXP}/val/alpha_${FIRST_A}" "${EXP}/val/alpha_${A}"
      [[ -z "${FIRST_A}" ]] && FIRST_A="${A}"
      apply "${EXP}/ref/vectors" "${VAL}" "${EXP}/val/alpha_${A}" "${EXP}/val/alpha_${A}/stats.json" \
          --strength_img "${A}" --steering_type separate --seed 42 --log_norms
    done
  fi

  if has_stage ablation; then
    V="${EXP}/ref/vectors"; O="${EXP}/val/ablation"
    for C in blocks_0-3 blocks_4-7 steps_0 steps_0-1 steps_2-3 pooled with_txt svm; do
      reuse_origin "${EXP}/val/alpha_2" "$O/$C"
    done
    apply "$V" "${VAL}" "$O/blocks_0-3"  "$O/blocks_0-3/stats.json"  --strength_img "${ALPHA}" --block_steering 0,1,2,3 --seed 42
    apply "$V" "${VAL}" "$O/blocks_4-7"  "$O/blocks_4-7/stats.json"  --strength_img "${ALPHA}" --block_steering 4,5,6,7 --seed 42
    apply "$V" "${VAL}" "$O/steps_0"     "$O/steps_0/stats.json"     --strength_img "${ALPHA}" --t_steering 0 --seed 42
    apply "$V" "${VAL}" "$O/steps_0-1"   "$O/steps_0-1/stats.json"   --strength_img "${ALPHA}" --t_steering 0,1 --seed 42
    apply "$V" "${VAL}" "$O/steps_2-3"   "$O/steps_2-3/stats.json"   --strength_img "${ALPHA}" --t_steering 2,3 --seed 42
    apply "$V" "${VAL}" "$O/pooled"      "$O/pooled/stats.json"      --strength_img "${ALPHA}" --steering_type mean --seed 42
    apply "$V" "${VAL}" "$O/with_txt"    "$O/with_txt/stats.json"    --strength_img "${ALPHA}" --strength "${ALPHA_TXT:-6}" --seed 42
    apply "$V" "${VAL}" "$O/svm"         "$O/svm/stats.json"         --strength_img "${ALPHA}" --use_cls --seed 42
  fi

  if has_stage nameprobe; then
    NT="$(awk -F'\t' -v s="${STEM}" '$1==s {print $2}' "${REF_DIR}/name_tags.tsv" 2>/dev/null || true)"
    if [[ -z "${NT}" ]]; then
      echo "  nameprobe: no name tag for ${STEM}; skipped"
    else
      D="${EXP}/val/name_probe"; mkdir -p "$D"
      awk -v t="${NT}" 'NF {print $0 t}' "${VAL}" > "$D/prompts.txt"
      if images_complete "$D/steered" "$D/prompts.txt"; then skip "$D"; else
        "${PYTHON}" ./src/steering/apply_steering_klein.py --model_name "${MODEL_NAME}" \
            --data_dir "${EXP}/ref/vectors" --prompts_path "$D/prompts.txt" --task "add concept" \
            --inference_steps 4 --guidance_scale 1.0 --width 1024 --height 1024 \
            --strength 0 --strength_img 0 --results_dir "$D" --seed 42
      fi
    fi
  fi

  if has_stage blocks; then
    V="${EXP}/ref/vectors"; O="${EXP}/val/blocks"
    run_blocks() {  # <cfg_name> <comma-separated 0-based blocks>
      local D="$O/$1"
      [[ -d "$D/origin" ]] || { mkdir -p "$D"; cp -r "${EXP}/val/alpha_2/origin" "$D/origin"; }
      apply "$V" "${VAL}" "$D" "$D/stats.json" --strength_img "${ALPHA}" --block_steering "$2" --seed 42
    }
    for K in 0 1 2 3 4 5 6 7; do
      [[ " ${BLOCK_MODES} " == *" only "* ]] && run_blocks "only_$((K + 1))" "$K"
      if [[ " ${BLOCK_MODES} " == *" drop "* ]]; then
        run_blocks "drop_$((K + 1))" "$(seq -s, 0 7 | tr ',' '\n' | grep -vx "$K" | paste -sd, -)"
      fi
      # first_1 = only_1, first_4 = ablation blocks_0-3, first_8 = main run: not repeated
      if [[ " ${BLOCK_MODES} " == *" prefix "* ]] && (( K >= 1 && K != 3 && K != 7 )); then
        run_blocks "first_$((K + 1))" "$(seq -s, 0 "$K")"
      fi
    done
  fi

  if has_stage test; then
    for SEED in ${SEEDS}; do
      apply "${EXP}/ref/vectors" "${TEST}" "${EXP}/test/ours/seed_${SEED}" "${EXP}/test/ours/seed_${SEED}/stats.json" \
          --strength_img "${ALPHA}" --steering_type separate --seed "${SEED}"
      if [[ -d "${EXP}/text/vectors" ]]; then
        reuse_origin "${EXP}/test/ours/seed_${SEED}" "${EXP}/test/shift_text/seed_${SEED}"
        apply "${EXP}/text/vectors" "${TEST}" "${EXP}/test/shift_text/seed_${SEED}" \
            "${EXP}/test/shift_text/seed_${SEED}/stats.json" \
            --strength_img "${ALPHA_TEXTBASE:-${ALPHA}}" --steering_type separate --seed "${SEED}"
      fi
    done
    # Teacher (I2I) with seed S + prompt index = the T2I origin of test/ours/seed_S, so content
    # preservation of the teacher (DINOv2 / LPIPS / DISTS to origin) is measured on a common noise.
    for SEED in ${TEACHER_SEEDS}; do
      TD="${EXP}/test/teacher/seed_${SEED}"
      if images_complete "${TD}/i2i" "${TEST}"; then
        skip "${TD}"
      else
        "${PYTHON}" ./src/steering/get_vector_klein.py --model_name "${MODEL_NAME}" --exp_type style_ref \
            --prompt_path "${TEST}" --reference_image "${REF}" --height 1024 --width 1024 --gs 1.0 \
            --num_inference_steps 4 --batch_size 1 --save_image_dir "${TD}" --i2i_only --seed_base "${SEED}"
      fi
    done
  fi

  if has_stage test_rule; then
    if compgen -G "${EXP}/scores/val_alpha_*.json" >/dev/null; then
      A_RULE="$("${PYTHON}" scripts/select_alpha.py "${EXP}/scores" --tau "${TAU:-0.85}" \
          --fallback --note "${EXP}/rule_alpha_note.txt")"
      echo "  rule alpha for ${STEM}: ${A_RULE} ($(cut -f2 "${EXP}/rule_alpha_note.txt"))"
      echo "${A_RULE}" > "${EXP}/rule_alpha.txt"
      for SEED in ${RULE_SEEDS}; do
        RD="${EXP}/test/ours_rule/seed_${SEED}"
        # reuse the T2I origin images of the main test run (same prompts and seeds)
        if [[ -d "${EXP}/test/ours/seed_${SEED}/origin" && ! -d "${RD}/origin" ]]; then
          mkdir -p "${RD}"; cp -r "${EXP}/test/ours/seed_${SEED}/origin" "${RD}/origin"
        fi
        apply "${EXP}/ref/vectors" "${TEST}" "${RD}" "${RD}/stats.json" \
            --strength_img "${A_RULE}" --steering_type separate --seed "${SEED}"
      done
    else
      echo "  test_rule: no val scores for ${STEM}; run the sweep and score stages first"
    fi
  fi

  if has_stage baselines; then
    S="${EXP}/scores"; mkdir -p "$S"; B="${EXP}/baselines"
    CSD_B=(); [[ -n "${CSD_CKPT:-}" ]] && CSD_B=(--csd_ckpt "${CSD_CKPT}")
    bscore() {  # <out_json> <gen_dir> <origin_dir> <prompts>
      [[ -f "$1" && "${FORCE_SCORE:-0}" != "1" ]] && { echo "  keep $1"; return 0; }
      "${PYTHON}" metrics/eval_klein_extended.py score --out "$1" --gen_dir "$2" --origin_dir "$3" \
          --reference "${REF}" --prompts "$4" "${CSD_B[@]}"
    }
    has_b() { [[ " ${BASELINES} " == *" $1 "* ]]; }
    TAG="$(awk -F'\t' -v s="${STEM}" '$1==s {print $2}' "${STYLE_TAGS}" 2>/dev/null || true)"

    if has_b act; then   # ---- Linear-AcT on Klein, same pairs as the mean-difference vector
      [[ -f "$B/act/act_maps.pt" ]] || "${PYTHON}" src/steering/calculate_act_maps.py \
          --data_dir "${EXP}/ref/data_vectors" --out "$B/act/act_maps.pt"
      for L in ${ACT_LAMS}; do
        D="$B/act/val/lam_${L}"
        [[ -d "$D/origin" ]] || { mkdir -p "$D"; cp -r "${EXP}/val/alpha_2/origin" "$D/origin"; }
        apply "${EXP}/ref/vectors" "${VAL}" "$D" "$D/stats.json" --act_path "$B/act/act_maps.pt" \
            --strength_img "$L" --seed 42
        bscore "$S/val_act_${L}.json" "$D/steered" "$D/origin" "${VAL}"
      done
      L_RULE="$("${PYTHON}" scripts/select_alpha.py "$S" --prefix val_act_ --tau "${TAU:-0.85}" \
          --fallback --note "$B/act/rule_lambda.txt")"
      echo "  AcT lambda for ${STEM}: ${L_RULE} ($(cut -f2 "$B/act/rule_lambda.txt"))"
      for SEED in ${SEEDS}; do
        D="$B/act/test/seed_${SEED}"
        [[ -d "$D/origin" ]] || { mkdir -p "$D"; cp -r "${EXP}/test/ours/seed_${SEED}/origin" "$D/origin"; }
        apply "${EXP}/ref/vectors" "${TEST}" "$D" "$D/stats.json" --act_path "$B/act/act_maps.pt" \
            --strength_img "${L_RULE}" --seed "${SEED}"
        bscore "$S/act_seed_${SEED}.json" "$D/steered" "$D/origin" "${TEST}"
      done
    fi

    ext_baseline() {  # <method> <strengths> <extra generate args...>
      local m="$1" strengths="$2"; shift 2
      "${PYTHON}" src/baselines/external_baselines.py generate --method "$m" --prompts "${VAL}" \
          --strengths ${strengths} --seed 42 --out "$B/$m/val" "$@"
      for X in ${strengths}; do
        # folder names follow external_baselines.py: f"s_{strength:g}" (1.0 -> s_1)
        local XG; XG="$("${PYTHON}" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$X")"
        bscore "$S/val_${m}_${XG}.json" "$B/$m/val/s_${XG}" "$B/$m/val/origin" "${VAL}"
      done
      local X_RULE; X_RULE="$("${PYTHON}" scripts/select_alpha.py "$S" --prefix "val_${m}_" --tau "${TAU:-0.85}" \
          --fallback --note "$B/$m/rule_strength.txt")"
      echo "  ${m} strength for ${STEM}: ${X_RULE} ($(cut -f2 "$B/$m/rule_strength.txt"))"
      for SEED in ${EXT_SEEDS}; do
        "${PYTHON}" src/baselines/external_baselines.py generate --method "$m" --prompts "${TEST}" \
            --strengths "${X_RULE}" --seed "${SEED}" --out "$B/$m/test/seed_${SEED}" "$@"
        bscore "$S/${m}_seed_${SEED}.json" "$B/$m/test/seed_${SEED}/s_${X_RULE}" \
            "$B/$m/test/seed_${SEED}/origin" "${TEST}"
      done
    }
    if has_b casteer; then   # ---- CASteer vectors on SDXL (text pairs, no image)
      if [[ -z "${TAG}" ]]; then echo "  casteer: no style tag for ${STEM}; skipped"; else
        [[ -f "$B/casteer/vectors.pt" ]] || "${PYTHON}" src/baselines/external_baselines.py casteer_extract \
            --prompts "${TRAIN}" --tag "${TAG}" --out "$B/casteer/vectors.pt"
        ext_baseline casteer "${CASTEER_S}" --vectors "$B/casteer/vectors.pt"
      fi
    fi
    if has_b ipadapter; then  # ---- IP-Adapter on FLUX.1-dev (one image)
      ext_baseline ipadapter "${IP_SCALES}" --reference "${REF}" ${IP_ARGS:-}
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

    # per-block analysis runs and the unsteered validation images (T2I reference level)
    for D in "${EXP}"/val/blocks/*; do
      has_images "$D/steered" || continue
      score "$S/val_blk_$(basename "$D").json" --gen_dir "$D/steered" --origin_dir "$D/origin" \
          --reference "${REF}" --prompts "${VAL}"
    done
    has_images "${EXP}/val/name_probe/steered" && score "$S/val_nameprobe.json" \
        --gen_dir "${EXP}/val/name_probe/steered" --origin_dir "${EXP}/val/alpha_2/origin" \
        --reference "${REF}" --prompts "${VAL}"
    has_images "${EXP}/val/alpha_2/origin" && score "$S/val_t2i.json" --gen_dir "${EXP}/val/alpha_2/origin" \
        --reference "${REF}" --prompts "${VAL}"

    TEACHER="${EXP}/test/teacher/i2i"            # old unpaired teacher (FID target of v2)
    has_images "${TEACHER}" || TEACHER="${EXP}/test/teacher/seed_42/i2i"
    FID_ARGS=()
    has_images "${TEACHER}" && FID_ARGS=(--fid_target_dir "${TEACHER}")
    for SEED in ${SEEDS}; do
      for M in ours shift_text; do
        D="${EXP}/test/${M}/seed_${SEED}"
        has_images "$D/steered" || continue
        score "$S/${M}_seed_${SEED}.json" --gen_dir "$D/steered" --origin_dir "$D/origin" \
            --reference "${REF}" --prompts "${TEST}" "${FID_ARGS[@]}"
      done
      D="${EXP}/test/ours_rule/seed_${SEED}"
      has_images "$D/steered" && score "$S/ours_rule_seed_${SEED}.json" --gen_dir "$D/steered" \
          --origin_dir "${EXP}/test/ours/seed_${SEED}/origin" --reference "${REF}" --prompts "${TEST}" \
          "${FID_ARGS[@]}"
      D="${EXP}/test/ours/seed_${SEED}/origin"
      has_images "$D" && score "$S/t2i_seed_${SEED}.json" --gen_dir "$D" \
          --reference "${REF}" --prompts "${TEST}" "${FID_ARGS[@]}"
      # paired teacher: same noise as the origin above -> content metrics are defined
      D="${EXP}/test/teacher/seed_${SEED}/i2i"
      has_images "$D" && score "$S/teacher_seed_${SEED}.json" --gen_dir "$D" \
          --origin_dir "${EXP}/test/ours/seed_${SEED}/origin" --reference "${REF}" --prompts "${TEST}"
    done
    has_images "${EXP}/test/teacher/i2i" && score "$S/teacher.json" --gen_dir "${EXP}/test/teacher/i2i" \
        --reference "${REF}" --prompts "${TEST}"

    if [[ -f "$S/ours_seed_42.json" && -f "$S/t2i_seed_42.json" ]]; then
      RUNS=("$S/ours_seed_42.json" "$S/t2i_seed_42.json")
      if [[ -f "$S/teacher_seed_42.json" ]]; then RUNS+=("$S/teacher_seed_42.json")
      elif [[ -f "$S/teacher.json" ]]; then RUNS+=("$S/teacher.json"); fi
      [[ -f "$S/shift_text_seed_42.json" ]] && RUNS+=("$S/shift_text_seed_42.json")
      "${PYTHON}" metrics/eval_klein_extended.py compare --runs "${RUNS[@]}" --out "$S/compare_seed_42.json"
    fi
  fi
done

echo "Done. Results under ${EXP_ROOT}/"
