#!/bin/bash
# Full from-scratch run of every experiment of revision 2, in the order the stages depend on
# each other. Run from the repository root on a GPU with >= 48 GB.
#
#   bash scripts/run_revision2_all.sh                      # all 7 references, then twins + analysis
#   REFS="data/reference_images/open/degas_jockey.jpg data/reference_images/open/seurat_grande_jatte.jpg" \
#     CUDA_VISIBLE_DEVICES=1 SKIP_FINAL=1 bash scripts/run_revision2_all.sh   # a subset on another GPU
#
# Every stage skips finished work, so the script can simply be restarted after a crash.
# Logs: logs/revision2_<phase>.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
export PYTHON="${PYTHON:-python}"
export ALPHA="${ALPHA:-6}"
[[ -n "${REFS:-}" ]] && export REFS
run() { local name="$1"; shift; echo "=== $(date '+%F %T') ${name}"; "$@" 2>&1 | tee -a "logs/revision2_${name}.log"; }

# ---- 0. checks -----------------------------------------------------------------
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON}" - <<'PY'
import torch, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "diffusers", diffusers.__version__)
from diffusers import Flux2KleinPipeline  # needs diffusers >= 0.37
for m in ("lpips", "piq", "torchmetrics", "scipy"):
    __import__(m)
PY
if [[ -z "${CSD_CKPT:-}" ]]; then
  [[ -f weights/csd/pytorch_model.bin ]] && export CSD_CKPT=weights/csd/pytorch_model.bin \
    || { echo "CSD checkpoint missing: see REVISION2_RUNBOOK.md, step 0"; exit 1; }
fi
export PYTHONPATH="$PWD/third_party/CSD:$PWD/third_party/CSD/models:${PYTHONPATH:-}"
ls data/reference_images/open/*.jpg >/dev/null 2>&1 || run download "${PYTHON}" scripts/download_open_references.py

# ---- 1. generation that does not depend on scores --------------------------------
run gen1 env STAGES="extract textbase sweep nameprobe ablation blocks test" bash scripts/paper_klein_experiments.sh
# ---- 2. scores (validation scores are needed for the rule-selected alpha / strengths)
run score1 env STAGES="score" bash scripts/paper_klein_experiments.sh
# ---- 3. rule-alpha test and baselines (Linear-AcT, CASteer/SDXL, IP-Adapter/FLUX.1-dev)
run gen2 env STAGES="test_rule baselines" bash scripts/paper_klein_experiments.sh
run score2 env STAGES="score" bash scripts/paper_klein_experiments.sh

[[ "${SKIP_FINAL:-0}" == "1" ]] && { echo "SKIP_FINAL=1: twins and analysis skipped"; exit 0; }

# ---- 4. same-style second references (stability check)
ls data/reference_images/twins/*.jpg >/dev/null 2>&1 || run download "${PYTHON}" scripts/download_open_references.py --set twins
run twins env -u REFS REF_DIR=data/reference_images/twins EXP_ROOT=experiments/klein_9b/twins SEEDS=42 \
    STAGES="extract test score" bash scripts/paper_klein_experiments.sh

# ---- 5. CPU analysis -> paper_results/
run analysis bash -c '
  set -e
  $PYTHON scripts/collect_paper_results.py --alpha $ALPHA
  $PYTHON scripts/collect_paper_results.py --exp_root experiments/klein_9b/twins --out paper_results/twins --alpha $ALPHA
  $PYTHON scripts/paper_stats.py --alpha $ALPHA
  $PYTHON scripts/block_analysis.py --alpha $ALPHA
  $PYTHON scripts/vector_stability.py --twin_root experiments/klein_9b/twins
  for d in experiments/klein_9b/paper/*/; do
    echo "$(basename $d): gram $($PYTHON scripts/select_alpha.py $d/scores) / csd $($PYTHON scripts/select_alpha.py $d/scores --metric csd)"
  done > paper_results/rule_alpha.txt
  nvidia-smi --query-gpu=name,memory.total --format=csv > paper_results/gpu.txt
  pip freeze > paper_results/pip_freeze.txt'
# figure inputs for the qualitative comparison with baselines (prompts 0 and 10, seed 42)
mkdir -p paper_results/baseline_examples
for d in experiments/klein_9b/paper/*/; do
  n=$(basename "$d")
  for m in act/test/seed_42/steered casteer/test/seed_42/s_* ipadapter/test/seed_42/s_*; do
    for f in "$d"baselines/$m/00_* "$d"baselines/$m/10_*; do
      [[ -f "$f" ]] && cp "$f" "paper_results/baseline_examples/${n}__${m%%/*}__$(basename "$f")"
    done
  done
done
"${PYTHON}" -c "import shutil; shutil.make_archive('paper_results_revision2', 'zip', 'paper_results')"
echo "DONE: send paper_results_revision2.zip"
