#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=/data1/wyd/research/foundation_mlip_adaptation
RUN="$ROOT/experiments/future_relation_control_20261117/current_only"
FULL="$ROOT/experiments/water6_reference_20261113/dsrt_radius6"
DATA="$ROOT/data/smdp_reference_md_v1/water"
PY=/data1/wyd/envs/mlip/bin/python
cd "$PROJECT"
export PYTHONPATH="$PROJECT:$ROOT/upstream/trajcast-c52f107"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONWARNINGS=ignore::FutureWarning
mkdir -p "$RUN"
exec 9>"$RUN/run.lock"
flock -n 9 || { echo 'Current-Only is already running'; exit 1; }
RESUME=()
if [ "${1:-}" = --resume ]; then
  RESUME=(--resume)
elif [ "$#" -ne 0 ]; then
  echo 'Only --resume is supported'; exit 2
fi
if [ -e "$RUN/config.json" ]; then
  cmp configs/water6_current_only_20261117.json "$RUN/config.json"
else
  cp configs/water6_current_only_20261117.json "$RUN/config.json"
fi
if [ -e "$RUN/last.pt" ] && [ "${1:-}" != --resume ]; then
  echo 'Refusing overwrite: use --resume'; exit 1
fi
printf '%s\n' "$$" > "$RUN/launcher.pid"
if [ ! -e "$RUN/source_commit.txt" ]; then
  git rev-parse HEAD > "$RUN/source_commit.txt"
  "$PY" -m pip freeze > "$RUN/environment.txt"
  nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv > "$RUN/gpu_at_launch.csv"
fi
stage() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$1" >> "$RUN/pipeline_stages.txt"
  printf '%s\n' "$1" > "$RUN/pipeline_stage.txt"
}
on_exit() {
  local code=$?
  if [ "$code" -ne 0 ]; then
    printf 'exit_code=%s time=%s\n' "$code" "$(date --iso-8601=seconds)" > "$RUN/pipeline_failure.txt"
    "$PY" scripts/export_future_relation_control.py || true
  fi
}
trap on_exit EXIT
stage correctness
"$PY" -u -m water_reference.check_relation_control --run "$RUN" --baseline-run "$FULL" \
  >> "$RUN/correctness.log" 2>&1
if [ ! -e "$RUN/test_metrics.json" ]; then
  stage formal_training
  "$PY" -u train_relational.py --run "$RUN" "${RESUME[@]}" >> "$RUN/train.log" 2>&1
  # A graceful stop is not completion and must not launch evaluation.
  test -f "$RUN/test_metrics.json" || exit 1
fi
stage validation_multistep
mkdir -p "$RUN/evaluation"
if [ ! -e "$RUN/evaluation/val_multistep.json" ]; then
  "$PY" -u evaluate_relational_rollout.py --checkpoint "$RUN/best.pt" \
    --transitions "$DATA/transitions/5fs/val.h5" --batch-size 8 \
    --output "$RUN/evaluation/val_multistep.json" > "$RUN/evaluation/val_multistep.log" 2>&1
fi
stage validation_nve
OUT="$RUN/validation_nve_20ps"
mkdir -p "$OUT"
if [ ! -e "$OUT/trajectory.h5" ]; then
  "$PY" -u rollout_relational.py --checkpoint "$RUN/best.pt" \
    --initial "$DATA/raw_nve/val_04/trajectory.h5" --initial-frame 0 \
    --output "$OUT/trajectory.h5" --ensemble nve --steps 4000 \
    --execution-note 'Shared GPU; same val_04/frame0 as frozen Full 6A; validation-selected checkpoint' \
    > "$OUT/rollout.log" 2>&1
fi
test -f "$OUT/trajectory.summary.json" || { echo 'Incomplete trajectory retained; inspect before retry'; exit 1; }
stage physical_analysis
FAILED=0
if ! "$PY" -u analyze_water.py --candidate "$OUT/trajectory.h5" \
  --reference "$DATA/raw_nve/val_04/trajectory.h5" \
  --topology "$DATA/equilibration/snapshots/water_5.data" \
  --output "$OUT/analysis" --tasks motion energy rdf vdos diffusion --frames 2000 \
  --model-label 'Current-Only 6A' >> "$OUT/analysis.log" 2>&1; then
  FAILED=1
  echo 'Physical analysis incomplete; failures retained, independent tasks continue'
fi
stage paired_inference_timing
if [ ! -e "$RUN/evaluation/paired_inference.json" ]; then
  "$PY" -u -m water_reference.time_relation_control --full "$FULL/best.pt" \
    --control "$RUN/best.pt" --output "$RUN/evaluation/paired_inference.json" \
    > "$RUN/evaluation/paired_inference.log" 2>&1
fi
if [ "$FAILED" = 0 ]; then stage requested_analysis_completed; else stage analysis_incomplete; fi
"$PY" scripts/export_future_relation_control.py
# Publish only this report, never unrelated staged work or large experiment data.
if [ "$(git branch --show-current)" = experiment/future-relation-control-20261117 ] && git diff --cached --quiet; then
  git add reports/future_relation_control_20261117
  if ! git diff --cached --quiet; then
    git commit -m 'Report completed Current-Only training and matched validation evaluation'
    git push origin HEAD
  fi
fi
exit "$FAILED"
