#!/usr/bin/env bash
# Frozen best checkpoints, same validation initial state; no model selection.
set -euo pipefail
ROOT=/data1/wyd/research/foundation_mlip_adaptation
cd "$ROOT/smdp"
export PYTHONPATH="$PWD:$ROOT/upstream/trajcast-c52f107"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/data1/wyd/envs/mlip/bin/python
BASE="$ROOT/experiments/water6_reference_20261113"
DATA="$ROOT/data/smdp_reference_md_v1/water"
exec 9>"$BASE/validation_physics.lock"
flock -n 9 || exit 1
for arm in dsrt_radius6 dsrt_radius4p5_historical; do
  if [ "$arm" = dsrt_radius6 ]; then
    CHECKPOINT="$BASE/dsrt_radius6/best.pt"
  else
    CHECKPOINT="$ROOT/experiments/relational_transition/water5_20261103_r3_opt/best.pt"
  fi
  OUT="$BASE/$arm/validation_nve_20ps"
  mkdir -p "$OUT"
  if [ ! -e "$OUT/trajectory.h5" ]; then
    "$PY" -u rollout_relational.py --checkpoint "$CHECKPOINT" \
      --initial "$DATA/raw_nve/val_04/trajectory.h5" --initial-frame 0 \
      --output "$OUT/trajectory.h5" --ensemble nve --steps 4000 \
      --execution-note 'Shared L40S; validation source val_04 frame 0; frozen validation endpoint best; not checkpoint selection' \
      > "$OUT/rollout.log" 2>&1
  fi
  test -e "$OUT/trajectory.summary.json" || { echo 'Incomplete trajectory: inspect, no overwrite'; exit 1; }
  "$PY" -u analyze_water.py --candidate "$OUT/trajectory.h5" \
    --reference "$DATA/raw_nve/val_04/trajectory.h5" \
    --topology "$DATA/equilibration/snapshots/water_5.data" \
    --output "$OUT/analysis" --tasks motion energy full_energy rdf vdos diffusion \
    --frames 2000 --model-label "$arm" > "$OUT/analysis.log" 2>&1
  mkdir -p "$BASE/$arm/evaluation"
  if [ ! -f "$BASE/$arm/evaluation/val_multistep.json" ]; then
    "$PY" -u evaluate_relational_rollout.py --checkpoint "$CHECKPOINT" \
      --transitions "$DATA/transitions/5fs/val.h5" \
      --output "$BASE/$arm/evaluation/val_multistep.json" --batch-size 8 \
      > "$BASE/$arm/evaluation/val_multistep.log" 2>&1
  fi
done
