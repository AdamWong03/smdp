#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/wyd/research/foundation_mlip_adaptation
cd "$ROOT/smdp"
export PYTHONPATH="$PWD:$ROOT/upstream/trajcast-c52f107"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
RUN="$ROOT/experiments/water6_reference_20261113/trajcast_native/interim_epoch75"
DATA="$ROOT/data/smdp_reference_md_v1/water"
exec 9>"$RUN/diagnosis.lock"
flock -n 9 || exit 1
/data1/wyd/envs/trajcast-c52f107/bin/python -u -m water_reference.diagnose_native_nve \
  --checkpoint "$RUN/snapshot.pt" --initial "$DATA/raw_nve/val_04/trajectory.h5" \
  --output "$RUN/trajectory.h5" > "$RUN/rollout.log" 2>&1
/data1/wyd/envs/mlip/bin/python -u analyze_water.py \
  --candidate "$RUN/trajectory.h5" --reference "$DATA/raw_nve/val_04/trajectory.h5" \
  --topology "$DATA/equilibration/snapshots/water_5.data" --output "$RUN/analysis" \
  --frames 2000 --model-label 'TrajCast interim epoch75' --tasks motion rdf vdos diffusion \
  > "$RUN/analysis.log" 2>&1
