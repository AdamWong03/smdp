#!/usr/bin/env bash
# Fixed selected R3-B checkpoint. No training or architecture changes.
set -euo pipefail
ROOT="${ROOT:-/data1/wyd/research/foundation_mlip_adaptation}"
PYTHON="${PYTHON:-/data1/wyd/envs/mlip/bin/python}"
export LD_LIBRARY_PATH="/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$ROOT/smdp"
"$PYTHON" -u run_experiment.py --after-training \
  --data-root "$ROOT/data/smdp_reference_md_v1" \
  --output "$ROOT/experiments/motion_modes/quartz16_r3b_spatial_composition" \
  --previous "$ROOT/experiments/motion_modes/quartz16_20261022_r2_additive" \
  --r1 "$ROOT/experiments/motion_modes/quartz16_20261022_r1" \
  --spectral-workers 6 --publish-summary --report-name quartz16_r3b_physics
