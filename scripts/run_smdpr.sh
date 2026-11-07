#!/usr/bin/env bash
set -euo pipefail
repo=/data1/wyd/research/foundation_mlip_adaptation/smdp
project=/data1/wyd/research/foundation_mlip_adaptation
export PYTHONPATH="$repo:$project/upstream/trajcast-c52f107${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo"
python_bin=/data1/wyd/envs/mlip/bin/python
run=$1
shift
"$python_bin" -u -m smdpr.train --run "$run" "$@"
