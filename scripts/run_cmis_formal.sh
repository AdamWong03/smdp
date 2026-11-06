#!/usr/bin/env bash
set -euo pipefail
system="$1"
project=/data1/wyd/research/foundation_mlip_adaptation
python_bin=/data1/wyd/envs/mlip/bin/python
case "$system" in
  quartz) name=quartz30 ;;
  water) name=water5 ;;
  *) exit 2 ;;
esac
run=$project/experiments/cmis/${name}_20261106_r1
export PYTHONPATH=$project/smdp:$project/upstream/trajcast-c52f107
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -f "$run/last.pt" ]]; then
  "$python_bin" -u -m cmis.train_formal --run "$run" --resume
else
  "$python_bin" -u -m cmis.train_formal --run "$run"
fi
"$python_bin" -u -m cmis.evaluate_formal --run "$run" --split val
"$python_bin" -u -m cmis.run_physics --run "$run" --phase validation
"$python_bin" -u -m cmis.evaluate_formal --run "$run" --split test
"$python_bin" -u -m cmis.run_physics --run "$run" --phase test
