#!/usr/bin/env bash
set -euo pipefail
repo=/data1/wyd/research/foundation_mlip_adaptation/smdp
root=/data1/wyd/research/foundation_mlip_adaptation/experiments/dynamics_diagnosis_20261109
system=$1
mkdir -p "$root/$system"
exec 9>"$root/$system/run.lock"
flock -n 9 || exit 1
echo $$ > "$root/$system/run.pid"
export PYTHONPATH="$repo:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107"
export LD_LIBRARY_PATH="/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/data1/wyd/envs/mlip/bin/python
cd "$repo"
if [[ ! -f "reports/dynamics_diagnosis_20261109/${system}_implementation_audit.json" ]]; then
    "$python_bin" -u -m dynamics_diagnosis.audit --system "$system"
fi
"$python_bin" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' \
    "reports/dynamics_diagnosis_20261109/${system}_implementation_audit.json"
"$python_bin" -u -m dynamics_diagnosis.experiment --system "$system"
