#!/usr/bin/env bash
set -euo pipefail
repo=/data1/wyd/research/foundation_mlip_adaptation/smdp
export PYTHONPATH="$repo:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107"
export LD_LIBRARY_PATH="/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$repo"
system=$1
queue=$2
root=/data1/wyd/research/foundation_mlip_adaptation/experiments/conditional_transition_20261109/$system
exec 9>"$root/queue_$queue.lock"
flock -n 9 || { echo 'Queue already running' >&2; exit 1; }
echo $$ > "$root/queue_$queue.pid"
if [[ "$queue" == method ]]; then arms=(autonomous extra_reference); else arms=(two_step original); fi
for arm in "${arms[@]}"; do
    /data1/wyd/envs/mlip/bin/python -u -m conditional_transition.pipeline --run "$root/$arm"
done
