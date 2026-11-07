#!/usr/bin/env bash
# Wait for a real completed primary run, then execute the frozen protocol.
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
unit=$2
control=$3
while ! "$python_bin" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1])/"status.json"; sys.exit(0 if p.exists() and json.loads(p.read_text()).get("state")=="endpoint_training_completed" else 1)' "$run"; do
    if ! systemctl --user is-active --quiet "$unit"; then
        echo 'Primary training is not active and not completed; preserve state and stop follow-up.' >&2
        exit 1
    fi
    sleep 30
done
"$python_bin" -u -m smdpr.pipeline --run "$run" --baseline-control
if ! "$python_bin" -c 'import json,sys; s=json.load(open(sys.argv[1]+"/pipeline_status.json"))["state"]; sys.exit(0 if s in {"completed","completed_with_physical_failure"} else 1)' "$run"; then
    echo 'K3 failed a protective gate; K1 queue left unstarted.' >&2
    exit 0
fi
if [[ ! -f "$control/status.json" ]] || ! "$python_bin" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["state"]=="endpoint_training_completed" else 1)' "$control/status.json"; then
    resume=()
    if [[ -f "$control/last.pt" ]]; then resume=(--resume); fi
    "$python_bin" -u -m smdpr.train --run "$control" "${resume[@]}"
fi
"$python_bin" -u -m smdpr.pipeline --run "$control"
