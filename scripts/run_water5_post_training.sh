#!/usr/bin/env bash
set -euo pipefail

project=/data1/wyd/research/foundation_mlip_adaptation
repo=$project/smdp
run=$project/experiments/relational_transition/water5_20261103_r3_opt
data=$project/data/smdp_reference_md_v1/water
python_bin=/data1/wyd/envs/mlip/bin/python
export PYTHONPATH=$project/upstream/trajcast-c52f107
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

mkdir -p "$run/evaluation" "$run/physics/nve_test_05"

if [[ ! -f "$run/evaluation/multistep_test.json" ]]; then
    "$python_bin" -u "$repo/evaluate_relational_rollout.py" \
        --checkpoint "$run/best.pt" \
        --transitions "$data/transitions/5fs/test.h5" \
        --output "$run/evaluation/multistep_test.json" \
        --batch-size 8 \
        > "$run/evaluation/multistep_test.log" 2>&1
fi

if [[ ! -f "$run/physics/nve_test_05/trajectory.h5" ]]; then
    "$python_bin" -u "$repo/rollout_relational.py" \
        --checkpoint "$run/best.pt" \
        --initial "$data/raw_nve/test_05/trajectory.h5" \
        --output "$run/physics/nve_test_05/trajectory.h5" \
        --ensemble nve --steps 20000 \
        > "$run/physics/nve_test_05/rollout.log" 2>&1
fi
