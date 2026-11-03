#!/usr/bin/env bash
set -euo pipefail

project=/data1/wyd/research/foundation_mlip_adaptation
run=$project/experiments/relational_transition/quartz30_20261102_r1/physics
checkpoint=$project/experiments/relational_transition/quartz30_20261102_r1/best.pt
python_bin=/data1/wyd/envs/mlip/bin/python
export PYTHONPATH=$project/upstream/trajcast-c52f107
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for replica in 09 10; do
    output=$run/nve_test_$replica
    reference=$project/data/smdp_reference_md_v1/quartz/raw_nve/test_$replica/trajectory.h5
    mkdir -p "$output"
    "$python_bin" -u rollout_relational.py \
        --checkpoint "$checkpoint" \
        --initial "$reference" \
        --output "$output/trajectory.h5" \
        --ensemble nve --steps 3333 \
        > "$output/rollout.log" 2>&1
    "$python_bin" -u analysis_tasks.py \
        --candidate "$output/trajectory.h5" \
        --reference "$reference" \
        --output "$output/analysis" \
        --tasks thermo rdf energy vdos_macro \
        > "$output/analysis.log" 2>&1
done
