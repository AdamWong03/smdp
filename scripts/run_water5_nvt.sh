#!/usr/bin/env bash
set -euo pipefail
project=/data1/wyd/research/foundation_mlip_adaptation
run=$project/experiments/relational_transition/water5_20261103_r3_opt
data=$project/data/smdp_reference_md_v1/water
python_bin=/data1/wyd/envs/mlip/bin/python
export PYTHONPATH=$project/upstream/trajcast-c52f107
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
out=$run/physics/nvt_replica_01
mkdir -p "$out"
if [[ ! -e "$out/trajectory.h5" ]]; then
    "$python_bin" -u rollout_relational.py \
        --checkpoint "$run/best.pt" \
        --initial "$data/raw_nve/test_05/trajectory.h5" \
        --output "$out/trajectory.h5" \
        --ensemble nvt --steps 500000 --thermalization-steps 50000 \
        --thermostat-metadata "$data/physical_reference/nvt_2p5ns/metadata.json" \
        --seed 2026110401 \
        --execution-note 'Shared L40S; other GPU workloads present at launch; efficiency is not an exclusive-GPU benchmark' \
        > "$out/rollout.log" 2>&1
elif [[ ! -e "$out/trajectory.summary.json" ]]; then
    echo 'Existing trajectory has no completion summary; inspect before resuming' >&2
    exit 1
fi
"$python_bin" -u analyze_water.py \
    --candidate "$out/trajectory.h5" \
    --reference "$data/physical_reference/nvt_2p5ns/production.h5" \
    --topology "$data/equilibration/snapshots/water_5.data" \
    --output "$out/analysis" \
    --tasks motion energy rdf vdos thermostat diffusion \
    > "$out/analysis.log" 2>&1
