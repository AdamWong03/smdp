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

"$python_bin" -u analyze_water.py \
    --candidate "$run/physics/nve_test_05/trajectory.h5" \
    --reference "$data/raw_nve/test_05/trajectory.h5" \
    --topology "$data/equilibration/snapshots/water_5.data" \
    --output "$run/physics/nve_test_05/analysis"

if [[ ! -e "$run/evaluation/signed_bias_deployed/results.json" ]]; then
    "$python_bin" -u diagnose_water_bias.py \
        --checkpoint "$run/best.pt" \
        --transitions "$data/transitions/5fs/test.h5" \
        --trajectory "$run/physics/nve_test_05/trajectory.h5" \
        --reference "$data/raw_nve/test_05/trajectory.h5" \
        --topology "$data/equilibration/snapshots/water_5.data" \
        --output "$run/evaluation/signed_bias_deployed"
fi
"$python_bin" scripts/export_water_diagnostics.py --run "$run" \
    --output reports/water5_20261103
