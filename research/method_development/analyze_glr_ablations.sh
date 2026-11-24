#!/usr/bin/env bash
# Analyze the 100 ps GLR ablation runs (motion, RDF, VDOS).
set -euo pipefail

ROOT=/data2/wyd/research/foundation_mlip_adaptation
DATA=/data1/wyd/research/foundation_mlip_adaptation
cd "$ROOT/code/h1h2"
export PYTHONPATH=$DATA/upstream/trajcast-c52f107:$PWD
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for name in "$@"; do
  run=$ROOT/experiments/${name}_20261122
  if [ ! -f "$run/trajectory.h5" ]; then
    echo "missing $run/trajectory.h5"
    continue
  fi
  /data1/wyd/envs/mlip/bin/python -u analyze_water.py \
    --candidate "$run/trajectory.h5" \
    --reference "$DATA/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5" \
    --topology "$DATA/data/smdp_reference_md_v1/water/equilibration/snapshots/water_5.data" \
    --output "$run/analysis" --tasks motion rdf vdos --frames 2000 \
    --model-label "$name"
done
