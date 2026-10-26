#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data1/wyd/research/foundation_mlip_adaptation}
PYTHON=${PYTHON:-/data1/wyd/envs/mlip/bin/python}
OUT=${OUT:-$ROOT/experiments/neural_segment/quartz16_20261026_v1}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$ROOT/smdp"
set +e
"$PYTHON" -u -m neural_segment.run --data-root "$ROOT/data/smdp_reference_md_v1" --output "$OUT"
result=$?
echo "NEURAL_SEGMENT_PIPELINE_EXIT_CODE=$result"
"$PYTHON" -m neural_segment.report --output "$OUT" --push
publish=$?
echo "REPORT_PUBLISH_EXIT_CODE=$publish"
if [[ "$result" -ne 0 ]]; then exit "$result"; fi
exit "$publish"
