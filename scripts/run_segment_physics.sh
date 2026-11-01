#!/usr/bin/env bash
set -euo pipefail
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
ANALYSIS_PY=/data1/wyd/envs/mlip/bin/python
RUN=/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_segment_physics_20261101
exec >> "$RUN/pipeline.log" 2>&1
publish() { "$PY" -m trajcast_smdp.publish --run "$RUN" --push || true; }
trap publish EXIT
if [[ ! -f "$RUN/test_metrics.json" ]]; then
    "$PY" -m trajcast_smdp.evaluate --run "$RUN"
fi
for tag in nve_0 nve_1 nvt_0; do
    if [[ ! -f "$RUN/$tag/run.json" ]]; then
        "$PY" -m trajcast_smdp.rollout --run "$RUN" --ensemble "${tag%_*}" --replica "${tag##*_}"
    fi
    LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib \
        "$ANALYSIS_PY" -m trajcast_smdp.analyze --run "$RUN" --tag "$tag"
    publish
done
