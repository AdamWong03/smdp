#!/usr/bin/env bash
# One fixed SMDP experiment, no baselines, no automatic architecture changes.
set -euo pipefail
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
RUN=/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028
mkdir -p "$RUN"
exec >> "$RUN/pipeline.log" 2>&1
publish_exit() { "$PY" -m trajcast_smdp.publish --push || true; }
trap publish_exit EXIT
if [[ ! -f "$RUN/config.yaml" ]]; then "$PY" -m trajcast_smdp.data; fi
if [[ ! -f "$RUN/correctness.json" ]]; then "$PY" -m trajcast_smdp.check; fi
if [[ ! -f "$RUN/training_complete.json" ]]; then "$PY" -m trajcast_smdp.train; fi
if [[ ! -f "$RUN/test_metrics.json" ]]; then "$PY" -m trajcast_smdp.evaluate; fi
for ensemble in nve nvt; do
    replicas=0
    if [[ "$ensemble" == nve ]]; then replicas='0 1'; fi
    for replica in $replicas; do
        tag="${ensemble}_${replica}"
        if [[ ! -f "$RUN/$tag/run.json" ]]; then
            "$PY" -m trajcast_smdp.rollout --ensemble "$ensemble" --replica "$replica"
        fi
        LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib \
          /data1/wyd/envs/mlip/bin/python -m trajcast_smdp.analyze --tag "$tag"
    done
done
