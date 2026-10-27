#!/usr/bin/env bash
# Run under a persistent user service; never resume/overwrite a partial trajectory.
# Dynamics, initialization and thermostat are the unmodified upstream Forecast.
set -euo pipefail
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
RUN=/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_based/quartz_20261027
TAG="${1:?Provide a new trajectory tag}"
[[ "$TAG" =~ ^official30_15ns_[a-zA-Z0-9_]+$ ]] || exit 2
if [[ -e "$RUN/$TAG" ]]; then
    echo "Refusing to overwrite existing run: $RUN/$TAG" >&2
    exit 2
fi
exec > "$RUN/${TAG}.log" 2>&1
"$PY" -m trajcast_based.official forecast --production-steps 500000 --warmup-steps 50000 --tag "$TAG"
"$PY" -m trajcast_based.analyze_official --tag "$TAG"
"$PY" -m trajcast_based.publish --push
