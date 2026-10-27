#!/usr/bin/env bash
# Upstream only. Does not launch baseline training or any SMDP modification.
set -euo pipefail
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
RUN=/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_based/quartz_20261027
"$PY" -m trajcast_based.check_interfaces
if ! test -f "$RUN/official_test_metrics.json"; then "$PY" -m trajcast_based.official evaluate; fi
"$PY" -m trajcast_based.configure
if ! test -f "$RUN/official30_100ps/run.json"; then
    "$PY" -m trajcast_based.official forecast --tag official30_100ps
fi
"$PY" -m trajcast_based.analyze_official --tag official30_100ps
if ! "$PY" -m trajcast_based.publish --push; then echo 'Publication failed; results retained locally. Retry push separately.'; fi
# Optional full published duration, explicitly requested on the command line.
if [[ "${1:-}" == --full-paper-duration ]]; then
    "$PY" -m trajcast_based.official forecast --production-steps 500000 --warmup-steps 50000 --tag official30_15ns
    "$PY" -m trajcast_based.analyze_official --tag official30_15ns
    if ! "$PY" -m trajcast_based.publish --push; then echo 'Publication failed; results retained locally. Retry push separately.'; fi
fi
