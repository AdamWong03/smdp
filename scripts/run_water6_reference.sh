#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/wyd/research/foundation_mlip_adaptation
cd "$ROOT/smdp"
export PYTHONPATH="$PWD:$ROOT/upstream/trajcast-c52f107"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONWARNINGS=ignore::FutureWarning
ARM=${1:?choose dsrt or trajcast}
case "$ARM" in
  dsrt)
    PY=/data1/wyd/envs/mlip/bin/python
    CONFIG=configs/water6_dsrt_20261113.json
    CHECK=water_reference.check_dsrt
    RUN="$ROOT/experiments/water6_reference_20261113/dsrt_radius6"
    ;;
  trajcast)
    PY=/data1/wyd/envs/trajcast-c52f107/bin/python
    CONFIG=configs/water6_trajcast_native_20261113.json
    CHECK=water_reference.train_native
    RUN="$ROOT/experiments/water6_reference_20261113/trajcast_native"
    ;;
  *) exit 2 ;;
esac
mkdir -p "$RUN"
exec 9>"$RUN/run.lock"
flock -n 9 || { echo 'Run already active'; exit 1; }
test -f "$ROOT/data/smdp_reference_md_v1/water/derived_graphs/water6_reference_20261113/preparation.json"
if test -f "$RUN/config.json"; then
  cmp "$CONFIG" "$RUN/config.json"
else
  cp "$CONFIG" "$RUN/config.json"
fi
test ! -f "$RUN/last.pt" || { echo 'Use explicit resume command; no overwrite'; exit 1; }
printf '%s\n' "$$" > "$RUN/launcher.pid"
nvidia-smi --query-gpu=index,name,uuid,memory.free,utilization.gpu --format=csv > "$RUN/gpu_at_launch.csv"
git rev-parse HEAD > "$RUN/source_commit.txt"
"$PY" -m pip freeze > "$RUN/environment.txt"
if [ "$ARM" = dsrt ]; then
  "$PY" -u -m "$CHECK" --run "$RUN"
  "$PY" -u train_relational.py --run "$RUN"
else
  "$PY" -u -m "$CHECK" --run "$RUN" --check
  "$PY" -u -m water_reference.train_native --run "$RUN"
fi
