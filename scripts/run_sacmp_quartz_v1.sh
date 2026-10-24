#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data1/wyd/research/foundation_mlip_adaptation}
PYTHON=${PYTHON:-/data1/wyd/envs/mlip/bin/python}
OUT=${OUT:-$ROOT/experiments/sacmp/quartz_20261024_v1}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$ROOT/smdp"
if [[ -e "$OUT/history.jsonl" || -e "$OUT/train.log" ]]; then
    echo "Existing run: refusing to overwrite or restart automatically" >&2
    exit 1
fi
"$PYTHON" -u -m sacmp.train \
    --data-root "$ROOT/data/smdp_reference_md_v1" --output "$OUT" \
    --anchor "$OUT/anchor_train_only.pt" --device cuda --rank 192 \
    --batch-size 4 --segment-epochs 80 --rollout-epochs 20 \
    --seed 20261024 > "$OUT/train.log" 2>&1
"$PYTHON" -u -m sacmp.run \
    --data-root "$ROOT/data/smdp_reference_md_v1" --output "$OUT" \
    --after-training --device cuda --dense > "$OUT/evaluation_pipeline.log" 2>&1
echo "SACMP_TRAIN_AND_SCREEN_COMPLETED; full production requires review"
