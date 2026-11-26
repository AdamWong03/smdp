#!/usr/bin/env bash
# Translation-blend sweep: keep rotation/internal at 0.1, raise translation.
set -euo pipefail

ROOT=/data2/wyd/research/foundation_mlip_adaptation
DATA=/data1/wyd/research/foundation_mlip_adaptation
GPU=${GPU:-3}
P10=$ROOT/experiments/glr_fit_20261122/glr_coefficients.json
cd "$ROOT/code/h1h2"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$DATA/upstream/trajcast-c52f107:$PWD

for blend in 0.2 0.3; do
  run=$ROOT/experiments/glr_ltrans${blend}_100ps_20261122
  rm -rf "$run"; mkdir -p "$run"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
    /data1/wyd/envs/mlip/bin/python -u rollout_relational.py \
    --checkpoint "$DATA/experiments/water6_reference_20261113/dsrt_radius6/best.pt" \
    --initial "$DATA/data/smdp_reference_md_v1/water/raw_nve/val_04/trajectory.h5" \
    --initial-frame 0 --output "$run/trajectory.h5" --ensemble nve --analysis-ensemble nvt \
    --steps 20000 --thermalization-steps 0 --glr-coefficients "$P10" \
    --fd-blend 0.1 --fd-blend-translation "$blend" --fd-rescale --seed 2026110401 \
    --execution-note "GLR AR(10) + translation blend ${blend} + rescale, 100 ps" > "$run/rollout.log" 2>&1
done
