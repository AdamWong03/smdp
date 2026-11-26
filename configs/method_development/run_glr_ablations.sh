#!/usr/bin/env bash
# GLR component ablations, 100 ps each, val_04 start. Run sequentially on one GPU.
# Usage: GPU=1 bash run_glr_ablations.sh
set -euo pipefail

ROOT=/data2/wyd/research/foundation_mlip_adaptation
DATA=/data1/wyd/research/foundation_mlip_adaptation
GPU=${GPU:-1}
COEFF=$ROOT/experiments/glr_fit_20261122/glr_coefficients.json
BASE=(
  --checkpoint "$DATA/experiments/water6_reference_20261113/dsrt_radius6/best.pt"
  --initial "$DATA/data/smdp_reference_md_v1/water/raw_nve/val_04/trajectory.h5"
  --initial-frame 0 --steps 20000 --thermalization-steps 0
  --ensemble nve --analysis-ensemble nvt
  --glr-coefficients "$COEFF" --fd-blend 0.1
)
cd "$ROOT/code/h1h2"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 PYTHONPATH=$DATA/upstream/trajcast-c52f107:$PWD

run_case () {
  local name=$1; shift
  local out=$ROOT/experiments/${name}_20261122
  rm -rf "$out"; mkdir -p "$out"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
    /data1/wyd/envs/mlip/bin/python -u rollout_relational.py \
    "${BASE[@]}" --output "$out/trajectory.h5" --seed 2026110401 \
    --execution-note "$name" "$@" > "$out/rollout.log" 2>&1
}

run_case glr_rescale_off_100ps --no-fd-rescale
run_case glr_noise_off_100ps --glr-noise-off --fd-rescale
run_case glr_int10_100ps --fd-rescale --fd-blend-internal 1.0
run_case glr_seed2_100ps --fd-rescale --seed 2026110402
run_case glr_seed3_100ps --fd-rescale --seed 2026110403
