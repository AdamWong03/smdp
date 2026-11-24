#!/usr/bin/env bash
# Post-gate GLR queue: run on a free RTX GPU after the main 2.5 ns finishes.
# Usage: GPU=3 bash run_glr_postgate.sh
set -euo pipefail

ROOT=/data2/wyd/research/foundation_mlip_adaptation
DATA=/data1/wyd/research/foundation_mlip_adaptation
GPU=${GPU:-3}
P10=$ROOT/experiments/glr_fit_20261122/glr_coefficients.json
BASE=(
  --checkpoint "$DATA/experiments/water6_reference_20261113/dsrt_radius6/best.pt"
  --initial "$DATA/data/smdp_reference_md_v1/water/raw_nve/val_04/trajectory.h5"
  --initial-frame 0 --steps 20000 --thermalization-steps 0
  --ensemble nve --analysis-ensemble nvt --fd-blend 0.1
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

# key component ablations
run_case glr_rescale_off_100ps --glr-coefficients "$P10" --no-fd-rescale
run_case glr_noise_off_100ps --glr-coefficients "$P10" --fd-rescale --glr-noise-off
run_case glr_seed2_100ps --glr-coefficients "$P10" --fd-rescale --seed 2026110402
run_case glr_seed3_100ps --glr-coefficients "$P10" --fd-rescale --seed 2026110403

# order ablation (p=20/5/1)
run_case glr_p20_100ps --glr-coefficients "$ROOT/experiments/glr_fit_p20_20261122/glr_coefficients.json" --fd-rescale
run_case glr_p5_100ps --glr-coefficients "$ROOT/experiments/glr_fit_p5_20261122/glr_coefficients.json" --fd-rescale
run_case glr_p1_100ps --glr-coefficients "$ROOT/experiments/glr_fit_p1_20261122/glr_coefficients.json" --fd-rescale
