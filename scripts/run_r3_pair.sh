#!/usr/bin/env bash
# Frozen R3-A (temporal composition) / R3-B (+ spatial mode exchange).
# Runs training and validation only, never test/NVE/NVT or new reference MD.
set -euo pipefail

ROOT="${ROOT:-/data1/wyd/research/foundation_mlip_adaptation}"
REPO="${REPO:-$ROOT/smdp}"
DATA="${DATA:-$ROOT/data/smdp_reference_md_v1}"
EXPERIMENTS="${EXPERIMENTS:-$ROOT/experiments/motion_modes}"
PYTHON="${PYTHON:-/data1/wyd/envs/mlip/bin/python}"

mkdir -p "$EXPERIMENTS"
cd "$REPO"

A="$EXPERIMENTS/quartz16_r3a_composition"
B="$EXPERIMENTS/quartz16_r3b_spatial_composition"

"$PYTHON" -m pytest -q test_neural_smdp.py test_integration.py test_r3.py
"$PYTHON" -u train_quartz.py --data-root "$DATA" --output "$A" \
  --r3-composition-weight 0.05 --r3-composition-samples 2 \
  --max-rollout-steps 8 --seed 20261022
"$PYTHON" -u train_quartz.py --data-root "$DATA" --output "$B" \
  --r3-composition-weight 0.05 --r3-composition-samples 2 \
  --max-rollout-steps 8 --spatial-mode-exchange --seed 20261022

"$PYTHON" -u validate_r3.py --data-root "$DATA" \
  --checkpoints "$A/best_rollout.pt" "$B/best_rollout.pt" \
  --output "$EXPERIMENTS/quartz16_r3_pair_validation.json"

"$PYTHON" -u benchmark_r3.py --data-root "$DATA" \
  --checkpoint "$A/best_rollout.pt" --output "$A/benchmark.json"
"$PYTHON" -u benchmark_r3.py --data-root "$DATA" \
  --checkpoint "$B/best_rollout.pt" --output "$B/benchmark.json"

echo "R3 A/B training, matched validation and benchmark finished. No test/MD rollout was run."
