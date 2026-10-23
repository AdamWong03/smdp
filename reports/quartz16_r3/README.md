# Quartz 16 fs R3 paired experiment

Status at launch, 2026-10-23: **R3-A running; R3-B queued; no R3 results yet.**
Training/evaluation code: `0cfbb21`, branch
`experiment/r3-dynamics-consistent-modes`. Correctness checks: 14 passed.
No new reference MD, production NVE/NVT, test-set tuning, or R4 is scheduled.

## Frozen comparison

- A: R2 topology, composition coefficient 0.05, two selected samples per batch.
- B: A plus one mid-propagation spatial mode exchange.
- Both: seed 20261022, 5000/1250 train/validation samples, batch 8,
  at most 100 segment epochs, 12 rollout4 and 12 rollout8 epochs,
  cross-rollout-stage selection using the common 16-step validation objective.
- Post-selection composition evaluation uses every valid validation window;
  the reference-target objective excluding composition is also reported.
- Common 16/256 fs errors, species velocity correlations/kinetic statistics,
  kinematics and mode diagnostics are evaluated without optimizing any weights.
- Fixed-input and short autonomous-block timing are separate, each with endpoint
  and dense time queries. These exclude thermostat and trajectory-output I/O.

A and B are matched controls. Historical R2 additionally differs in rollout
schedule and checkpoint selection (it included rollout16 and selected only
within that final stage). Thus A-versus-R2 gains alone cannot establish that
composition, rather than those training changes, caused the improvement.
No third model is added to resolve that confound in this round.

## Server paths

Base: `/data1/wyd/research/foundation_mlip_adaptation`.

- Data: `data/smdp_reference_md_v1` (unchanged).
- A: `experiments/motion_modes/quartz16_r3a_composition`.
- B: `experiments/motion_modes/quartz16_r3b_spatial_composition`.
- Pair logs/diagnostics: `experiments/motion_modes/quartz16_r3_pair`.
- Common validation: `experiments/motion_modes/quartz16_r3_pair_validation.json`
  (created after training, not present at launch).
- Per-arm checkpoints/config/history: `best_rollout.pt`, `config.json`,
  `history.jsonl`; phase checkpoints and optimizer state are also retained.
- Logs: `quartz16_r3_pair/pipeline.log`, `train_a.log`, `train_b.log`.

Both arms run sequentially on NVIDIA L40S, UUID
`GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900` (physical `nvidia-smi` index 2;
CUDA ordinal 1 on this host; the two orderings differ). Other users' workloads were already
active at launch. Runtime figures must disclose this contention and must not
be presented as isolated-hardware speedups against older production timings.

Reproducible launch from the repository (refuses to overwrite existing runs):

```bash
CUDA_VISIBLE_DEVICES=GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  PYTHON=/data1/wyd/envs/mlip/bin/python bash scripts/run_r3_pair.sh
```

Large weights, trajectory caches and raw logs stay on the server. This page is
a launch record, not a completed experiment report. Actual validation metrics,
curves, costs and conclusions will be added after both runs finish.
