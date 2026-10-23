# Frozen R3-B full physical evaluation

Launch status (2026-10-23): independent test evaluation running; production
trajectories and physical results are **not yet complete**. The pipeline runs
the requested stages sequentially and publishes observed results on completion.
This launch record is not evidence of successful 15 ns dynamics.

Selected checkpoint (no retraining/model changes):

`/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_r3b_spatial_composition/best_rollout.pt`

SHA256: `667dcbc8a3989b4eea1204e5455724e0b1a9105a5574cb6df0ffddf5579b58ba`.
Execution code commit: `86ce650`; model/training definitions remain frozen.

## Protocol

- Full 1000-sample independent test set; 1–16 fs segment errors and every
  autonomous endpoint from 32 to 256 fs (valid reference continuations only).
- NVE: original `test_09`, `test_10` initial states, 6250 steps each (100 ps).
- NVT: original reference production initial state; 93750 thermalization steps
  (1.5 ns), then 937500 production steps (15 ns), all at 16 fs.
- CSVR: unchanged 300 K target, 100 fs coupling time, seed 2026102201,
  483 internal degrees of freedom, original COM convention.
- BKS offline only; no force correction during neural propagation.
- Analysis: unchanged 2000 uniform thermodynamic frames, 512 RDF frames,
  identical spectral window/normalization; 16 fs endpoint and 1 fs queried
  velocity spectra reported separately. Six CPU spectral workers preserve
  the mathematical definition and reuse completed cached blocks.
- Actual production timing includes graph construction, model, thermostat,
  output and diagnostics. Separate endpoint/dense short-block timings are
  not represented as 15 ns production throughput.

## Execution and recovery

20 software correctness tests passed (18 model/integration/analysis, 2 overwrite
guards). Restored MPICH runtime rechecked on 24 fixed R1/reference configurations:
maximum BKS energy discrepancy `1.72804e-11 eV`. No library relinking or BKS
parameter changes. Runtime verification is saved alongside experiment outputs.

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
CUDA_VISIBLE_DEVICES=GPU-ed405bf3-4069-2d20-0432-9ecda26de366 \
  bash scripts/run_r3b_physics.sh
```

GPU: NVIDIA L40S (physical index 5 at launch). Complete production command,
checkpoint hash and runtime paths are saved in `evaluation_config.json`.

Server experiment directory is the checkpoint's parent above. Logs:
`production_pipeline.log`, `test.log`, `nve_test_09.log`, `nve_test_10.log`,
`nvt.log`, `analyze_nve_test_09.log`, `analyze_nve_test_10.log`, `analyze_nvt.log`.
Raw HDF5, checkpoint and logs stay there; bounded metrics and figures are
published here automatically after the pipeline finishes.

Rerunning the entrypoint never overwrites existing trajectories or checkpoints.
Completed physical tasks are reused; failed tasks can be rerun independently.
An interrupted/physically failed trajectory is retained and explicitly flagged,
not silently regenerated or treated as complete. For an interrupted trajectory
requiring state/RNG recovery, inspect its actual progress before taking action.

`pipeline_finished.json` distinguishes execution finishing from
`all_requested_completed`. Per-run trajectory and physical-analysis completion
are separate. No failed or partial result may be called a complete experiment.

The comparison uses original R1 results and corrected R2 `reanalysis_v1` results.
No R4, new training, thermostat tuning or architecture change is scheduled.
