# Frozen segment-best physical evaluation

This evaluation does not wait for or modify rollout training. The frozen source is
`experiments/trajcast_smdp/quartz30_20261028/segment/best.pt` (epoch 389,
zero-based), selected on validation before opening the independent test set.

Run directory (relative to `/data1/wyd/research/foundation_mlip_adaptation`):
`experiments/trajcast_smdp/quartz30_segment_physics_20261101`.
The snapshot SHA256 and full selection record are in `evaluation_manifest.json`.

Protocol: all 1000 independent test transitions, two test-trajectory initial
states for 100 ps NVE, then 1.5 ns thermalization and 1 ns production NVT at
300 K using the unchanged configured CSVR tau of 100 fs. The production length
is storage-limited and is **not** the 15 ns paper-scale comparison. Output keeps
30 fs positions/velocities plus 1 fs neural-query velocities. Macro spectra are
compared to equally downsampled BKS MD (Nyquist 16.67 THz); dense spectra use
the high-resolution reference separately. Queried NVT endpoints include the
actual thermostat impulse, as recorded in the trajectory manifest.

Offline energies use the same BKS parameters for both trajectories and report
absolute raw DSF energies, uncentered Wasserstein distance and energy drift.
Independent energy initialization checks and remaining stored-energy precision
differences are documented in `reference_energy_check.json`.

Reproduction (preparation refuses to overwrite an existing evaluation):

```bash
/data1/wyd/envs/trajcast-c52f107/bin/python -m trajcast_smdp.prepare_segment_evaluation
CUDA_VISIBLE_DEVICES=<compatible-L40S-UUID> bash scripts/run_segment_physics.sh
```

The runner reuses completed test/trajectory outputs, runs independently saved
analysis tasks, and publishes small results after each trajectory. Raw logs,
checkpoints and HDF5 stay on the server. Explicit trajectory failure or disk
safety stops are retained, never silently extended or replaced. For an existing
partial HDF5 without `run.json`, investigate before restarting; overwrite is
deliberately refused. No BKS force is called during neural propagation.

Current service: `smdp-segment-physics-20261101-gpu2.service`. GPU 2 is shared;
recorded wall times are observed shared-hardware throughput, not an exclusive
GPU performance claim. The initial Blackwell launch failed before generating
trajectories because the existing PyTorch build lacks sm_120 support; the log
is preserved. No new environment was installed.
