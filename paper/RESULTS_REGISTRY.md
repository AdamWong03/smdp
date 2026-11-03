# Results registry

## Frozen checkpoint

- Model: Quartz 30 fs R3-Opt, validation-selected epoch 364
- Parameters: 2,357,616
- Checkpoint SHA256: `d59c8110762e328c8c59debf60820090b6ab912fc896e675db619af02e395508`
- Test endpoint RMSE: 0.00180043 Å position; 0.00012208 Å/fs velocity
- Retained R1 independent test: 0.00165393 Å position RMSE and
  0.000112936 Å/fs velocity RMSE (1,000 samples; validation-selected epoch 348).
  R1 is therefore the stronger endpoint predictor; autonomous physical quality
  remains to be compared under the same protocol.

## Completed evidence

- 1/2/4/8-step autonomous error on 999 test continuations.
- Two independent 100 ps NVE rollouts with offline BKS energy analysis.
- One complete 1.5 ns + 15 ns NVT rollout with thermodynamics, RDF, VDOS,
  thermostat balance and end-to-end efficiency.
- A second independent 1.5 ns + 15 ns NVT trajectory from reference frame
  7,500,000 and seed 2026110302 completed with all applicable physical tasks.
  Its temperature, potential-energy distribution, RDF and VDOS reproduce the
  first run closely. Dense 1 fs VDOS is explicitly not applicable to this
  endpoint-only model.

See `reports/stable_geometry_optimization/R3_OPT_PHYSICS_RESULTS.md` for values.

## Missing evidence

- R1 basic autonomous physical evaluation (independent endpoint test complete).
- Third independent Quartz NVT and at least three additional NVE starts.
- Water 5 fs training and physical evaluation.
- Current-Only and Ballistic-Geometry matched controls.
- Mechanism analysis versus true pair-geometry change.
- 250/350 K and independently thermalized large-supercell transfer.
- Matched 5K/25K data-scale experiment.

No missing entry may be represented as zero, failure or a projected result.
