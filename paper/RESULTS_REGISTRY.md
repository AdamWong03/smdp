# Results registry

## Frozen checkpoint

- Model: Quartz 30 fs R3-Opt, validation-selected epoch 364
- Parameters: 2,357,616
- Checkpoint SHA256: `d59c8110762e328c8c59debf60820090b6ab912fc896e675db619af02e395508`
- Test endpoint RMSE: 0.00180043 Å position; 0.00012208 Å/fs velocity
- Retained R1 independent test: 0.00165393 Å position RMSE and
  0.000112936 Å/fs velocity RMSE (1,000 samples; validation-selected epoch 348).
  R1 is therefore the stronger endpoint predictor; autonomous physical quality
  is much worse under the same protocol: two 100 ps NVE runs average
  114.59/115.72 K and lose 8.02/8.49 eV, versus R3-Opt at 298.64/304.57 K and
  +1.30/+0.69 eV. R1 Si-O RDF RMSE is 1.87/1.93 versus 0.27/0.20 for R3-Opt.

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

- R1 endpoint and two-run 100 ps NVE physical evaluation are complete.
- Third independent Quartz NVT and at least three additional NVE starts.
- Water 5 fs training and physical evaluation.
- Current-Only and Ballistic-Geometry matched controls.
- Mechanism analysis versus true pair-geometry change.
- 250/350 K and independently thermalized large-supercell transfer.
- Matched 5K/25K data-scale experiment.

No missing entry may be represented as zero, failure or a projected result.
