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

- R1 endpoint and two-run 100 ps NVE physical evaluation are complete.
- Water 5 fs training and independent test are complete: position/velocity
  RMSE 0.00065413 Å / 0.00023764 Å/fs at validation-selected epoch 400.
  The 999-start multistep test reaches position RMSE 0.008369 Å at 40 fs.
  A complete 100 ps NVE run exhibits heating: common 0–50 ps mean temperature
  406.18 K versus reference 275.99 K; full 100 ps mean 413.93 K, maximum
  608.65 K. Production throughput is 40.34 ns/day on L40S.
  See `reports/water5_20261103/RESULTS.md` and `results.json`.
- Water SPC/Fw reference energy/force verification and matched 0–50 ps NVE
  analysis are complete. Total energy increases by 6.296 eV versus -0.0235 eV
  in the reference. Mean internal/translation/rotation kinetic energies are
  8.327/0.973/0.727 eV versus 2.253/2.260/2.297 eV. H/O VDOS overlaps are
  0.391/0.614 and O–O RDF RMSE is 0.408. The liquid is not physically faithful.
- Signed-error diagnosis completed on 1,000 true test pairs and 160 autonomous
  states with their own true 5 fs SPC/Fw continuations. These diagnostic labels
  are never used for training. Real-state mean total-energy endpoint error is
  -0.00272 eV, whereas early autonomous-state sample means are positive;
  later means change sign and have large variance. No unique causal mechanism
  is claimed from these state/temperature-confounded samples.
- Water 250 ps thermalization + 2.5 ns production and all six physical tasks
  are complete. Full-step post-thermostat mean temperature is 311.427 K.
  Translation/rotation/internal kinetic energy averages 0.152/0.212/7.330 eV
  versus 2.459/2.502/2.450 eV. H/O VDOS overlap is 0.215/0.233; O–O RDF peak
  is 11.46 versus 3.16. Mean potential energy is -29.740 versus -27.682 eV.
  Model MSD plateaus: its tiny, sometimes negative block slopes do not establish
  a normal positive diffusion coefficient. Reference diffusion is 1.883e-9 m²/s.
  The original energy-task image/step-alignment errors were repaired, checked
  against raw reference data and rerun without regenerating trajectories.

## Missing evidence

- Third independent Quartz NVT and at least three additional NVE starts.
- Two additional Water NVT initial states from the original experiment matrix;
  the one completed run is a negative physical-fidelity result.
- Current-Only and Ballistic-Geometry matched controls.
- Mechanism analysis versus true pair-geometry change.
- 250/350 K and independently thermalized large-supercell transfer.
- Matched 5K/25K data-scale experiment.

No missing entry may be represented as zero, failure or a projected result.
