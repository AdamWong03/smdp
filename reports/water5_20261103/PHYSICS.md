# Water: energy, motion and signed prediction-error diagnosis

Checkpoint: validation-selected epoch 400; no retraining or architecture change.
NVE comparison uses 2,000 matched frames over 0–50 ps. The model-only
trajectory continues to 100 ps. Frames are correlated; this is one run.

## Reference dynamics verification

The offline calculator uses the original flexible SPC/Fw topology, bonded
terms, excluded intramolecular pairs and PPPM 1e-5. Raw energies were not
centered. Eight fixed reference frames were checked against saved forces
and potential energy. The maximum differences are about 0.00032 eV and
0.0056 eV/Å; rounded raw dump coordinates limit reproducibility.
A fresh 5 fs continuation reproduces the stored endpoint within
0.000106 Å and 0.0000215 Å/fs (maximum component errors).

## NVE energy and motion

| Quantity | Model | Reference |
|---|---:|---:|
| Mean molecular translation KE (eV) | 0.97256 | 2.26005 |
| Mean molecular rotation KE (eV) | 0.72656 | 2.29741 |
| Mean molecular internal KE (eV) | 8.32709 | 2.25251 |
| 0–50 ps potential energy change (eV) | 2.00723 | -0.07463 |
| 0–50 ps kinetic energy change (eV) | 4.28867 | 0.05115 |
| 0–50 ps total energy change (eV) | 6.29590 | -0.02349 |
| 0–50 ps bond energy change (eV) | 8.43187 | 0.10521 |
| 0–50 ps nonbonded energy change (eV) | -5.98487 | 0.18741 |

Internal kinetic energy is strongly overexcited while molecular translation
and rotation are underexcited. O–H bond-length standard deviation is
0.05394 Å versus 0.02389 Å.
The angle distribution is narrower, not uniformly hotter. Total energy
increases, so the temperature rise is not explained by redistribution alone.

The complete model-only 100 ps total-energy change is +3.64176 eV.

## Signed endpoint bias

The existing 1,000 test pairs provide real-state labels. For autonomous
states, 32 states were sampled in each of five time windows and the original
SPC/Fw potential generated an independent 5 fs continuation from each state
itself (160 diagnostic labels, 0.8 ps total reference integration). These
labels are never training data. No sample failed or was filtered out.

| Input states | Samples | Mean total-energy endpoint error (eV) | Std (eV) | Mean velocity RMSE (Å/fs) |
|---|---:|---:|---:|---:|
| reference/test | 1000 | -0.002719 | 0.013350 | 0.0002367 |
| autonomous/0-1ps | 32 | +0.004338 | 0.012157 | 0.0002292 |
| autonomous/1-5ps | 32 | +0.004920 | 0.018797 | 0.0002372 |
| autonomous/5-20ps | 32 | +0.006073 | 0.035135 | 0.0003010 |
| autonomous/20-50ps | 32 | -0.013964 | 0.072648 | 0.0004180 |
| autonomous/50-100ps | 32 | +0.002943 | 0.059773 | 0.0003636 |

The real-state test has a small negative mean energy bias. Early autonomous
states have positive sample means; later intervals have larger variance and
changing signs. This contradicts a simple constant velocity-amplification
explanation but does not identify a unique causal network mechanism.
Autonomous inputs also have different temperatures and kinetic partition,
so temperature and state-distribution effects remain confounded.
Stratified sample means are diagnostics, not precise long-time energy rates.
Signed per-element errors are conditional on the current state and must not
be confused with accumulated per-element energy: H can already be too hot
while its next-step conditional kinetic-energy error is negative.

## Structure and velocity spectra

| Metric | Value |
|---|---:|
| 8-8 RDF RMSE | 0.407523 |
| 8-1 RDF RMSE | 1.538260 |
| 1-1 RDF RMSE | 0.152815 |
| Z=1 VDOS overlap | 0.390900 |
| Z=8 VDOS overlap | 0.613845 |

The O–O peak is substantially over-sharpened. VDOS uses 5 fs matched
sampling (100 THz Nyquist); the heated NVE is nonstationary, so these
overlaps are not equilibrium spectra or an independent diffusion result.

## Follow-up NVT protocol

One 300 K CSVR run uses the original 50 fs coupling time, 250 ps
thermalization and 2.5 ns production. The frozen checkpoint is unchanged.
Per-element pre/post thermostat kinetic energies are recorded. Molecular
COM diffusion uses eight blocks, 50 fs sampling and a predeclared 10–100 ps
MSD fitting interval. The GPU is shared, so runtime is not an exclusive-GPU
benchmark. Completed NVT results are reported separately in NVT_RESULTS.md.

## Software correction

A new diagnostic initially failed an algebraic kinetic-error identity due
to mixing float32 squaring with float64 evaluation (1.5e-9 eV discrepancy).
All offline kinetic arithmetic now uses float64; the identity passes.
No model weights, labels or trajectory were changed by this correction.

## Reproduction

```bash
LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib OMP_NUM_THREADS=1 \
/data1/wyd/envs/mlip/bin/python analyze_water.py \
  --candidate /data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/water5_20261103_r3_opt/physics/nve_test_05/trajectory.h5 \
  --reference /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/raw_nve/test_05/trajectory.h5 \
  --topology /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/equilibration/snapshots/water_5.data \
  --output /data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/water5_20261103_r3_opt/physics/nve_test_05/analysis
```

See `configs/water5_physics.json`, `diagnose_water_bias.py` and
`scripts/run_water5_nvt.sh` for the remaining exact execution settings.
Large trajectory and diagnostic-label HDF5 files stay on the server.
