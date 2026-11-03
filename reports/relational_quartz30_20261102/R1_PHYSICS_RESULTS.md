# Retained DSRT-R1 Quartz 30 fs physical evaluation

The validation-selected lightweight R1 checkpoint (0.854M parameters, epoch
348) was evaluated on the same two independent 100 ps NVE initial states and
with the same offline BKS analysis used for frozen R3-Opt. No retraining or
trajectory correction was performed.

## Endpoint versus autonomous dynamics

R1 has lower independent-test endpoint error than R3-Opt:

| Model | Position RMSE (Å) | Velocity RMSE (Å/fs) |
|---|---:|---:|
| R1 | 0.00165393 | 0.000112936 |
| R3-Opt | 0.00180043 | 0.000122079 |

This ordering reverses decisively under autonomous NVE propagation.

| Metric | R1 test 09 | R1 test 10 | R3-Opt test 09 | R3-Opt test 10 |
|---|---:|---:|---:|---:|
| Mean temperature (K) | 114.59 | 115.72 | 298.64 | 304.57 |
| 100 ps total-energy change (eV) | -8.017 | -8.492 | +1.302 | +0.689 |
| Energy drift (meV atom^-1 ps^-1) | -0.2307 | -0.2084 | +0.0579 | +0.0886 |
| Si-O RDF RMSE | 1.8729 | 1.9335 | 0.2736 | 0.2018 |
| O-O RDF RMSE | 0.3630 | 0.3824 | 0.01582 | 0.02467 |
| O VDOS overlap | 0.9154 | 0.9131 | 0.9543 | 0.9549 |
| Si VDOS overlap | 0.8712 | 0.8610 | 0.9555 | 0.9557 |

Both R1 trajectories completed 100 ps without a numerical safety stop and
retained minimum interatomic distances above 1.50 Å. They nevertheless lost
kinetic and total energy rapidly: mean temperatures fell to about 115 K and
the local structure became substantially over-sharpened. The behavior is
consistent across both independent starts.

## Research conclusion

The result rules out endpoint RMSE as a sufficient model-selection criterion
for learned molecular dynamics. The richer R3-Opt model is slightly worse on
one-step supervised error but substantially better on temperature, energy,
RDF and VDOS during autonomous propagation. R1 remains a useful lightweight
negative control rather than a candidate paper model.

Server artifacts:

- `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r1/physics/nve_test_09`
- `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r1/physics/nve_test_10`
