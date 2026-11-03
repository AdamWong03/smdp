# DSRT-R1 independent Quartz 30 fs endpoint test

The retained validation-selected R1 checkpoint (epoch 348) was evaluated once
on all 1,000 independent test transitions. The test set was not used for model
selection.

| Metric | R1 | Frozen R3-Opt |
|---|---:|---:|
| Position RMSE (Å) | 0.00165393 | 0.00180043 |
| Velocity RMSE (Å/fs) | 0.000112936 | 0.000122079 |
| Position MAE (Å) | 0.00131117 | 0.00142982 |
| Velocity MAE (Å/fs) | 0.0000895203 | 0.000096976 |

R1 is better on all four endpoint metrics. This result does not establish
better autonomous physical behavior; R1 remains a lightweight architecture
reference until evaluated with the same rollout protocol.

Server artifact:
`/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r1/test_metrics.json`
