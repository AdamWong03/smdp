# Stopped-run independent test results

Training was stopped at the user's request before the frozen 250,000-step budget completed. Each checkpoint was selected using validation data, then evaluated once on all 1,000 independent test samples.

| Model | Selected epoch | Selected steps | Stopped epoch | Position RMSE (A) | Velocity RMSE (A/fs) | Position MAE (A) | Velocity MAE (A/fs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3-Opt | 364 | 227,500 | 370 | 0.001800432 | 0.000122078 | 0.001429819 | 0.000096976 |
| Stable-DSRT | 365 | 228,125 | 365 | 0.002448112 | 0.000170581 | 0.001944950 | 0.000135545 |

Stable-DSRT has 36.0% higher position RMSE and 39.7% higher velocity RMSE than R3-Opt. The controlled test therefore supports the new optimization schedule for the original R3 block, but rejects the Stable residual reparameterization for this task.
