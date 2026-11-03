# Stable Geometry-DSRT training snapshot

Generated: `2026-11-03T02:25:47.044202+00:00`

This is a live training snapshot. It is not labeled as a final test result unless both fixed 250,000-step schedules and held-out tests have completed.

| Model | State | Epoch | Best epoch | Best steps | Best val loss | q RMSE (A) | v RMSE (A/fs) |
|---|---|---:|---:|---:|---:|---:|---:|
| R3-Opt | stopped_and_test_completed | 370 | 364 | 227,500 | 0.000864482 | 0.001777579 | 0.000120075 |
| Stable-DSRT | stopped_and_test_completed | 365 | 365 | 228,125 | 0.001643191 | 0.002407985 | 0.000167159 |

The complete per-epoch records, including learning rate, proposal/final losses, gradient diagnostics and feature statistics, are stored in `training_snapshot.json`.

At this snapshot R3-Opt improves over the old R3 validation loss (0.001036336), but remains above the historical R1 result (0.000736517). Stable-DSRT remains worse than R3-Opt, so there is currently no evidence that its residual reparameterization helps this task.
