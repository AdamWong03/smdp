# Stable Geometry-DSRT training snapshot

Generated: `2026-11-03T02:08:59.752365+00:00`

This is a live training snapshot. It is not labeled as a final test result unless both fixed 250,000-step schedules and held-out tests have completed.

| Model | State | Epoch | Best epoch | Best steps | Best val loss | q RMSE (A) | v RMSE (A/fs) |
|---|---|---:|---:|---:|---:|---:|---:|
| R3-Opt | training | 360 | 357 | 223,125 | 0.000866800 | 0.001779780 | 0.000120243 |
| Stable-DSRT | training | 355 | 351 | 219,375 | 0.001648094 | 0.002412446 | 0.000167376 |

The complete per-epoch records, including learning rate, proposal/final losses, gradient diagnostics and feature statistics, are stored in `training_snapshot.json`.

At this snapshot R3-Opt improves over the old R3 validation loss (0.001036336), but remains above the historical R1 result (0.000736517). Stable-DSRT remains worse than R3-Opt, so there is currently no evidence that its residual reparameterization helps this task.
