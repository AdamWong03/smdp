# Matched validation rollout update, 2026-11-16 13:35 CST

Both frozen DSRT checkpoints completed 20 ps NVE from the same val_04/frame 0,
with no thermostat and no numerical failure. Full available validation multistep
evaluation also completed, with the same 1,249 starts having reference coverage.
No checkpoint was selected or retrained using these physical measurements.

| 0–20 ps mean | 4.5 Å | 6 Å | Reference |
|---|---:|---:|---:|
| Temperature, K | 369.843 | 296.553 | 274.994 |
| Molecular translation KE, eV | 1.40235 | 1.36561 | 2.21525 |
| Molecular rotation KE, eV | 1.11604 | 1.53773 | 2.23611 |
| Molecular internal KE, eV | 6.61254 | 4.41816 | 2.33788 |

The radius change reduces excessive internal motion and improves rotation in
this rollout, but does not recover translation. The lower overall temperature
error must not be interpreted as correct partitioning or equilibrium sampling.
Both are single, nonstationary NVE trajectories; independent-state variation
has not been measured for this new comparison.

| Validation autonomous horizon | 4.5 Å q RMSE, Å | 6 Å q RMSE, Å | 4.5 Å v RMSE, Å/fs | 6 Å v RMSE, Å/fs |
|---|---:|---:|---:|---:|
| 5 fs | .000657066 | .000411067 | .000238454 | .000149897 |
| 10 fs | .00192043 | .00113081 | .000332004 | .000208870 |
| 20 fs | .00450822 | .00273064 | .000445903 | .000301639 |
| 40 fs | .00847737 | .00585153 | .000749625 | .000526537 |

The 40 fs position and velocity RMSE decreases are about 31.0% and 29.8%.
This demonstrates improvement beyond a single transition, while the molecular
motion imbalance remains at picosecond times.

Strict eight-frame reference energy verification still fails for both analyses
on the same reference source. No energy tolerance, frame or force field was
changed, and no verified energy-drift conclusion is added. See
`PHYSICS_PROGRESS_20261116.md` for the input-coordinate audit and the two-frame
versus eight-frame verification inconsistency in legacy full-energy analysis.
The analysis status remains incomplete; full-energy curves are withheld from
formal conclusions. Successful trajectories and independent tasks were retained.

Native TrajCast continues at epoch 75 (74 completed, approximately 185,825
updates). Current best composite validation loss is at epoch 69: q RMSE
0.000182242 Å, v RMSE 0.0000771629 Å/fs. This is an unfinished validation result,
not an independent test result or evidence of its autonomous physical quality.
Do not infer which representation component caused the difference from DSRT,
since the native optimizer, precision and budget differ.
