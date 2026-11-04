# Water 5 fs: completed prediction and NVE results

Frozen model: geometry_gated DSRT, 192/48/96 channels, 2+2 blocks,
2,357,616 parameters, validation-selected epoch 400 (250,000 updates).

## Prediction

The full independent endpoint test contains 1,000 samples: position RMSE/MAE
are 0.000654127/0.000453784 Å and velocity RMSE/MAE are
0.000237643/0.000165311 Å/fs. The multistep evaluation uses 999 starts with
enough remaining reference frames for all eight steps.

| Autonomous time (fs) | Position RMSE (Å) | Velocity RMSE (Å/fs) | Temperature RMSE (K) |
|---|---:|---:|---:|
| 5 | 0.00065410 | 0.00023764 | 0.446 |
| 10 | 0.00191147 | 0.00033023 | 0.679 |
| 20 | 0.00448005 | 0.00044308 | 0.934 |
| 40 | 0.00836886 | 0.00074508 | 1.520 |

## Autonomous NVE

The model completed 20,000 unthermostatted steps (100 ps) without triggering
the numerical safety stop, but it exhibits substantial nonphysical heating.
The source reference trajectory covers 50 ps; comparisons below explicitly
use that common duration.

| Metric | DSRT | Reference MD |
|---|---:|---:|
| Initial temperature (K) | 275.108 | 275.108 |
| Mean temperature over common 0–50 ps (K) | 406.180 | 275.987 |
| Mean temperature over model 0–100 ps (K) | 413.929 | Not available |
| Maximum temperature over model 0–100 ps (K) | 608.654 | Not compared |

The model temperature averages 284.75 K at 0–1 ps, 303.92 K at 1–5 ps,
363.18 K at 5–20 ps, 445.38 K at 20–50 ps and 421.68 K at 50–100 ps.
The final temperature is 315.23 K, illustrating why a final-frame value alone
would conceal the heating. The minimum interatomic distance is 0.86926 Å;
this all-pair minimum includes bonded O–H contacts.

The completed production run achieved 40.34 ns/day on an NVIDIA L40S,
including graph updates and trajectory output (214.18 seconds for 100 ps).
Peak allocated CUDA tensor memory was 78.27 MB. The separate batched multistep
test uses a different graph-building path and is not the production throughput.

## Interpretation and remaining work

Good endpoint accuracy has not produced thermally faithful autonomous Water
dynamics. Numerical completion is not physical validation. Potential/total
energy, RDF, VACF/VDOS, MSD/diffusion and formal NVT have not yet been evaluated
for this checkpoint. Water energy evaluation must use the reference flexible
water potential, not the Quartz BKS calculator. No explanation of the total
energy source is claimed from temperature alone.

`results.json` contains machine-readable completed metrics and explicit pending
analysis status. `training_and_nve.png` shows the full training curve and NVE
temperature evolution. Full trajectories, logs and weights remain on the server.

Reproduce the exported results from the repository root:

```bash
/data1/wyd/envs/mlip/bin/python scripts/export_water5_results.py \
  --run /data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/water5_20261103_r3_opt \
  --reference /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/raw_nve/test_05/trajectory.h5 \
  --output reports/water5_20261103
```
