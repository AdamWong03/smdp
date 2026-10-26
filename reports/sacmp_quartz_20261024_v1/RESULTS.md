# SACMP Quartz 16 fs — completed training and 10 ps NVE results

Published 2026-10-26. These are observed results, not projections of long-time
performance. **100 ps NVE and 15 ns NVT have not been run.**

## Protocol and provenance

- Code: `89d76a33c90a80028721d479a929e6ad1d377246`.
- Existing `smdp_reference_md_v1`, Quartz 162 atoms, independent trajectories:
  5,000 training / 1,250 validation / 1,000 test samples, 16 fs endpoints.
- Train-derived 192-mode acceleration PCA initialization, no test coordinates
  used for initialization; capture 81.6664%, velocity capture 40.7143%.
- Seed 20261024, batch 4. Completed 80 segment epochs and 20 rollout epochs
  (16/32/48 fs supervision). Best segment epoch 78, best rollout epoch 19
  (one-based, selected on validation, not test).
- 118,951 parameters, 24,892.48 s (6.91 h) training, peak allocated training
  GPU memory 4,930,156,544 bytes; RTX PRO 6000 Blackwell Server Edition.
- Checkpoint SHA256:
  `037c79794a5b151c54ffb31fb320a2368fb3acf6349874ce106cf28754675c46`.
- No BKS forces, thermostat, energy rescaling or reference-state correction
  during the two NVE trajectories. BKS energy is evaluated offline.

The model is the uploaded SACMP candidate (learned fixed-crystal modal frame,
harmonic fast flow, learned shared potential kicks and slow drift), not the
previous R3-B network. Its crystal-size dependence and 4 fs internal substeps
remain limitations; these results establish no transfer or force-free-kernel claim.

## Independent test (all 1,000 samples)

| Metric | SACMP | Historical R3-B |
|---|---:|---:|
| 16 fs position MAE / A | 0.00365913 | 0.00196155 |
| 16 fs position RMSE / A | 0.00466407 | 0.00249977 |
| 16 fs velocity MAE / A/fs | 0.000381761 | 0.000263196 |
| 16 fs velocity RMSE / A/fs | 0.000491009 | 0.000342548 |

SACMP endpoint accuracy is worse on this shared test set. Query-by-query
errors are in [test/query_errors.csv](test/query_errors.csv).
The central position-difference versus velocity RMSE is 0.000178269 A/fs,
versus 0.00000872548 A/fs for reference MD (~20.4 times larger).
An infinitesimal kinematic identity does not ensure accurate finite query segments.

COM velocity RMSE is 1.34e-10 A/fs. Reverse/8+8 composition diagnostics are
small, but the stored combined state norms mix position and velocity units;
they are numerical checks, not physically calibrated accuracy metrics.
8+8 and 16 use the same 4 fs substep grid, so their agreement is primarily a
construction property. Learned-H relative RMSE is 0.003863; learned H is not BKS H.

## Two independent 10 ps autonomous NVE screens

| Metric | test_09 | Matching BKS | test_10 | Matching BKS |
|---|---:|---:|---:|---:|
| Mean temperature / K | 289.633 | 287.894 | 300.438 | 302.907 |
| Temperature std / K | 13.524 | 12.520 | 13.656 | 13.645 |
| Mean O kinetic energy / eV | 4.01368 | 4.01870 | 4.20457 | 4.23680 |
| Mean Si kinetic energy / eV | 2.01383 | 1.97263 | 2.04779 | 2.06695 |
| BKS total energy end-start / eV | +0.294080 | -0.004229 | +0.087922 | -0.002432 |
| Total energy change RMS / eV | 0.348007 | 0.004479 | 0.250855 | 0.002479 |
| Fitted energy slope / meV/(atom ps) | +0.039189 | -0.002135 | +0.135265 | -0.002242 |
| Mean potential energy bias / eV | +0.298597 | 0 | +0.271965 | 0 |
| Minimum distance / A | 1.478849 | not evaluated | 1.487638 | not evaluated |
| O VDOS overlap, 16 fs | 0.917384 | 1 | 0.917710 | 1 |
| Si VDOS overlap, 16 fs | 0.913985 | 1 | 0.920379 | 1 |
| O VDOS overlap, queried 1 fs | 0.912592 | 1 | 0.914121 | 1 |
| Si VDOS overlap, queried 1 fs | 0.910368 | 1 | 0.917093 | 1 |
| Si-O RDF RMSE | 0.085051 | 0 | 0.082012 | 0 |
| O-O RDF RMSE | 0.019151 | 0 | 0.027835 | 0 |

Both completed 625 autonomous steps without triggering the existing failure
guards. Temperature, species kinetic allocation and RDF are encouraging over
10 ps. The model's BKS energy is nevertheless substantially less stable than
reference MD: its energy-change ranges are [0, 0.62248] and [-0.09804, 0.56777]
eV, respectively. Small endpoint changes alone are not proof of conservation.

The 16 fs VDOS Nyquist frequency is 31.25 THz. Dense queries and reference
use 1 fs sampling. Each 10 ps spectrum has only **one analysis window**;
the exported zero block standard deviation is not an uncertainty estimate.
Do not directly compare these overlaps to 100 ps NVE or 15 ns NVT overlaps.

**Known report-label limitation:** the inherited `metrics.json/nve_windows`
labels the final window `5–20 ps`, although this run ends at 10 ps and the
interpolation is clamped there. That entry only contains 5–10 ps data and
must not be interpreted as a 20 ps result. We retain the original JSON
unchanged; use the actual timestamps in `thermodynamics.csv`. Full-run
metrics above use only the real 0–10 ps interval.

Figures and time series:

- [test_09 energy evolution](nve_screen_test_09_analysis/energy_partition.png),
  [RDF](nve_screen_test_09_analysis/rdf.png),
  [16 fs VDOS](nve_screen_test_09_analysis/vdos_macro_16fs.png),
  [1 fs VDOS](nve_screen_test_09_analysis/vdos_queried_1fs.png).
- [test_10 energy evolution](nve_screen_test_10_analysis/energy_partition.png),
  [RDF](nve_screen_test_10_analysis/rdf.png),
  [16 fs VDOS](nve_screen_test_10_analysis/vdos_macro_16fs.png),
  [1 fs VDOS](nve_screen_test_10_analysis/vdos_queried_1fs.png).

## Actual computational cost

Dense-query NVE wall times: 155.91 / 154.91 s for 10 ps, or **5.54 / 5.58
ns/day**, including model queries, geometry checks and output. Model time is
148.29 / 147.81 s, roughly 95% of wall time. Peak allocated inference memory
is 79,589,888 bytes. Endpoint-only timing has not been measured for this model.

At this observed dense-output rate, 1.5 ns thermalization plus 15 ns production
would take about three days; that is an estimate, not an executed experiment.
Historical R3-B timing used L40S and a different shared-server load, so no
strict speed ratio is claimed. The current dense propagation cost is a major
limitation even though parameter count is small.

## Judgment and status

The 10 ps physical behavior warrants further examination, but this is not
evidence of 100 ps/15 ns stability, overall superiority to R3-B, or acceleration
over BKS. Endpoint errors, physical BKS energy deviations, finite-query
kinematic consistency and computational cost remain weaknesses.
No further training, architecture change, ablation or long trajectory was
started after these results.

## Reproduction and server artifacts

Server directory:
`/data1/wyd/research/foundation_mlip_adaptation/experiments/sacmp/quartz_20261024_v1`

Weights: `best_rollout.pt`; source logs: `train.log`, `history.jsonl`,
`evaluation_pipeline.log`; raw trajectories: `nve_screen_test_09.h5` and
`nve_screen_test_10.h5`. All stay on the server. Training invocation and
source split paths are recorded in [config.json](config.json).
Run protocol is in [README.md](README.md) and `scripts/run_sacmp_quartz_v1.sh`;
never rerun the training launcher into the completed output directory.

To reproduce this bounded report export (no training or trajectory generation):

```bash
/data1/wyd/envs/mlip/bin/python scripts/export_sacmp_results.py \
  --output /data1/wyd/research/foundation_mlip_adaptation/experiments/sacmp/quartz_20261024_v1 \
  --report-dir reports/sacmp_quartz_20261024_v1
```

The exporter copies only allowlisted small results and derives the training
curve and energy RMS from existing logs/CSV. It does not modify original results.
