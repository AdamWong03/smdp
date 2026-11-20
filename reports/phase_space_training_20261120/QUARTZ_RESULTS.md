# Phase-space operator: completed Quartz 30 fs results

2026-11-20. Run `experiments/phase_space_20261120/quartz30`, selected epoch 396
(247,500 updates), checkpoint SHA-256
`8ff2c78e0c2a2ace63b120dc7eef10fbc722cb65b1ff4ef474773dc0f1bea4a9`.
Validation selected the checkpoint; the test split was read once afterwards by
the trainer and again by the frozen evaluation pipeline.

## Endpoint and multistep

| Metric | phase-space v0 | frozen R3-Opt |
|---|---:|---:|
| Test position RMSE | 0.001879 Å | 0.001800 Å |
| Test velocity RMSE | 0.0001293 Å/fs | 0.0001221 Å/fs |
| Val 1-step position RMSE | 0.001851 Å | 0.001800 Å |
| Val 2-step position RMSE | 0.004158 Å | 0.004002 Å |
| Val 4-step position RMSE | 0.009280 Å | 0.008840 Å |
| Val 8-step position RMSE | 0.02004 Å | 0.01899 Å |

The new operator is consistently 3-5% worse on endpoint and autonomous multistep
error. It does not reproduce the frozen model's single-step accuracy.

## 100 ps NVE on two independent test starts

| Metric | phase test_09 | R3-Opt test_09 | phase test_10 | R3-Opt test_10 |
|---|---:|---:|---:|---:|
| Stable | yes | yes | yes | yes |
| Mean temperature / K | 311.15 | 298.64 | 288.80 | 304.57 |
| Reference temperature / K | 288.40 | 288.40 | 303.14 | 303.14 |
| Energy drift / meV atom^-1 ps^-1 | 0.0939 | 0.0579 | -0.0015 | 0.0886 |
| Total energy change / eV | 0.904 | 1.302 | -0.330 | 0.689 |
| VDOS overlap O / Si | 0.970 / 0.955 | 0.954 / 0.956 | 0.973 / 0.957 | 0.955 / 0.956 |
| RDF RMSE Si-O / O-O | 0.260 / 0.0266 | 0.274 / 0.0158 | 0.0594 / 0.0354 | 0.202 / 0.0247 |

Both replicas remained stable for 100 ps. Long-time metrics are mixed: the new
operator improves Si-O RDF in both replicas and VDOS overlap in three of four
element-replica comparisons, while O-O RDF is worse in both. Temperature and
energy drift do not improve consistently: one replica is better, the other
worse. No overall physical advantage is established.

## Cost

The pipeline measured 146.9 and 149.5 ns/day end-to-end while the GPU was
otherwise idle. These numbers are **not** comparable to the historical R3-Opt
values (39.2 ns/day), which were produced under heavy shared load. The
same-device benchmark remains the cost reference: one-graph model call
14.6 ms vs 5.5 ms, training step +65%.

## Limits

- One random-initialization run per architecture; no repeat seeds.
- Validation-selected checkpoint only; no test-based selection.
- Two 100 ps NVE replicas, no 15 ns NVT in this pipeline invocation.
- BKS analysis is offline; no force/energy evaluation in inference.
- No claim about the Water motion-bias hypothesis from Quartz outcomes.

Raw records: `summary.json`, `SUMMARY.md`, `evaluation/`,
`physics/nve_test_09`, `physics/nve_test_10` in the run directory.
