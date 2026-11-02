# Quartz 30 fs gated-geometry relational result

The run was stopped after epoch 138 because the validation optimum occurred at
epoch 35 and was not improved for the following 103 epochs. The selected model
was fixed before the independent test split was evaluated.

## Configuration

- 192 scalar / 48 vector / 96 edge channels.
- Four gated geometry blocks: two proposal and two future-conditioned
  refinement blocks.
- 2,357,616 parameters, full FP32, batch 8, seed 20261102.
- 5,000 train / 1,250 validation / 1,000 independent test endpoints at 30 fs.
- Main normalized endpoint loss plus proposal auxiliary coefficient 0.02.

The best model was selected at epoch 35 (21,875 optimizer updates). Training was
manually stopped at epoch 138 (86,250 updates). Total elapsed time was 6,488.8 s
on an L40S shared with another training process, so this run is not a clean
throughput benchmark.

## Endpoint results

| Split | Position RMSE (A) | Velocity RMSE (A/fs) | Position MAE (A) | Velocity MAE (A/fs) |
|---|---:|---:|---:|---:|
| Validation | 0.0019598 | 0.00013095 | 0.0015539 | 0.00010392 |
| Independent test | 0.0019883 | 0.00013307 | 0.0015756 | 0.00010556 |

For context, R1's best validation RMSE was 0.0016344 A / 0.00011107 A/fs;
R1 was paused without an independent test result. R2's best validation RMSE was
0.0027791 A / 0.00019233 A/fs. R3 therefore recovers most of R2's regression but
does not match R1 on validation accuracy.

The result supports a narrower conclusion than either accepting or rejecting
the geometry architecture: R3 converged early under this optimizer setup, and
continuing from 21,875 to 86,250 updates did not recover the optimum. It does
not show that a larger from-scratch network is intrinsically worse, nor does it
establish long-term dynamics because NVE/NVT propagation has not been run.

## Artifacts

The complete JSONL history, logs and 2.36M-parameter checkpoints remain under:

`/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r3_gated_geometry_fp32`

Large checkpoints and logs are not committed. `results.json` contains the
machine-readable endpoint metrics and exact server paths.
