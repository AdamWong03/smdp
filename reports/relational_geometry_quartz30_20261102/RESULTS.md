# Quartz 30 fs geometry-relational endpoint result

Status: training and independent endpoint testing completed. No autonomous
NVE/NVT trajectory was run for this checkpoint.

## Fixed experiment

- Data: 5,000 train / 1,250 validation / 1,000 independent test endpoints.
- Model: 160 scalar, 40 vector and 80 edge channels; 2 proposal and 2
  future-conditioned refinement blocks; 1,750,760 parameters.
- Optimization: BF16 autocast, batch 16, seed 20261102, validation-only model
  selection, early-stopping patience 80.
- Hardware: NVIDIA L40S.

Training stopped after 107 epochs. The selected checkpoint was epoch 27. Total
wall time was 1,970.7 s (32.8 min). The selected epoch processed 328.3 samples/s
and allocated a peak 2.263 GB of CUDA memory.

## Results

| Split | Position RMSE (A) | Velocity RMSE (A/fs) | Position MAE (A) | Velocity MAE (A/fs) |
|---|---:|---:|---:|---:|
| Validation, selected epoch | 0.0027791 | 0.00019233 | 0.0022059 | 0.00015241 |
| Independent test | 0.0028135 | 0.00019531 | 0.0022321 | 0.00015460 |

The previous 853,920-parameter legacy relational run was paused before an
independent test evaluation. Its best validation checkpoint (epoch 348) reached
0.0016344 A position RMSE and 0.00011107 A/fs velocity RMSE under the same data
split. Thus the enhanced model's best validation RMSE is 70.0% higher for
position and 73.2% higher for velocity. This is a negative endpoint-prediction
result for the tested architecture/optimization combination. It does not by
itself identify whether the regression arose from the new interactions, wider
model, BF16/batch-16 optimization, or their combination.

The final epoch was not selected: its validation selection loss was 0.0030565,
versus 0.0021800 at epoch 27. Checkpoint selection did not use test data.

## Artifact policy

The 27 MB `best.pt`, 27 MB `last.pt`, full log and complete JSONL history remain
on the server under:

`/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261102_r2_geometry`

They are deliberately not committed to Git. Machine-readable summary metrics
are in `results.json`; the reproducible configuration is in `config.json`.
