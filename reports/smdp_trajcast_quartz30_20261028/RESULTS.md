# SMDP / TrajCast infrastructure / Quartz 30 fs

All SMDP parameters start from random initialization. No pretrained weights or additional baseline training.
Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028`.
Method and commands: [README](../../trajcast_smdp/README.md).

Training started: True.
Training completed: False.
Independent test evaluation completed: False.

Execution snapshot: 2026-11-01T14:17:58.297195+00:00; service {'MainPID': '90130', 'ActiveState': 'active', 'SubState': 'running'}.
Free disk at snapshot: 15.45 GiB; dense NVT requires 40 GiB before launch.
See progress_snapshot.json for the last actual training record. Early training loss is not a physical result.

## Training and validation progress

These are validation results, not independent test or long-time physical results.
Each row uses the best checkpoint within that stage by the unchanged selection score.

| Stage | Completed epochs | Best epoch (zero-based) | q30 RMSE / Å | v30 RMSE / Å/fs | q240 RMSE / Å | Selection score |
|---|---:|---:|---:|---:|---:|---:|
| segment | 400 | 389 | 0.001867054 | 0.0001525811 | 0.0188368 | 0.02115413 |
| rollout4 | 7 | 5 | 0.001909675 | 0.0001540541 | 0.01801787 | 0.02013674 |

Full stage-best/latest validation traces: [validation_progress.json](validation_progress.json).
Server checkpoints: `<server artifacts>/<stage>/best.pt`, `<stage>/last.pt`, and `best.pt`.
Weights, raw logs and trajectories are not uploaded. Training is still running unless marked complete above.

## nve_0

Not completed; no physical results claimed.

## nve_1

Not completed; no physical results claimed.

## nvt_0

Not completed; no physical results claimed.

No phonon identification, stability improvement or speedup is inferred from architecture alone.
Official-paper BKS energy discrepancies remain recorded in the foundation report. This experiment compares only against its own independently generated, unchanged BKS reference.
