# SMDP / TrajCast infrastructure / Quartz 30 fs

All SMDP parameters start from random initialization. No pretrained weights or additional baseline training.
Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028`.
Method and commands: [README](../../trajcast_smdp/README.md).

Training started: True.
Training completed: False.
Independent test evaluation completed: False.

Execution snapshot: 2026-11-02T09:38:59.735172+00:00; service {'MainPID': '90130', 'ActiveState': 'deactivating', 'SubState': 'stop-sigterm'}.
Free disk at snapshot: 88.51 GiB; trajectory storage is checked against requested duration before launch.
See progress_snapshot.json for the last actual training record. Early training loss is not a physical result.

## Training and validation progress

These are validation results, not independent test or long-time physical results.
Each row uses the best checkpoint within that stage by the unchanged selection score.

| Stage | Completed epochs | Best epoch (zero-based) | q30 RMSE / Å | v30 RMSE / Å/fs | q240 RMSE / Å | Selection score |
|---|---:|---:|---:|---:|---:|---:|
| segment | 400 | 389 | 0.001867054 | 0.0001525811 | 0.0188368 | 0.02115413 |
| rollout4 | 57 | 52 | 0.001864529 | 0.0001505272 | 0.01662013 | 0.01781902 |

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
