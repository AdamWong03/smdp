# SMDP / TrajCast infrastructure / Quartz 30 fs

All SMDP parameters start from random initialization. No pretrained weights or additional baseline training.
Server artifacts: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_segment_physics_20261101`.
Method and commands: [README](../../trajcast_smdp/README.md).

Training started: False.
Training completed: False.
Independent test evaluation completed: False.

Execution snapshot: 2026-11-01T14:38:53.366342+00:00; service {'MainPID': '3323649', 'ActiveState': 'active', 'SubState': 'running'}.
Free disk at snapshot: 15.37 GiB; trajectory storage is checked against requested duration before launch.
See progress_snapshot.json for the last actual training record. Early training loss is not a physical result.

## Frozen segment checkpoint evaluation

Source: `/data1/wyd/research/foundation_mlip_adaptation/experiments/trajcast_smdp/quartz30_20261028/segment/best.pt`; SHA256 `36c0789a3919211fc14df643c7daa2c4a82940e29dd9ecca1ac81a5244b572dd`.
This run performs no training. It uses the segment-only validation-selected checkpoint, not rollout-trained weights.
Requested: two 100 ps NVE runs; NVT 1.5 ns thermalization + 1 ns production. This is not a 15 ns result.
NVT production length is limited by available storage. Dynamics, thermostat and analysis conventions are unchanged.
30 fs spectra have a 16.67 THz Nyquist limit; 1 fs queried spectra are evaluated separately.
BKS absolute energies use the same raw DSF convention for predicted and reference configurations; no centering.

## nve_0

Not completed; no physical results claimed.

## nve_1

Not completed; no physical results claimed.

## nvt_0

Not completed; no physical results claimed.

No phonon identification, stability improvement or speedup is inferred from architecture alone.
Official-paper BKS energy discrepancies remain recorded in the foundation report. This experiment compares only against its own independently generated, unchanged BKS reference.
