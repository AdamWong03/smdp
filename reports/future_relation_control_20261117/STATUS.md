# Full vs Current-Only: measured status

Recorded UTC: 2026-11-17T13:48:42.628208+00:00

Current-Only: analysis_incomplete; 400 epochs / 250,000 of 250,000 updates recorded.

| Model | Updates | Best epoch | Test q RMSE (Å) | Test v RMSE (Å/fs) |
|---|---:|---:|---:|---:|
| full | 250000 | 398 | 0.000409728997 | 0.000149672884 |
| current_only | 250000 | 393 | 0.000401362641 | 0.000151308838 |

A running model is not a completed comparison. This is one seed and one
paired validation NVE start; no general mechanism claim is made automatically.
Full trained on shared L40S; Current-Only on shared Blackwell. Optimization
budgets match, wall time and bitwise GPU arithmetic do not.

The strict Water reference energy verification remains separate. A numerically
completed trajectory or completed motion/RDF/VDOS task does not certify energy
analysis or physical fidelity. Do not use the legacy two-frame full-energy
result to bypass failed eight-frame verification.

[Protocol and reproduction](PROTOCOL.md). Full logs and checkpoints remain
at the absolute server paths recorded in snapshot.json.
