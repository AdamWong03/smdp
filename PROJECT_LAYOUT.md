# Project layout

## Repository

- `relational_transition.py`, `relational_inference.py`: frozen DSRT method.
- `train_relational.py`: formal supervised training.
- `rollout_relational.py`, `evaluate_relational_rollout.py`: autonomous use.
- `physics/`, `analysis_tasks.py`, `periodic.py`: shared physical evaluation.
- `analyze_water.py`, `physics/spcfw.py`, `physics/water_motion.py`: Water
  reference energy, molecular motion and long-trajectory statistics.
- `diagnose_water_bias.py`: conditional signed errors and evaluation-only labels.
- `configs/`: versioned formal configurations.
- `paper/`: manuscript, evidence registry and frozen experiment matrix.
- `reports/stable_geometry_optimization/`: evidence for the frozen Quartz model.
- `tests/`: active DSRT correctness tests.
- `upstream/`: pinned third-party source needed by the data/model environment.

Historical architectures and generated plots are intentionally absent from the
active layout. They remain in Git history rather than in the working tree.

## Server-only assets

```text
data/smdp_reference_md_v1/
  quartz/
    raw_nve/                 independent 1 fs reference trajectories
    transitions/30fs/       formal endpoint datasets
    physical_reference/     long NVT and short NVE references
    derived_graphs/          regenerable model-ready graph cache
  water/
    raw_nve/
    transitions/5fs/
    physical_reference/

experiments/relational_transition/
  quartz30_20261102_r1/      retained lightweight reference best checkpoint
  quartz30_20261103_r3_opt/  frozen paper checkpoint and physical trajectories
  water5_20261103_r3_opt/   Water checkpoint, NVE diagnosis and active NVT
```

Only `best.pt` is retained for completed training. Active trajectories must not
be modified while their systemd service is running.
