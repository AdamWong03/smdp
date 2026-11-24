# H1 — Selection-Proxy Alignment

**Date:** 2026-11-20
**Status:** implemented; 20k-step screening running (endpoint vs endpoint+rollout)
**Worktree:** `/data2/wyd/research/foundation_mlip_adaptation/code/h1h2`
**Branch:** `research/h1h2-selection-20261120`

---

## 1. Why H1

The formal configs select checkpoints on normalized final endpoint position/velocity MSE.
That metric is almost perfect for the frozen Water model on one-step validation states, yet
the same model fails in 2.5 ns NVT. Model selection is therefore blind to the failure the
method must fix.

H1 changes **model selection**, not the architecture:

```
selection = endpoint_loss + w * short-NVE-rollout partition error
```

The rollout proxy is validation-only. It never reads test trajectories.

---

## 2. What was implemented

### 2.1 `selection_metrics.py`

Vectorized molecular kinetic partition in eV:

- translation: molecular COM kinetic energy;
- rotation: angular momentum about the molecular COM with the minimum-image convention;
- internal: total minus translation minus rotation;
- total.

Inputs: positions [N,3], velocities [N,3], masses [N], molecule_ids [N], cell [3,3].
Units: Å, Å/fs, amu, eV. No reference energies are read.

### 2.2 `validation_rollout.py`

- Builds a fixed set of validation starts from the transition HDF5 and the raw NVE
  trajectory.
- Rolls out K deterministic NVE steps with the model (no thermostat, no force calls).
- Compares the final partition with the reference state K steps ahead.
- Returns per-component predicted/reference values and the mean relative error.

The starts are fixed by seed and sample count; test data is never touched.

### 2.3 `train_relational.py`

- `read_graphs` attaches `molecule_ids` from the transition HDF5 (no cache rewrite).
- `evaluate` can compute one-step partition metrics (for diagnostics only).
- The training loop builds the rollout starts once and computes the rollout proxy every
  epoch when `selection.mode` is `endpoint_rollout` or `rollout`.
- Selection score:
  - `endpoint`: original normalized endpoint MSE;
  - `endpoint_partition`: endpoint + w × one-step partition error (diagnostic; not the main
    path);
  - `endpoint_rollout`: endpoint + w × K-step rollout partition error (recommended);
  - `rollout`: rollout error only (diagnostic).
- The best checkpoint, `last.pt` and `status.json` now carry the rollout metrics.

### 2.4 Tests

`tests/test_selection_metrics.py`:

- pure translation → translation = total, rotation/internal ≈ 0;
- rigid rotation → rotation = total, translation/internal ≈ 0;
- internal stretch → internal = total, translation/rotation ≈ 0;
- periodic wrap preserves the partition;
- identical partitions give zero relative error.

All 5 tests pass.

---

## 3. Calibration on the frozen Water R3-Opt checkpoint

The frozen 4.5 Å Water R3-Opt checkpoint (`water5_20261103_r3_opt/best.pt`) was evaluated on
the validation split with the new metric.

### 3.1 One-step partition is almost perfect

| Component | Predicted | Reference |
|---|---:|---:|
| Translation | 2.2756 eV | 2.2747 eV |
| Rotation | 1.9898 eV | 1.9923 eV |
| Internal | 2.5715 eV | 2.5724 eV |
| Total | 6.8369 eV | 6.8394 eV |

Relative error: 0.000655.

**Conclusion:** one-step partition on reference states cannot select against the failure.
The model is already good there. The NVT failure is a long-time rollout phenomenon.

### 3.2 Rollout partition error grows with horizon

4 validation starts, frozen checkpoint:

| Rollout | Time | Relative error | Translation pred/ref | Rotation pred/ref | Internal pred/ref |
|---:|---:|---:|---:|---:|---:|
| 50 steps | 250 fs | 0.0456 | 2.388 / 2.430 | 2.343 / 2.436 | 2.172 / 2.267 |
| 100 steps | 500 fs | 0.1020 | 2.347 / 2.317 | 2.063 / 2.405 | 2.258 / 2.252 |
| 200 steps | 1 ps | 0.1225 | 2.464 / 2.331 | 2.061 / 2.234 | 2.188 / 2.414 |
| 400 steps | 2 ps | 0.1752 | 2.229 / 2.457 | 2.060 / 2.544 | 2.359 / 2.144 |

The frozen model loses rotation and shifts internal/translation over 0.5–2 ps. A 500 fs proxy
is the cheapest signal with a non-trivial effect; 1 ps is more stable but ~2× the cost.

---

## 4. Screening design

Two arms, same data/architecture/seed/budget:

| Arm | Selection | Run |
|---|---|---|
| A | endpoint only | `experiments/h1h2_endpoint_20k_20261120` |
| B | endpoint + 500 fs rollout partition | `experiments/h1h2_rollout_20k_20261120` |

Config:

- Water 6 Å cache (`water6_reference_20261113`)
- DSRT `geometry_gated`, 2.36M parameters
- 20,000 optimizer updates, batch 8, FP32
- 4 validation starts × 100 rollout steps (500 fs)
- partition weight 0.001
- selection only on validation; test split read once after selection

**What would support H1:** Arm B's selected checkpoint has a lower 500 fs/1 ps rollout
partition error than Arm A at similar endpoint RMSE. That would show the selection metric
chooses a physically better checkpoint from the same training trajectory.

**What would falsify H1:** Arm B's selected checkpoint is not better on the rollout proxy, or
its endpoint RMSE degrades enough to dominate the combined score without a physical gain.

---

## 5. Operational note

The first launch used `CUDA_VISIBLE_DEVICES` without `CUDA_DEVICE_ORDER=PCI_BUS_ID`; the
mapping did not match `nvidia-smi` and one arm OOM'd on an occupied RTX 6000. The relaunch
uses:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The endpoint arm runs on L40S GPU 1; the rollout arm on L40S GPU 4. The active phase-space
Water training on GPU 2 is not touched.

---

## 6. 20k A/B result (2026-11-20) — null at this budget

Both arms completed 20,000 updates. The selected checkpoint was **epoch 32 in both arms**:

| Arm | Selected epoch | Test position RMSE | Test velocity RMSE |
|---|---:|---:|---:|
| Endpoint only | 32 | 0.0013679 Å | 0.0004874 Å/fs |
| Endpoint + 500 fs rollout | 32 | 0.0013920 Å | 0.0004899 Å/fs |

The rollout-selection arm did **not** change the selected epoch. The 1.8% endpoint difference is
not a selection effect; both arms trained the same loss with the same seed and both chose the
final epoch.

Why: the combined score used `endpoint + 0.001 × rollout_error`. In the rollout arm's history:

- best endpoint: epoch 32 (`0.001042`);
- best rollout: epoch 27 (`0.1458`);
- best combined: epoch 32 (`0.001203`), because endpoint dominates.

The rollout proxy is also noisy at this budget: 4 starts × 100 steps gives values jumping
between 0.15 and 6.86 across epochs. The endpoint/rollout correlation is 0.638, so the two
signals are not independent enough for a 0.001 weight to change selection.

**Conclusion:** H1 as implemented is a null result at 20k. This is not evidence that selection
is irrelevant; it is evidence that the current proxy/weight/budget cannot discriminate. Do not
scale H1 blindly. If revisited:

- use more starts/steps (e.g., 16 × 200) to reduce noise;
- use a higher weight or a constrained rule (top-k endpoint checkpoints, then lowest rollout);
- save candidate checkpoints at the top-k endpoint epochs so they can be compared;
- test at 50k/250k where endpoint is closer to convergence.

E1 (motion-balanced training) is now the primary method change; H1 can be re-run later with a
better proxy if E1 shows a physical signal.

---

## 7. Files changed

```
code/h1h2/
├── selection_metrics.py          (new)
├── validation_rollout.py         (new)
├── train_relational.py           (selection config, molecule_ids, rollout proxy, logging)
├── tests/test_selection_metrics.py (new, 5 tests)
└── configs/
    ├── smoke_h1h2_20261120.json
    ├── water6_endpoint_20k_20261120.json
    └── water6_endpoint_rollout_20k_20261120.json
```

The active `smdp-future-control` worktree was not modified. All edits are in the separate
`research/h1h2-selection-20261120` branch.
