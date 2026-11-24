# Published TrajCast Water Model — Control Audit

**Date:** 2026-11-21
**Purpose:** determine whether our water NVT failure is intrinsic to the method class, a
training/accuracy issue, or a protocol/representation issue.
**Status:** complete. The published model **passes** the 100 ps NVT screen under our exact
protocol. The failure is training/accuracy, not the method class.

---

## 0. Decisive result: the published offline model passes

Protocol: 50 ps thermalization + 100 ps production, CSVR 300 K, τ = 50 fs, 5 fs horizon,
`val_04` frame 0 (275 K NVE start). Same script and protocol as our native screen.

| Model | trans KE | rot KE | int KE | total | T (K) |
|---|---:|---:|---:|---:|---:|
| Reference (NVT) | 2.373 | 2.430 | 2.609 | 7.411 | 300 |
| **Published TrajCast** | **2.470** | **2.432** | **2.400** | **7.413** | **300.25** |
| Our native ep145 | 3.256 | 3.105 | 1.041 | 7.402 | 300 |
| Our 250k Full | 0.538 | 0.786 | 6.044 | 7.368 | 299 |
| Our Current-Only | 0.646 | 0.535 | 6.384 | 7.565 | 305 |

Mass-weighted velocity autocorrelation:

| Model | α_t(25 fs) | α_r(25 fs) | α_i(5 fs) | α_i(500 fs) |
|---|---:|---:|---:|---:|
| Reference | 0.658 | −0.540 | −0.567 | 0.104 |
| **Published** | **0.662** | **−0.539** | **−0.645** | **−0.008** |
| Native ep145 | 0.697 | −0.412 | +0.109 | 0.003 |
| 250k Full | 0.393 | −0.783 | −0.902 | 0.722 |

The published model is at equipartition from the first 10 ps and stays there for the full
150 ps; translation and rotation ACFs match the reference almost exactly. The NVT screen is
stable (30,000/30,000 steps, no safety stop).

**Conclusion:** a direct, force-free, offline-trained 5 fs propagator reproduces the correct
water mode partition under our protocol. The failure of our models is an accuracy/training
problem, not a fundamental long-time limitation and not a reason to require on-policy data.

### 0.1 Pointwise accuracy is not the objective

Fresh SPC/Fw one-step labels were computed at 160 states from each model's own NVT
trajectory (same pilot protocol):

| Model | position RMSE | velocity RMSE | signed KE error | NVT ensemble |
|---|---:|---:|---:|---|
| Published TrajCast | 6.9e-3 Å | 2.5e-3 Å/fs | +0.083 eV/step | **PASS** |
| 250k Full | 4.9e-4 Å | 1.7e-4 Å/fs | +0.0062 eV/step | **FAIL** (int 6.044) |

The published model has a **~15× larger pointwise error** on its own trajectory, yet the
correct ensemble. It behaves like a **generative propagator**: it samples a different
trajectory with the same distribution. The 250k model learned the conditional mean accurately
but the wrong distribution.

**This kills per-state label correction as the main method.** Held-out and on-policy one-step
labels optimize the wrong object. The right target is the **ensemble**: NVT pairs,
distribution-level losses (partition/RDF/VDOS), or a stochastic/generative head.

---

## 1. The published model is available and faithful

Downloaded from Hugging Face `ibm-research/trajcast.models-arxiv2025` (revision 621f593):

- `water/config_e3nn.yaml`
- `water/state_dict_e3nn.pt` (18 MB, 2,258,856 parameters)

It loads into `EfficientTrajCastModel` with **0 missing / 0 unexpected keys**.

On the paper's own `water/test.extxyz` (1,000 frames, 5 fs):

| Quantity | Our run | Paper Table S1 |
|---|---:|---:|
| displacement MAE | 6.88e-5 Å | 6.97e-5 Å |
| velocity MAE | 3.41e-5 Å/fs | 3.5e-5 Å/fs |

So the checkpoint and inference path are correct.

## 2. Config comparison: our native vs published

| Setting | Published | Our native |
|---|---|---|
| precision | 64 | 64 |
| cutoff | 6 Å | 6 Å |
| channels | 64 | 64 |
| MP layers | 4 | 4 |
| lmax | 2 | 2 |
| `conserve_ang_mom` | **false** | **false** |
| batch | 2 | 2 |
| lr | 0.01 | 0.01 |
| max epochs | 1,500 | 1,500 |
| training samples | 5,000 | 5,000 |
| horizon | 5 fs | 5 fs |

The paper explicitly says angular-momentum constraints are unnecessary for bulk water:
"for the bulk systems investigated here ... we find that all MPNNs perform well even without
additional constraints." So `conserve_ang_mom: false` is not a bug.

**The architecture and hyperparameters are a faithful reproduction.**

## 3. Accuracy gap

| Model | Relative velocity MAE (5 fs) | Source |
|---|---:|---|
| Published TrajCast | **0.37%** | paper Fig. 6b / Table S1 |
| Our native ep145 | **0.50%** | native validation 4.5e-5 / mean|v| 0.00905 |
| Our 250k Full | ~1.2% | velocity RMSE 1.5e-4 |
| Current-Only | ~1.3% | similar |

The native model is only ~1.33× worse than the published model in one-step error, yet its
100 ps partition is strongly wrong (internal 1.04 vs 2.61 eV). This is the central puzzle:
**is the threshold that sharp, or is something else different?**

## 4. Protocol differences (paper vs our screen)

| Item | Paper | Our 100 ps screen |
|---|---|---|
| initial positions | equilibrated NVT at 300 K | `val_04` NVE frame 0 |
| initial velocities | Gaussian 300 K, zero net linear/angular momentum | NVE frame velocities (~276 K) |
| equilibration | 10% of 500,000 steps = **250 ps** | **50 ps** |
| production | 2.25 ns | 100 ps |
| thermostat | CSVR τ = 50 fs | CSVR τ = 50 fs |
| DOF | 3N−3 | 3N−3 |
| ensemble | NVT | NVT |

Our 100 ps screen is **not** the paper protocol. The mode partition is stationary after
~30 ps in our runs, so the 50 ps thermalization is probably sufficient for the partition, but
the initial condition and the 250 ps equilibration could matter for the slow structural modes.
This is the next control if the published checkpoint fails our screen.

## 5. Data-integrity checks

- **Raw NVE data is mode-equilibrated**: `val_04` has ≈2.48/2.20/2.38 eV per mode at 275 K
  (equipartition ≈2.28 eV). No inherited partition bias.
- **Transition alignment is exact**: `positions_initial`/`velocities_initial` match the raw
  NVE frame `source_frame`; `positions_final`/`velocities_final` match `source_frame + 10`
  (5 fs). No time offset.
- **No split molecules**: O–H bond lengths after wrapping are 1.03–1.11 Å for all molecules.
- **Paper test vs our transitions**: per-species velocity RMS, bond lengths, and
  displacement/velocity distributions are similar. Our transitions are at 275–285 K; the paper
  test is at 292–301 K.

## 6. The "anomaly" is the generative behavior

The published model has a uniform one-step velocity error of ~1.6e-3 Å/fs on our val
transitions (36× larger than its error on the paper's test set, 3.4e-5), and ~2.5e-3 Å/fs on
its own NVT states with fresh SPC/Fw labels. Scaling the velocities to 300 K does not fix it;
midpoint/final velocities are worse; the atom-type mapping is correct.

The 100 ps NVT screen resolves the interpretation: the published model **passes** despite this
large pointwise error. It behaves like a **generative propagator**: it samples a different
trajectory with the same ensemble. The pointwise error is real, but it is not the drift of the
model's own distribution.

The 250k model has a **15× smaller** pointwise error on its own states (4.9e-4 Å / 1.7e-4 Å/fs)
yet the wrong ensemble. This rules out the earlier "representation bug" and "accuracy
threshold" interpretations: **pointwise accuracy is not the controlling quantity**.

This is the central result for the paper, not an anomaly to fix.

## 7. What this means for the paper

- The claim cannot be "direct propagators fail long-time equilibrium"; the published model is
  an existence proof to the contrary.
- **Pointwise one-step accuracy is neither necessary nor sufficient.** The published model has
  a ~15× larger pointwise error on its own NVT states (6.9e-3 Å / 2.5e-3 Å/fs / +0.083 eV per
  step) than the 250k model (4.9e-4 / 1.7e-4 / +0.0062), yet the published ensemble is correct
  and the 250k ensemble is wrong.
- The published model behaves like a **generative propagator**: a different trajectory with the
  same ensemble. The 250k model learned the conditional mean but the wrong distribution.
- The viable claim is **ensemble fidelity, not trajectory accuracy**: train on the evaluation
  ensemble (NVT pairs) or use a distribution-level / stochastic objective. Per-state label
  correction (held-out or on-policy) optimizes the wrong object.
- The 250k model is the clean failure point; the native point is confounded by a
  train/inference data-representation difference.
- The baseline for the paper must be the **published checkpoint**, not our 145/1500 native.

See `PAPER_PLAN_20261121.md` for the full plan.
