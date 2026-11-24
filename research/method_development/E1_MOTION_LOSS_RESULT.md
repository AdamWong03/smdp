# E1 Motion-Balanced Loss — Result

**Date:** 2026-11-20
**Status:** negative as implemented; one unexpected signal worth following
**Run dirs:**
- `experiments/e1_baseline_20k_20261120`
- `experiments/e1_motion_20k_20261120`

---

## 1. Setup

| Arm | Training loss | Selection | Data / budget |
|---|---|---|---|
| Baseline | global endpoint q/v MSE | endpoint | Water 6 Å, 20k updates, batch 8, seed 20261120 |
| Motion | mode-balanced q/v MSE (equal weights) | endpoint | same |

Both arms selected epoch 32 (20k). The only intended difference was the training loss.

---

## 2. Endpoint

| Metric | Baseline | Motion | Change |
|---|---:|---:|---:|
| Test position RMSE | 0.0013961 Å | **0.0012630 Å** | −9.5% (better) |
| Test velocity RMSE | **0.0004914 Å/fs** | 0.0005293 Å/fs | +7.7% (worse) |
| Validation endpoint loss | 0.001048 | 0.001123 | +7.2% (worse) |
| Validation motion loss | — | 0.000506 | — |

The motion arm improves position but degrades velocity. The endpoint selection still chose the
final epoch in both arms.

---

## 3. 20 ps NVE (same validation start, val_04 frame 0)

| 0–20 ps | Baseline | Motion | Reference |
|---|---:|---:|---:|
| Translation KE | 2.421 eV | 2.689 eV | 2.215 eV |
| Rotation KE | 1.820 eV | 2.081 eV | 2.236 eV |
| Internal KE | 2.873 eV | **5.396 eV** | 2.338 eV |
| Total KE | 7.114 eV | **10.166 eV** | 6.789 eV |
| Temperature trajectory | ~297 → 271 K | ~417 → 400 K | — |
| O–O RDF RMSE | 0.0493 | 0.0835 | — |
| O–H RDF RMSE | 0.447 | 1.125 | — |
| H–H RDF RMSE | 0.0348 | 0.0638 | — |
| H VDOS overlap | 0.8495 | 0.7364 | — |
| O VDOS overlap | 0.9135 | 0.8849 | — |

**Motion is worse in every physical metric.** It overshoots translation (2.689 vs 2.215),
under-shoots rotation (2.081 vs 2.236), and severely over-excites internal motion (5.396 vs
2.338), with total KE 10.17 vs 6.79 eV. The NVE trajectory heats to ~400–420 K.

---

## 4. Why it failed (hypotheses)

1. **Proposal-loss inflation.** The motion-balanced objective was also applied to the proposal
   velocity, normalized by the full mode RMS. The proposal is a small correction, so its
   normalized loss became huge: validation proposal loss 0.387 vs baseline 0.017. Weighted by
   0.02, it contributed ~0.0077, much larger than the final loss (~0.0005). The model was
   partly optimizing a badly scaled auxiliary term.
2. **Equal mode weights overcorrected translation.** Translation was 7.8% of the original
   velocity MSE; setting it to 1/3 of the loss increased its effective weight ~4.3×. The model
   overshoots translation and destabilizes internal motion.
3. **20k budget.** The baseline itself is not converged (endpoint 0.001396 vs 0.000410 for the
   250k Full model). The loss change was tested in an undertrained regime.

**Conclusion:** E1 as implemented is a negative result. Do not scale it. The mode-balanced
idea is not disproven, but this implementation is confounded and too aggressive.

---

## 5. Unexpected signal: 20k baseline is physically better than 250k Full

Compare the 20k baseline with the frozen 250k Full 6 Å model on the same 20 ps NVE start:

| 0–20 ps | 20k baseline | 250k Full 6 Å | Reference |
|---|---:|---:|---:|
| Endpoint position RMSE | 0.001396 Å | **0.000410 Å** | — |
| Translation KE | 2.421 eV | 1.366 eV | 2.215 eV |
| Rotation KE | 1.820 eV | 1.538 eV | 2.236 eV |
| Internal KE | **2.873 eV** | 4.418 eV | 2.338 eV |
| Total KE | **7.114 eV** | 7.321 eV | 6.789 eV |
| H VDOS overlap | **0.8495** | 0.7173 | — |
| O VDOS overlap | **0.9135** | 0.8837 | — |

The 20k model has **worse endpoint but better physical partition** than the 250k model. This is
not a controlled seed comparison (different seeds and budgets), but the direction is striking:
**longer endpoint training may degrade long-time physical fidelity.**

If this survives a controlled training-time sweep, the main lever is **objective/selection
alignment**, not "train less". Related work trains for a long time (TrajCast up to 1,500 epochs;
FlashMD long-stride training), so "long training is bad" is not a general claim. The plausible
mechanism is that our objective (global q/v MSE + proposal loss) and endpoint-only selection
push the model toward endpoint-optimal but physically worse states as training proceeds.
Related methods may also train long without this degradation because their loss/selection is
better aligned, or because they do not report molecular kinetic partition at all. H1 was tested
at 20k, where endpoint and physics still improve together; if they diverge at 250k, a physical
proxy can matter there. Early stopping is a diagnostic/control, not the proposed method.

---

## 6. Next experiments

### E1b (optional, low priority)
Fix the proposal loss (keep original global normalization or proposal-specific RMS), reduce
translation weight (e.g., 0.25–0.5), train 20k. Only if the training-time sweep says the loss
matters.

### E0 (higher priority): controlled training-time sweep
Train the 6 Å Full model with checkpoint saves at 5k/10k/20k/50k/100k/250k (same seed/data).
Evaluate each checkpoint on the same 20 ps NVE and physical panel. This directly tests whether
physical fidelity peaks before endpoint convergence.

If confirmed:
- design **physics-aware selection/early stopping** (H1 with a better proxy, at 250k);
- use the endpoint–physics divergence as the method paper's motivation;
- do not claim a new operator; the method contribution is the training/selection rule plus the
  mechanism.

### E2 (later)
If the sweep shows the loss is still relevant after selection, revisit mode-aware readout with
a corrected objective.

---

## 7. What not to claim

- Do not claim the motion-balanced loss improves physics; it does not in this run.
- Do not claim the 20k vs 250k comparison is controlled; it is a signal, not a result.
- Do not scale E1 to 250k; fix the objective first.
- Do not write a paper on the current evidence.
