# Paper Plan — Ensemble Fidelity, Not Trajectory Accuracy

**Date:** 2026-11-21
**Mode:** academic-paper `plan` (method/positioning; not a draft)
**Target:** Nature Machine Intelligence, method paper
**Status:** **GLR v2 (AR(10) memory kernel) passes 100 ps and is initial-condition independent**
(O–O 0.032–0.036, O–H 0.098–0.104, H–H 0.026–0.027; VDOS 0.848–0.851 / 0.963–0.964), with the
canonical partition guaranteed. Only the 2.5 ns **diffusion** remains before the core method is
scientifically complete for water. The positioning is close to lockable around **"backbone
structure + mode-resolved generalized-Langevin thermodynamics"**, but not locked until diffusion
passes.

---

## 0. Why the earlier on-policy framing was wrong

The question is not "on-policy or not." TrajCast is **offline**, trained only on NVE
trajectories, and reports faithful water equilibrium at 2.5 ns. On-policy correction (DAgger /
Dynamic Training) is one way to reduce the state-dependent error; it is not a necessary
condition.

The right variable is the **size and structure of the one-step error relative to the
closed-loop amplification**. A deterministic propagator composed with a thermostat is a
closed-loop map. Its stationary mode partition is sensitive to the mode-resolved one-step
error, with an amplification factor of order `1/(1-α²)`:

| Mode | one-step gain α | amplification `1/(1-α²)` | sensitivity `2α/(1-α²)` |
|---|---:|---:|---:|
| translation | 0.981 | 27.0 | 53 |
| rotation | 0.837 | 3.34 | 5.6 |
| internal | −0.393 | 1.18 | 0.93 |

The translation mode is ~50× more sensitive than the internal mode. Global endpoint MSE is
dominated by internal motion and therefore a poor predictor of NVT fidelity.

**Hypothesis:** there is a sharp **mode-resolved accuracy threshold**. Below it, the
propagator's stationary partition is faithful; above it, the partition drifts. TrajCast sits
below; our models sit above.

---

## 1. Evidence available now

| Model | Relative velocity MAE (5 fs) | 100 ps NVT internal KE (ref 2.61 eV) | Partition verdict |
|---|---:|---:|---|
| IBM published TrajCast (water) | **0.37%** (3.5e-5 Å/fs) | **2.44** (near equipartition; trans 2.51, rot 2.46) | **passes** |
| Our native ep145 (same architecture/config) | **0.50%** (4.5e-5 Å/fs) | 1.04 (−58%) | fails (possible graph-representation confound) |
| Our 250k Full (6 Å) | ~1.2% (RMSE 1.5e-4) | 6.04 (+144%) | fails |
| Current-Only 250k | ~1.3% | 6.38 (+157%) | fails |
| Reference NVT | — | 2.61 | — |

The published model is at equipartition from the first 10 ps and stays there for 150 ps; the
translation/rotation ACFs match the reference (0.662 vs 0.658; −0.539 vs −0.540). It also
matches the reference O–O RDF much better than our models (RMSE 0.043 vs 0.180 for native).

Other facts established:

- Published config and our native config **match exactly** (64 channels, 4 layers, lmax 2,
  cutoff 6 Å, batch 2, lr 0.01, 1,500 epochs, `conserve_ang_mom: false`). The paper explicitly
  says angular-momentum constraints are unnecessary for bulk water. This is not a config bug.
- Our raw NVE training data is **mode-equilibrated** (≈2.2/2.2/2.3 eV per mode at 275 K), so
  the bias is not inherited from the training data.
- The 100 ps partition is **stationary** after ~30 ps (native: flat 3.25/3.10/1.04 from 30 to
  150 ps; 250k: flat ~0.5/0.8/6.0 from 50 to 150 ps). The 100 ps screen is not a transient.
- Our VDOS analysis agrees with the paper's MD reference for the low-frequency O modes
  (cosine 0.996; peak 50.1 cm⁻¹). The H stretch is sensitive to the water model/sampling and is
  not a reliable discriminator.
- The published model loads and reproduces the paper's test MAE on the paper's own test set
  (6.88e-5 Å / 3.41e-5 Å/fs vs reported 6.97e-5 / 3.5e-5).

---

## 2. Falsifiable core claim

**C1 (main, 2026-11-21). Pointwise one-step accuracy is neither necessary nor sufficient for
long-time ensemble fidelity.**

Pointwise error on each model's **own** NVT states, with fresh SPC/Fw one-step labels:

| Model | position RMSE | velocity RMSE | signed KE error | NVT ensemble |
|---|---:|---:|---:|---|
| Published TrajCast | 6.9e-3 Å | 2.5e-3 Å/fs | +0.083 eV/step | **PASS** (trans 2.505, rot 2.463, int 2.444) |
| 250k Full | 4.9e-4 Å | 1.7e-4 Å/fs | +0.0062 eV/step | **FAIL** (int 6.044) |

The published model has a **~15× larger pointwise error** on its own trajectory yet the
correct ensemble. It behaves like a **generative propagator**: it samples a different
trajectory with the same distribution. The 250k model learned the conditional mean accurately
but the wrong distribution.

**Consequence:** per-state one-step labels — held-out or on-policy — are the **wrong training
target** for ensemble fidelity. On-policy DAgger / signed-KE correction is **not** the main
route. The right target is the ensemble: NVT pairs, distribution-level losses
(partition/RDF/VDOS), or a stochastic/generative head.

**C2.** A deterministic force-free model can match the canonical ensemble if trained on the
evaluation ensemble; no stochastic head is necessarily required. M0 tests this directly.

**C3.** If M0 is insufficient, a distribution-level loss or a stochastic head is needed. This
connects to Timewarp/ITO/MDGen, but keeps deterministic force-free inference if possible.

The earlier "mode-resolved accuracy threshold" is a **secondary diagnostic**, not the main
axis: the published model shows that pointwise accuracy is not necessary, and the 250k shows
that it is not sufficient.

### M0. Ensemble-matched training

Build 5 fs pairs from the **reference NVT trajectory** and fine-tune the propagator on those
pairs (endpoint loss). No new labels, no approval, no on-policy reference calls. The model
then learns the thermostatted map on the evaluation ensemble instead of the NVE map on the
training ensemble. Test: 250k Full fine-tune → 100 ps NVT screen.

This is the cleanest test of C2: if the bias is a train/eval ensemble mismatch, M0 should move
the partition without changing the architecture.

---

## 3. Method candidates (in order)

**M0 (ensemble-matched training)** is described above and is the first experiment: fine-tune
on 5 fs pairs from the reference NVT trajectory, 100 ps screen. It is fully offline and needs
no approval.

The earlier candidates below are now **secondary**, because per-state labels optimize the wrong
object (see C1):
- **M1** sensitivity-weighted mode error is a within-ensemble diagnostic, not the main fix.
- **M2** architecture/inductive bias is the fallback if M0 is insufficient.
- **M3** on-policy signed-KE is **deprecated as the main route**; it can be a diagnostic but not
  the method.

### M1. Sensitivity-weighted mode error (offline; no approval)

Replace the global velocity MSE with a mode-resolved objective:

```
L = L_endpoint + λ Σ_mode w_mode · MSE_mode(v_pred, v_ref)
w_mode ∝ (2 α_mode / (1 - α_mode²)) · Var_mode / σ_mode²
```

The weights are computed from the training split once (α from the data, Var from the data).
This is **not** the E1 motion-balanced loss: E1 equalized modes and heated the system. M1
down-weights internal and up-weights collective according to closed-loop sensitivity.

Test: fine-tune 250k Full (and native ep145) for 5–20k steps; 100 ps NVT. Success: partition
moves toward reference without RDF/VDOS/diffusion collapse. Cost: 2–4 h per arm.

### M2. Architecture/accuracy route (offline)

If M1 does not move the needle, the controlling error is capacity/inductive bias, not loss
weighting. Compare:
- native architecture at convergence (published checkpoint) vs our simpler architectures;
- linear-momentum conservation and velocity tensor-product coupling ablations;
- model size / training length scaling.

The paper then becomes "which inductive biases set the accuracy threshold," with TrajCast as
the existence proof and our leaner model as the contribution (if it matches at lower cost).

### M3. On-policy fallback (requires approval)

Only if M1/M2 show that the threshold cannot be reached offline at reasonable cost. DAgger /
Dynamic Training on visited states, with the signed mode-KE loss. This is a fallback, not the
default.

---

## 4. Experimental matrix

| Arm | Base | Objective | Training | 100 ps NVT | Purpose |
|---|---|---|---|---|---|
| A0 | published TrajCast | — | — | **passed** | threshold upper bound |
| A1 | native ep145 | — | — | existing | just above threshold |
| A2 | 250k Full | — | — | existing | far above threshold |
| B0 | 250k Full | ensemble-matched (NVT pairs) | 5k / 20k | yes | C1″ test, no approval |
| B1 | 250k Full | sensitivity-weighted | 5k / 20k | yes | C3 test |
| B2 | native ep145 | sensitivity-weighted | 5k / 20k | yes | C3 test, matched architecture |
| C1 | native ep145 | uniform MSE | +100 epochs | yes | training-length control |
| C2 | native ep145 | uniform MSE | +300 epochs | yes | convergence control |
| D1 | 250k Full | E1 equal-weight | existing | existing | negative control |

Primary metric: mode partition (translation / rotation / internal) in 100 ps production.
Secondary: O–O RDF RMSE, O VDOS overlap, diffusion (2.5 ns only if 100 ps passes), stability.

---

## 5. Positioning against related work

| Prior work | What it establishes | What it does not answer | Our delta |
|---|---|---|---|
| TrajCast (NMI 2026) | A 5 fs force-free propagator can reproduce water VDOS/RDF/diffusion at 2.5 ns | Does not explain why pointwise accuracy is not predictive; no ensemble-training rule | Pointwise-vs-ensemble evidence + ensemble-matched training |
| Dynamic Training (Žugec 2025) | On-policy relabeling improves long MD | Requires reference calls at visited states; optimizes per-state labels | We show per-state labels are the wrong object; offline ensemble training |
| Timewarp / ITO / MDGen | Stochastic/generative kernels can target equilibrium | Cost/complexity; not a deterministic finite-time propagator | Deterministic force-free inference can be ensemble-faithful if trained on the ensemble |
| Hamiltonian / symplectic NN | Structure guarantees conservation | Requires forces/energy; not force-free | Ensemble criterion applies to force-free maps; structural priors as fallback |
| MLIP active learning | Data coverage improves stability | Does not target the invariant measure | Ensemble-level criterion and NVT-pair training |

**One-sentence claim:** *Pointwise one-step accuracy is neither necessary nor sufficient for
long-time equilibrium; force-free propagators should be trained and evaluated on the ensemble,
not the trajectory.*

---

## 6. Decision tree

1. **M0 (NVT-pair fine-tune) moves the partition toward equipartition** → method lead; go to
   2.5 ns validation. This is the first experiment.
2. **M0 null** → try a distribution-level loss (partition/RDF/VDOS) or a stochastic/generative
   head (M2). Per-state labels stop.
3. **Published checkpoint fails under a fair protocol** → audit the graph/data representation
   and the analysis before any method claim.
4. **All ensemble objectives null** → the architecture/capacity is the limit; change the
   backbone.

Per-state on-policy correction (M3) is a **diagnostic**, not a method candidate: the published
model has a larger per-state label error and the correct ensemble.

---

## 7. Paper outline (if C1–C3 hold)

1. Introduction: force-free propagators promise large speed-ups; the default one-step MSE
   objective assumes trajectory accuracy is the goal; long-time equilibrium is the open failure
   mode.
2. Results 1: two models with opposite pointwise accuracy and opposite ensemble outcomes — the
   published model is ~15× worse pointwise yet passes; our 250k is better pointwise yet fails.
3. Results 2: pointwise error is not predictive; the published model behaves generatively
   (different trajectory, same distribution); the right object is the ensemble.
4. Results 3: ensemble-matched training (M0) moves the partition; 100 ps and 2.5 ns validation
   (partition, RDF, VDOS, diffusion).
5. Results 4: ablations — NVE vs NVT pairs, distribution-level loss, stochastic head, on-policy
   label diagnostic.
6. Discussion: when deterministic force-free propagators suffice; relation to stochastic and
   structure-preserving methods.

---

## 8. What would falsify the paper

- **M0 (NVT-pair fine-tune) does not move the partition** while the 100 ps screen is valid.
- **A controlled accuracy sweep shows pointwise error *does* predict the ensemble** across
  models in the same pipeline (then the pointwise story, not the ensemble story, is right).
- The published model fails under a fair protocol and analysis.
- The 2.5 ns result does not follow the 100 ps screen.
- Ensemble-level training improves the partition only by collapsing RDF/VDOS/diffusion.

Each falsifier is a legitimate negative result and will be recorded.

---

## 9. Why the positioning is not locked yet

The current "ensemble fidelity, not trajectory accuracy" framing is a **working hypothesis**.
It has the following unresolved flaws, and the paper should not be locked until they are
addressed or explicitly scoped.

1. **Data leakage / same-trajectory overfitting.** M0 trained on the same 2.5 ns reference NVT
   trajectory used for evaluation. The independent-start check shows the **bias is the same**
   (trans 3.41, rot 3.21, int 1.01; T 313.8 K), so the bias is not a same-trajectory artifact.
   But the independent start **fails at 130 ps** (preT 3798 K), so the model is not stable.
   Same-trajectory leakage is not the main flaw; stability/generalization is worse than expected.
2. **Double thermostatting / NVE energy is a red herring.** M0 trains on post-CSVR 5 fs pairs
   and applies CSVR again at inference, but the 25 ps NVE control shows the published model
   heats *more* (302 → 476 K) than M0/base yet passes NVT; the base is the most NVE-stable yet
   NVT-wrong. **Do not add an NVE energy-drift penalty.** The relevant property is the composed
   propagator + CSVR stationary measure.
3. **Conditional mean vs distribution.** Endpoint MSE still learns a conditional mean. It may
   fix the partition but leave RDF/VDOS/diffusion wrong. The published model's generative
   behavior suggests a stochastic component may be required. **M0 confirms this:** it improved
   O–O RDF (0.646 → 0.191) and O VDOS (0.487 → 0.888) but overshot the partition
   (trans 3.37, rot 3.27, int 1.00; T=311.6 K). Endpoint NVT pairs are not enough.
4. **Short-rollout ensemble moments can produce an unstable map.** The 5-step CSVR rollout
   loss drove the validation moment 3.25 → 0.028 but the endpoint 7.8e-5 → 0.041; the 100 ps
   NVT failed at **4.3 ps** (min 0.647 Å, preT 1473 K). Stability must be constrained explicitly
   or built into the architecture; a local moment loss can find a partition-correct but
   dynamically invalid map.
4. **Novelty.** "Train on NVT data" is not new by itself (MLIPs and generative models do it).
   The delta must be the **pointwise-vs-ensemble decoupling** and a principled ensemble
   objective, not a data swap. Otherwise reviewers will call it obvious.
5. **Thin evidence.** The decoupling is based on one published model and one 250k model, with
   different architectures, data, and training lengths. A controlled sweep within one pipeline
   is needed.
6. **Native confound.** The native 0.50% point has a train/inference representation difference;
   it cannot support the threshold or accuracy arguments.
7. **Pipeline/label uncertainty.** The published model's large pointwise error is measured with
   our SPC/Fw labels and graph pipeline. It is consistent with generative behavior, but the
   label pipeline must be validated on the paper's own test data (where it gives 3.4e-5).
8. **Short-horizon evaluation.** 100 ps is a screen; 2.5 ns is required for diffusion and slow
   modes. M0 must pass 2.5 ns, not just 100 ps.
9. **Statistics.** One seed and one trajectory. Need multiple seeds and block error bars.
10. **Transfer.** Water only. The NMI claim needs either a second system (quartz) or a clear
    scope statement.

## 10. Lock criteria

Lock the title/claim only when all of the following hold:

- M0 passes 100 ps (partition near equipartition, RDF/VDOS stable) **and** 2.5 ns (diffusion
  within reference error bars).
- The improvement replicates on an **independent NVT trajectory/seed** and a fresh initial
  condition.
- A controlled ablation shows the gain comes from **ensemble matching**, not just more training
  or more data.
- The pointwise-vs-ensemble decoupling replicates in a **controlled same-pipeline sweep**.
- The novelty statement is sharper than "train on NVT data."
- The limitations (double thermostat, data cost, water-only scope) are explicitly stated.

Until then, the positioning stays **open**, with "ensemble fidelity" as the leading hypothesis.
