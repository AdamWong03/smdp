# Current Method Status

**Date:** 2026-11-20
**Purpose:** method-development baseline, not a paper draft

---

## 1. One-paragraph status

The project has a working finite-time equivariant propagator (DSRT R3-Opt) that is strong on
α-quartz but fails on flexible water: endpoint error is small, yet autonomous NVT produces the
wrong kinetic-energy partition, over-sharpened structure and collapsed diffusion. A matched
Current-Only control shows that removing future conditioning does not fix the shared
internal-motion bias. Increasing the graph radius to 6 Å improves multistep error and rotation
but does not recover translation. The phase-space operator v0 is a new method candidate; on
Quartz it is 3–5% worse on endpoint, mixed on physics and 1.9–2.6× slower; its Water run is
still in progress (epoch 329/400 at 2026-11-20 15:00). No current method meets the NMI method
paper bar.

---

## 2. Evidence table

| Experiment | Endpoint | Autonomous physics | Status |
|---|---|---|---|
| Quartz R3-Opt (30 fs) | 0.00180043 Å; 0.00012208 Å/fs | 15 ns NVT: O/Si VDOS 0.946/0.950; 100 ps NVE drift +1.30/+0.69 eV; RDF residual | frozen baseline |
| Quartz R1 | 0.00165393 Å; 0.000112936 Å/fs (lower) | 100 ps NVE 114.59/115.72 K, −8.02/−8.49 eV; Si–O RDF RMSE 1.87/1.93 | endpoint/physics inversion |
| Water R3-Opt (5 fs) | 0.00065413 Å; 0.00023764 Å/fs | 100 ps NVE +6.30 eV; internal KE 8.327 vs 2.253 eV; NVT translation 0.152 vs 2.459 eV, rotation 0.212 vs 2.502 eV, internal 7.330 vs 2.450 eV; O–O RDF 11.461 vs 3.165; VDOS 0.215/0.233; MSD plateau | **failure to fix** |
| Water 6 Å R3-Opt | 40 fs pos RMSE 0.00585153 vs 0.00847737 Å (4.5 Å) | rotation improves; translation 1.36561 vs 2.21525 eV reference; not recovered | radius is not sufficient |
| Water Current-Only | 0.000401363 vs 0.000409729 Å (Full) | internal KE 4.50728 vs 4.41816 eV; translation 2.22007 vs 1.36561 eV; rotation 2.23382 vs 1.53773 eV; total T 362.967 vs 296.553 K (ref 274.994 K); O–O RDF 0.03092 vs 0.20798 | future conditioning is not the sole cause |
| Phase-space v0 Quartz | 0.001879 vs 0.001800 Å; 8-step 0.02004 vs 0.01899 Å | two 100 ps NVE stable; temperature/energy drift mixed; Si–O RDF better in both, O–O worse in both; VDOS mixed | current method candidate; not better |
| Phase-space v0 Water | test pos RMSE 0.0004908 Å; vel 0.0001687 Å/fs; **worse** than 6 Å Full (0.0004097/0.0001497) | 20 ps NVE looked better (internal 3.126, H/O VDOS 0.861/0.942), but **2.5 ns NVT fails**: translation 0.111 vs 2.459; rotation 0.808 vs 2.502; internal 6.338 vs 2.450; VDOS 0.356/0.236; diffusion 2.81e-13 vs 1.88e-9 | completed; **negative** |
| H-C 20k prefix (same seed as 250k) | 0.001163 Å / 0.000401 Å/fs | **NVT unstable** at ~50 ps: min 0.852 Å, preT 3338 K | early-stopping line not supported |
| Current-Only 250k Water | 0.000401 Å / 0.000151 Å/fs | 100 ps NVT: translation 0.646 vs 2.373; rotation 0.535 vs 2.430; internal 6.384 vs 2.609; H/O VDOS 0.348/0.600 | stable but **fails NVT**; future conditioning not the cause |
| Native TrajCast ep145 Water | 0.000144 Å / 0.000063 Å/fs (native val) | 100 ps NVT: translation 3.256 vs 2.373; rotation 3.105 vs 2.430; internal 1.041 vs 2.609; H/O VDOS 0.683/0.905 | stable, **opposite bias**; incomplete native protocol (149/1500), diagnostic only |
| Mode-aware fine-tune (frozen 250k) | 0.000412 Å / 0.000150 Å/fs | no NVT; correction RMS 9.6e-6 | **null**; endpoint loss cannot train the correction |
| Native TrajCast | incomplete | stopped at epoch 149/145 | not a completed baseline |

**Do not project the v0 Water result.** The validation RMSE at epoch 329 is not a test result
and says nothing about NVT physics.

---

## 3. What the evidence says about the method

1. **Endpoint accuracy is not aligned with physical fidelity.** R1 is the cleanest example:
   lower endpoint error, much worse autonomous physics. The Water model is the second.
2. **Future conditioning is not the sole cause.** Removing it changes some observables but
   leaves the shared internal-motion bias (~2× reference in both Full and Current-Only).
3. **Graph radius is not the sole cause.** 6 Å improves multistep error and rotation, but
   translation remains suppressed.
4. **The v0 operator is not a solution.** Quartz is worse on endpoint and mixed on physics.
   Water v0 is worse on endpoint/multistep than both 6 Å baselines, and its 20 ps NVE
   partition gain did **not** survive 2.5 ns NVT: translation 0.111 vs 2.459 eV, rotation
   0.808 vs 2.502, internal 6.338 vs 2.450, VDOS 0.356/0.236, diffusion 2.81e-13 vs
   1.88e-9 m²/s. The 20 ps NVE proxy was misleading.
5. **Model selection is endpoint-only.** The formal configs select on normalized final endpoint
   MSE. This is a likely contributor: it selects models that are endpoint-optimal but
   physics-poor.
6. **The training objective is one-step.** The loss is endpoint MSE + 0.02 proposal loss. There
   is no multi-horizon or rollout-consistency term. Error accumulation is not directly
   penalized.
7. **The reference verification is incomplete.** The 8-frame energy check fails at frame
   28,571; energy-drift claims are blocked until fixed.

---

## 4. What is missing for an NMI method paper

- A method change that fixed the Water partition/structure/dynamics, not just endpoint.
- A clear mechanism: which information path, normalization, or training signal is responsible.
- A seed-replicated advantage over matched controls (Current-Only, 6 Å, v0 variants).
- **v0 Water NVT 2.5 ns and diffusion** — the 20 ps NVE trade-off is promising but not an
  equilibrium result.
- Quartz preservation and a boundary test (temperature/size).
- A measured cost–accuracy–physical-fidelity frontier.
- A matched external baseline if claiming an advantage over TrajCast/ChronoCast.
- A reference-verification fix and an implementation-bug audit.

---

## 5. What not to do

- Do not write a paper, abstract or cover letter yet.
- Do not claim the v0 operator is better.
- Do not claim the Water failure is universal.
- Do not launch a large hyperparameter sweep before the low-cost selection/objective
  experiments.
- Do not change the architecture or add a physical loss without research-lead approval.

---

## 6. Update 2026-11-21 — mechanism isolated

**The NVT failure is a closed-loop mode-partition drift.**

| Model | trans KE | rot KE | int KE | α_i(5 fs) | α_i(500 fs) |
|---|---:|---:|---:|---:|---:|
| Reference | 2.373 | 2.430 | 2.609 | −0.567 | 0.104 |
| 250k Full | 0.538 | 0.786 | 6.044 | −0.902 | 0.722 |
| Current-Only | 0.646 | 0.535 | 6.384 | −0.917 | 0.590 |
| Native ep145 | 3.256 | 3.105 | 1.041 | +0.109 | 0.003 |

(eV, 64 molecules; equipartition ≈2.48 eV per mode.)

- Total kinetic energy is right in every model (CSVR enforces it); the **partition** is wrong.
- Our models: collective cold, internal hot and underdamped (internal ACF 0.72 at 500 fs).
- Native ep145: opposite bias; internal cold and overdamped.

**The one-step map on held-out data is correct:**

| Mode | α_ref | α_pred | Var ratio | residual / variance |
|---|---:|---:|---:|---:|
| translation | 0.98133 | 0.98154 | 1.00044 | 2.1e-4 |
| rotation | 0.83709 | 0.83680 | 0.99917 | 2.1e-4 |
| internal | −0.39310 | −0.39314 | 0.99973 | 1.3e-4 |

No one-step gain bias and no missing one-step noise. The failure is a positive-feedback
distribution shift: a tiny state-dependent error moves the model off the training
distribution, where the error grows and pushes the partition further.

On-policy pilot (160 autonomous states, fresh SPC/Fw one-step labels): the one-step gain still
matches to ~1e-3, but there is a small **signed** internal-KE surplus (+0.0062 eV per step,
+0.15% of internal KE; model 4.483 vs true 4.476 eV). Over ~10^4 steps it accumulates; CSVR
removes total energy but not the partition.

**Consequence:** endpoint, 2-step, 20 ps NVE, and spectral/gain regularizers cannot fix this.
The root-cause fix is on-policy correction with a **signed mode-resolved kinetic-energy loss**
(not magnitude MSE; E1 heated the system). A mode-resolved thermostat is a cheap control.
See `LONG_TIME_TRAINING_LITERATURE.md` and GAP_LOG G14.

**External control caveat:** published TrajCast reports VDOS overlap ~0.99 and diffusion
2.18 vs 1.84e-9 m²/s for the same water system at 2.5 ns. Our native control is at epoch
145/1500 (~10% of the schedule) and cannot be used to claim the architecture fails. A
converged native baseline is still required.

---

## 7. Published IBM TrajCast control — PASS (2026-11-21)

The official water checkpoint (`ibm-research/trajcast.models-arxiv2025`, 2.26M parameters)
was run under the **same 100 ps protocol** as our models (50 ps thermalization + 100 ps
production, CSVR 300 K, τ=50 fs, `val_04` frame 0).

| Model | trans KE | rot KE | int KE | total | T (K) |
|---|---:|---:|---:|---:|---:|
| Reference | 2.373 | 2.430 | 2.609 | 7.411 | 300 |
| **Published TrajCast** | **2.505** | **2.463** | **2.444** | **7.411** | **300.2** |
| Our native ep145 | 3.256 | 3.105 | 1.041 | 7.402 | 300 |
| Our 250k Full | 0.538 | 0.786 | 6.044 | 7.368 | 299 |

The published model is at equipartition from the first 10 ps, stays flat for 150 ps, and its
translation/rotation ACFs match the reference (0.662 vs 0.658; −0.539 vs −0.540). O–O RDF
RMSE is 0.043 vs 0.180 for our native.

**This is decisive:** offline training + sufficient one-step accuracy gives faithful long-time
NVT. On-policy data is **not** required. The failure is an accuracy/training problem.

**Twist:** pointwise accuracy is not the objective. Fresh SPC/Fw one-step labels at each
model's own NVT states:

| Model | position RMSE | velocity RMSE | signed KE error | NVT ensemble |
|---|---:|---:|---:|---|
| Published TrajCast | 6.9e-3 Å | 2.5e-3 Å/fs | +0.083 eV/step | **PASS** |
| 250k Full | 4.9e-4 Å | 1.7e-4 Å/fs | +0.0062 eV/step | **FAIL** (int 6.044) |

The published model has ~15× larger pointwise error yet the correct ensemble; the 250k has
smaller pointwise error yet the wrong ensemble. The published model behaves like a generative
propagator (different trajectory, same distribution). The 250k learned the conditional mean
but the wrong distribution.

**Consequence:** per-state label correction (held-out or on-policy) optimizes the wrong object.
The next method test is **ensemble-matched training**: fine-tune 250k Full on 5 fs pairs from
the reference NVT trajectory (offline, no new labels), then 100 ps NVT. On-policy label
correction is a diagnostic, not the main route.

Caveat: the native 0.50% point may be confounded by a train/inference data-representation
difference; do not use it as a quantitative threshold point.

---

## 8. M0 ensemble-matched fine-tune — negative (2026-11-21)

Fine-tuned the 250k Full model on 5 fs pairs from the reference NVT trajectory (5k steps,
batch 8, lr 3e-5). Validation endpoint loss fell from 6.4e-4 to 2.06e-4, but the autonomous
100 ps NVT rollout did **not** reach equipartition:

| Model | trans KE | rot KE | int KE | total | T (K) |
|---|---:|---:|---:|---:|---:|
| Reference | 2.373 | 2.430 | 2.609 | 7.411 | 300 |
| Published TrajCast | 2.505 | 2.463 | 2.444 | 7.411 | 300.2 |
| Base 250k | 0.538 | 0.786 | 6.044 | 7.368 | 299 |
| **M0 5k** | **3.373** | **3.27** | **1.00** | **7.69** | **311.6** |

The bias **flipped sign and overshot** (collective hot, internal cold), similar to the native
ep145 bias, and the total temperature rose to 311.6 K. Translation ACF at 25 fs is 0.703
(ref 0.658); rotation −0.389 (ref −0.540); internal 5 fs +0.183 (ref −0.567). Stable, no
safety stop, but dynamically wrong.

Ensemble-level structure/spectra:

| Model | O–O RDF RMSE | O–H RDF RMSE | H–H RDF RMSE | H VDOS | O VDOS |
|---|---:|---:|---:|---:|---:|
| Base 250k | 0.646 | 1.312 | 0.249 | 0.406 | 0.487 |
| **M0 5k** | **0.191** | 1.790 | **0.082** | **0.668** | **0.888** |
| Published | 0.043 | 0.118 | 0.096 | 0.624 | 0.937 |

M0 substantially improves the O–O RDF, H–H RDF, and both VDOS overlaps, but the O–H RDF gets
worse and the partition overshoots. This is a **partial** result: endpoint NVT pairs move the
model toward the right structure/spectra, but not to the right stationary partition.

**Conclusion:** one-step endpoint MSE on NVT pairs is **not sufficient** to control the
stationary measure. The model fits the pairs but the closed-loop rollout drifts to a different
partition (and injects energy, T=311.6 K). This confirms the "conditional mean ≠ distribution"
flaw. The next route is a **distribution-level loss** (penalize the rollout partition/RDF/VDOS)
or a **stochastic/generative head**, not more endpoint fine-tuning.

**Independent-start check (same M0 checkpoint, independent NVT frame 0 at 301.8 K):**

- Valid production 50–130 ps: trans **3.41**, rot **3.21**, int **1.01** eV; T **313.8 K**.
  The bias is the same as the val_04 start — it is not a same-trajectory artifact.
- The run **failed at 130 ps**: min distance 0.965 Å, pre-thermostat T 3798.7 K.
  The model injects energy and is **not stable** on an independent start.

This is a stronger negative: M0's bias is robust, but its stability is not. Endpoint NVT
training (M0) is not viable as the method.

---

## 9. NVE energy-stability diagnostic — energy stability is NOT the differentiator (2026-11-21)

25 ps NVE, no thermostat, same independent frame (301.8 K):

| Model | 25 ps end T | min / max | NVT partition |
|---|---:|---:|---|
| **Published** | **475.6 K** | 272.6 / **545.7** | **PASS** (2.505/2.463/2.444) |
| M0 5k | 387.3 K | 277 / 484 | overshoot (3.41/3.21/1.01) |
| Base 250k | 270.5 K | 222 / 361 | wrong (0.54/0.79/6.04) |

- The **published model is the least NVE-stable but the only NVT-correct**. The base is the most
  NVE-stable but NVT-wrong.
- **The energy-drift penalty is falsified:** it would penalize the only working model. Do not add
  an NVE energy term.
- The M0 NVT instability (independent start failed at 130 ps, preT 3798 K) is a real **NVT**
  failure, not an NVE one.
- The relevant property is the **stationary measure of the composed propagator + CSVR map**.

**Method consequence:** differentiable short CSVR rollout + ensemble moment loss (partition,
temperature, optionally RDF/VDOS) from the base 250k. No NVE energy penalty.
See `NVE_STABILITY_20261121.md`.

---

## 10. Composed CSVR rollout loss — negative (2026-11-21)

A 5-step differentiable rollout with a deterministic 300 K thermostat + ensemble moment loss
(partition + T) was trained from the base 250k on the model's own NVT states.

| Step | val moment | val T | endpoint |
|---:|---:|---:|---:|
| 1 | 2.34 | 322 K | 7.8e-5 |
| 50 | 0.162 | 307.6 K | — |
| 100 | **0.028** | 307.4 K | **0.041** |

The partition moment fell ~80×, but the one-step endpoint degraded ~500×. A 100 ps NVT screen
of the step-100 checkpoint:

- **failed at step 855 (4.3 ps)**: min distance **0.647 Å**, pre-thermostat T **1473 K**.

**Interpretation:** the 5-step moment loss does not constrain stability. The endpoint collapse
was not merely "pointwise accuracy"; it signaled the map had left the stable basin. This is a
negative result for the short-rollout composed loss. Next: a trust-region/stability-constrained
version (longer rollout, contact/energy penalty, bounded correction) or an architecture change.

---

## 11. FD readout v1 — positive 100 ps (2026-11-21)

Architecture: frozen 250k backbone + mode-resolved FD readout.

```
v_fd  = a_mode v + sqrt(1-a_mode²) noise_mode   (target mode KE)
v_new = 0.9 v_pred + 0.1 v_fd
v_new = per-mode variance rescale to 0.5*dof*kT
q'    = q + dq_pred
```

100 ps screen (val_04 start, no external CSVR):

| Item | Value |
|---|---|
| Stability | 20,000/20,000 steps |
| T | **300.0 K** |
| Partition | **2.482 / 2.533 / 2.392 eV** (vs 2.5 ns ref 2.459/2.502/2.450; the first 100 ps block is 2.373/2.430/2.609) |
| O–O RDF | **0.062** (base 0.646; published 0.043) |
| O–H RDF | 0.301 (base 1.312; published 0.118) |
| H–H RDF | **0.110** (base 0.249; published 0.096) |
| H VDOS | **0.807** (base 0.406; published 0.624) |
| O VDOS | **0.851** (base 0.487; published 0.937) |

**Verdict:** first viable method lead. The FD readout fixes the partition and stability by
construction and substantially improves the structure/VDOS. The remaining gap is the **O–H
RDF** (intramolecular bond distribution), controlled by the position head, not by the velocity
readout. Independent start and 2.5 ns are pending.

**Independent-start 100 ps:** stable; T=300.0; partition 2.482/2.532/2.393; O–O RDF 0.057,
O–H 0.304, H–H 0.104; VDOS H 0.808, O 0.843. This reproduces the val_04 result almost exactly:
the FDR stationary measure and structure are **independent of the initial condition**. The
O–H gap is reproducible, not a start artifact.

**2.5 ns + diffusion:** stable; O–O RDF **0.0373** (better than published 0.043), O–H **0.4095**,
H–H 0.1125; VDOS 0.797/0.851; **diffusion 3.59e-9 ± 0.28e-9 vs reference 1.88e-9** (1.9× too
fast). The translation ACF has a long positive tail (~0.08 at 5–10 ps) and the internal spectrum
is still off. A single-gain AR(1) cannot match the multi-lag ACF.

**Verdict:** v1 is a **partial method**: partition/T/O–O/H–H/VDOS strong; O–H and diffusion off.
The next architecture is a **mode-resolved Generalized Langevin Readout (GLR)**: AR(p) per mode
with Yule–Walker coefficients from the reference ACF, FD noise, and the variance rescale. This
targets both the diffusion tail and the O–H/internal spectrum.

---

## 12. GLR v2 — positive 100 ps (2026-11-22)

AR(10) per mode, Yule–Walker coefficients from the reference ACF; prediction-error ratios
translation 0.0042, rotation 0.0576, internal 0.1146; blend 0.1 + variance rescale.

| Metric | GLR | FDR v1 | Published |
|---|---:|---:|---:|
| T | 300.0 K | 300.0 K | 300.2 K |
| partition | 2.482 / 2.526 / 2.399 | 2.482/2.533/2.392 | 2.505/2.463/2.444 |
| O–O RDF | **0.0322** | 0.062 | 0.043 |
| O–H RDF | **0.1043** | 0.301 | 0.118 |
| H–H RDF | **0.0256** | 0.110 | 0.096 |
| H VDOS | **0.8484** | 0.807 | 0.624 |
| O VDOS | **0.9637** | 0.851 | 0.937 |

Translation ACF α(5/25/100/500 fs) = 0.981/0.640/0.014/−0.030 (ref 0.982/0.658/0.039/−0.019).
**All structural/spectral metrics beat the published model in our analysis.** 2.5 ns diffusion
and independent-start 100 ps are running.

**Independent-start 100 ps:** O–O 0.0358, O–H **0.0981**, H–H 0.0268; VDOS 0.851/0.963;
partition/ACF identical to val_04. Initial-condition independence is confirmed. Only the 2.5 ns
diffusion remains.

---

## 13. Displacement correction — negative (2026-11-22)

Adding \(\Delta t(v^{\rm GLR}-v^{\rm pred})\) to the backbone displacement
(`--fd-correct-displacement`) crashed at **24.8 ps** (4,969 steps; min distance 0.647 Å;
pre-thermostat T 299.7 K). The failure is structural, not thermal. The decoupled scheme
(positions from the backbone, velocities from the readout) is a stability requirement.

## 14. 100 ps MSD diagnostic (2026-11-22)

Molecular-COM MSD, GLR AR(10) val_04 vs the first 100 ps of the reference:

| Fit window | GLR D (m²/s) | Reference D (m²/s) |
|---|---:|---:|
| 10–50 ps | 1.40e-9 | 1.66e-9 |
| 10–80 ps | 1.45e-9 | 1.81e-9 |
| 10–95 ps | 1.54e-9 | 1.76e-9 |
| 40–95 ps | 1.69e-9 | 1.75e-9 |
| 60–95 ps | **1.90e-9** | 1.52e-9 |

The GLR MSD is superdiffusive at 50–100 ps (local slope 0.84 → 1.14), so the short-window fit
underestimates D. The 2.5 ns block analysis is the production number. The FDR v1 fast tail
(3.59e-9) is absent.

## 15. Paper artifacts (2026-11-22)

Phase 0–3 artifacts are in `paper/`: `PAPER_CONFIG.md`, `NOVELTY_MATRIX.md`, `OUTLINE.md`,
`EVIDENCE_MAP.md`, `ARGUMENT_BLUEPRINT.md`, `METHODS_DRAFT.md`. Provisional prose drafts are in
`paper/draft/`. Figures 1–4 are generated by `paper/figures/fig{1,2,3,4}_*.py`; the LaTeX skeleton
is `paper/latex/main.tex` and compiles. The 2.5 ns diffusion gate is the only remaining blocker
for the paper positioning.

## 16. GLR 2.5 ns — structure passes, diffusion is 25% low (2026-11-22)

| Quantity | GLR 2.5 ns | Reference (2.5 ns) |
|---|---:|---:|
| Stable | **yes** (500k steps) | — |
| T | 299.994 K | 300.17 K |
| Partition (trans/rot/int) | **2.482 / 2.526 / 2.399 eV** | 2.463 / 2.501 / 2.447 |
| O–O / O–H / H–H RDF | **0.0307 / 0.0376 / 0.0282** | 0 |
| H / O VDOS | 0.847 / 0.953 | 1 |
| D (MSD) | **1.406 ± 0.155e-9** | 1.883 ± 0.306e-9 |
| D (Green–Kubo) | 1.420e-9 | 1.836e-9 |

The 2.5 ns run is the best structural result so far: O–H RDF 0.038 (much better than the 100 ps
0.104 and the published 0.118), O–O 0.031, H–H 0.028, O VDOS 0.953. The partition is within
0.05 eV of the 2.5 ns reference. The **only failing metric is diffusion**: D is 25% below the
reference, while the published checkpoint is 16% above. The Green–Kubo integral agrees with the
MSD, so the position update is consistent with the readout velocity; the bottleneck is the
translation ACF integral (GLR 100 fs 0.011 vs reference 0.039). The AR(20) translation fit matches
the 100 fs lag. A mixed readout (translation AR(20), rotation/internal AR(10)) is under test as the
targeted fix.

**Independent 2.5 ns (2026-11-22):** stable, T = 300.0005 K; partition
2.482/2.527/2.398 eV; RDF 0.028/0.044/0.027; H/O VDOS 0.847/0.953;
**D = 1.393 ± 0.079e-9** (26% low). The two 2.5 ns starts agree on the stationary measure,
including the diffusion deficit. This confirms initial-condition independence and shows the
deficit is a property of the decoupled architecture, not of val_04.

**Failed diffusion fixes (2026-11-22).** The mixed translation AR(20) readout raises the
Green–Kubo estimate to 1.61e-9 but the MSD only to 1.46e-9 at 100 ps. Raising the translation
blend to 0.3/0.5 makes D worse (0.97/1.05e-9) because the AR(10) translation ACF is negative at
100 fs. A translation-only displacement correction (\(\alpha=0.5/1.0\)) is stable but gives
1.36/1.43e-9. Scaling each molecule's COM displacement (\(\beta=1.05/1.15\)) can raise the MSD to
2.33/3.04e-9 but breaks the MSD–Green–Kubo consistency and degrades the RDF at \(\beta=1.15\).
None of these fixes closes the gap without a structural or consistency cost.

**Verdict.** The decoupled position update is the limiting factor. The 2.5 ns GLR passes
partition, RDF, VDOS, and stability, but diffusion is 25% low. The paper should report this as a
quantified limitation, not claim exact diffusion recovery. Do not lock a "recovers diffusion"
claim. The positioning can be locked around **mode-resolved thermodynamic consistency and
structure**, with diffusion as the remaining transport limitation.

## 17. Component ablations — rescale necessary, FD noise not (2026-11-22)

| Ablation | Result | Interpretation |
|---|---|---|
| **Variance rescale off** | stable but **T=172 K**; partition 0.763/0.649/2.839 eV; RDF 0.398/0.474/0.279; VDOS 0.432/0.608 | rescale is the partition guarantee; removing it collapses the temperature and structure |
| **FD noise off** | stable; partition 2.482/2.526/2.399; RDF 0.034/0.098/0.026; VDOS 0.849/0.962; D 1.55e-9 | **not necessary at λ=0.1**: the backbone + rescale provide enough fluctuation; the memory kernel and rescale are the load-bearing components |
| Internal blend 1.0 (FDR v1) | O–H 0.957, VDOS 0.618/0.613 | single-gain readout cannot fix the internal spectrum; GLR int 1.0 pending |
| Translation blend 0.3/0.5 (p10) | D 0.97/1.05e-9 | wrong AR(10) translation ACF at 100 fs; larger admixture makes diffusion worse |
| **Seeds 1/2/3** (100 ps) | partition identical 2.482/2.526/2.399; O–H 0.104/0.111/0.110; O VDOS 0.964/0.966/0.959 | the 100 ps structure is seed-robust; the 2.5 ns O–H 0.038 is a longer-sampling effect |

The FD-noise negative result means the paper must not claim "FD noise is necessary." The method
still includes FD-consistent noise for theoretical consistency, but the ablation shows the
structure and diffusion are robust to its removal at the small blend used here.

## 18. Backbone-agnostic test — v0 rescued; quartz 30 fs is a boundary (2026-11-22)

**v0 phase-space backbone + GLR (100 ps, two starts).** The v0 operator (3.04M parameters, a
different architecture) fails NVT without the readout: partition 0.111/0.808/6.338, RDF
0.944/1.409/0.229, VDOS 0.356/0.236. With the same GLR coefficients and λ=0.1:

| Start | T (K) | partition (eV) | RDF O–O/O–H/H–H | VDOS H/O |
|---|---:|---|---:|---:|
| val_04 | 299.99 | 2.482 / 2.529 / 2.396 | 0.042 / 0.109 / 0.019 | 0.855 / 0.969 |
| independent | 300.00 | 2.482 / 2.527 / 2.398 | 0.050 / 0.131 / 0.023 | 0.857 / 0.977 |

**The same readout rescues a completely different backbone.** This is the strongest empirical
support for "backbone-agnostic" and is now Table 3 / Results 2.4.

**Quartz species basis (mixed result).** The quartz relational backbone (30 fs horizon) is stable
at 100 ps with T=311.6 K and species KE near equipartition; RDF Si–O 0.175, VDOS O/Si 0.837/0.819.
A full species-resolved GLR (Si/O AR(10), λ=0.1) enforces T=301.9 and exact species KE, but the
RDF/VDOS collapse (Si–O 0.699, VDOS 0.225); at λ=0.02, Si–O 0.301 and VDOS 0.48, still worse than
baseline. The 30 fs ACF is nearly white (prediction error 0.91), so the memory kernel is
uninformative and the readout injects nearly-white noise into the backbone.

**Zero-memory limit (positive cross-material result).** Setting the memory/noise contribution to
zero and keeping only the species variance rescale gives T=301.9 K, exact Si/O KE
(0.03878 eV per atom), VDOS O/Si 0.838/0.835 (baseline 0.837/0.819), and Si–O RDF 0.149 (baseline
0.175). The **variance-rescale component generalizes across a covalent solid**; the memory kernel
needs a horizon shorter than the relevant vibrational period. A 5 fs quartz backbone or a
phonon-mode basis is future work. This is now Results 2.5 + Table 4 and a Discussion design
principle/limitation.

## 19. 5 fs quartz extension — in progress (2026-11-22)

- Built `quartz/transitions/5fs/{train,val,test}.h5` (5000/1250/1000) and the 1.4 GB
  `quartz5_*;P:64.pt` cache under `experiments/quartz5fs_20261122`.
- Training `experiments/quartz5_relational_20261122` (RelationalTransition, 4.5 Å, 5 fs,
  2.36M params) started at 17:12; epoch 1/400, 58 s/epoch, ETA ~6.5 h on L40S.
- Fit `experiments/quartz5_glr_fit_20261122/species_coefficients.json`: 5 fs species AR(10)
  prediction errors **O 0.0137, Si 0.0150** (vs 0.91 at 30 fs). The memory kernel is now
  informative.
- Next: 100 ps baseline and species GLR with the 5 fs checkpoint; analyze species KE, RDF,
  VDOS. If positive, this becomes the full cross-material system for the NMI paper.

## 20. Stochastic rescale — positive; kernel-order D test — negative (2026-11-22)

**Stochastic per-mode rescale** (`--fd-stochastic-rescale`; Gamma/CSVR target) on the 250k water
backbone, 100 ps, val_04:

| Quantity | Stochastic GLR | Reference |
|---|---:|---:|
| T | 300.16 ± 17.84 K | 300.17 ± 17.9 K |
| total KE std | **0.4405 eV** | **0.4417 eV** |
| translation / rotation / internal KE std | 0.2527 / 0.2569 / 0.2549 eV | 0.2391 / 0.2342 / 0.2499 eV |
| partition mean | 2.483 / 2.531 / 2.397 eV | 2.373 / 2.430 / 2.609 eV |
| RDF O–O/O–H/H–H | 0.0355 / 0.1141 / 0.0218 | — |
| H/O VDOS | 0.854 / 0.957 | — |
| D (10–80 ps) | 1.342e-9 | 1.883e-9 |

The **canonical KE distribution** is restored (total KE std matches to 0.3%); mean partition,
RDF, VDOS remain good. D is still 26–29% low. The 2.5 ns stochastic run later showed a GK/MSD
inconsistency (§23), so the deterministic target remains the water production choice and the
stochastic target is the thermodynamic ablation.

**Kernel-order feasibility (B3).** Reference translation ACF: −0.078 at 200 fs, −0.019 at
500 fs. Positive Prony cannot fit the negative dip. Raw AR orders give D_GK 1.42 (p10), 1.61
(p20), 4.20e-9 (p40): p10 tail too negative, p40 tail too positive. A proper GK-constrained
Markovian embedding with oscillatory modes and a zero-frequency D constraint is a separate method
project. Current paper: deterministic water production (GK/MSD consistent), stochastic ablation;
D 26% as a quantified limitation with the kernel-order mechanism.

## 21. 5 fs quartz — zero-memory positive, full species GLR negative (2026-11-23)

The 5 fs quartz RelationalTransition (162 atoms, 4.5 Å, trained 250k steps; test pos MAE
7.7e-5 Å, vel MAE 2.8e-5 Å/fs) was evaluated on 100 ps NVT.

| 5 fs quartz | T (K) | Si/O KE per atom (eV) | RDF Si–O/O–O/Si–Si | VDOS O/Si |
|---|---:|---|---:|---:|
| baseline | 235.3 | 0.0298 / 0.0305 | 0.271 / 0.096 / 0.133 | 0.918 / 0.944 |
| **zero-memory species rescale** | **301.86** | **0.03878 / 0.03878** | **0.078 / 0.022 / 0.055** | **0.866 / 0.914** |
| full species GLR (λ=0.1) | 301.86 | 0.03878 / 0.03878 | 0.529 / 0.263 / 0.441 | 0.074 / 0.067 |

**Positive:** the variance-rescale component generalizes to a covalent solid; it fixes the 5 fs
baseline cooling (235 → 301.9 K), enforces Si/O equipartition, improves the Si–O RDF from 0.271
to 0.078, and preserves the phonon VDOS (0.866/0.914).

**Negative boundary:** the full species GLR (memory + noise) collapses the VDOS to 0.07. Species
velocity is not a phonon eigenbasis; the memory kernel needs a physically meaningful collective
basis (molecular modes for water; phonon modes for solids), not just a finer horizon.

This gives the paper a **second positive system in the zero-memory limit** and a clean component-
selection principle: water uses memory + rescale; quartz uses rescale only. The full-memory
species negative is reported as a boundary, not hidden.

## 22. Phonon-mode GLR — panel GO-WITH-MODIFICATIONS, basis audit passed (2026-11-23)

Four independent reviewer roles (ML-MD/NMI, lattice dynamics, quartz/ML potential, NMI editor)
all returned **GO-WITH-MODIFICATIONS**: run Project B as a **timeboxed kill-test**, not an
open-ended method project. The current L1 paper remains the submission baseline; B adds a figure
only if a pre-registered gate passes. The NMI editor and the ML-MD reviewer both require a
**Cartesian covariance/whitening control**; the lattice and quartz reviewers require the three
acoustic modes to be removed and a DHO kernel below ~5 THz.

**Basis audit (artifacts `/home/wyd/quartz_phonon/`):**

| Check | Result |
|---|---|
| Hessian | finite differences of LAMMPS forces, ±0.01 Å, 972 points, same Buckingham+Coulomb/DSF potential |
| Structure | 0 K fixed-cell relaxed (the 300 K average has 27 imaginary modes; frame 0 has 14) |
| ASR correction | \(7.6\times10^{-15}\) relative |
| Residual force | 0.0067 kcal mol⁻¹ Å⁻¹ |
| Modes | 486; 3 acoustic zeros; 4 imaginary (3 non-acoustic, \(f=-0.298/-0.134/-0.082\) THz) |
| Reference mode KE | 99.4% of all modes within 10% of \(k_BT/2\); 100% of stable optical modes |
| Off-diagonal covariance | 0.55% |
| AR(10) prediction error | max 0.0068; pole modulus max 0.998 |
| AR(10) ACF error at 500 fs | 0.14 (1–3 THz), 0.20 (3–5 THz) |
| DHO ACF error at 500 fs | 0.042 (1–3 THz), 0.035 (3–5 THz) |

The basis is clean; the reference is canonically populated. AR(10) captures frequencies but not
low-frequency linewidths; DHO is 3–6× better at 500 fs in the 1–5 THz bands. The 0–1 THz band is
dominated by the three soft imaginary modes and is not a valid DHO target.

**Implementation:** `code/h1h2/phonon_glr.py` (fixed basis, AR readout, per-mode stochastic Gamma
rescale, COM projection, backbone displacement); CLI flags `--phonon-glr-basis`,
`--phonon-glr-coefficients`, `--phonon-memory-off`. A 1 ps smoke test passed: mode KE mean
0.012949 eV vs target 0.012926 (ratio 1.0018), mode-KE std 0.01783 vs 0.01828 expected, COM
velocity < 5e-9 Å/fs. The 100 ps zero-memory and full AR(10) runs are on GPU 1.

**Gate:** stable 100 ps; \(T=300\pm3\) K; ≥90% of 483 non-acoustic modes within ±10% of
\(k_BT/2\); VDOS O/Si ≥ 0.866/0.914; Si–O RDF ≤ 0.08; full GLR beats the Cartesian covariance
control on band-resolved VDOS and low-frequency mode ACF; second seed within noise before any
production claim. NO-GO: VDOS < 0.80, RDF > 0.15, no gain over zero-memory/control, or any
unstable seed. No rescue sweeps.

**Publication plan:** L1 paper unchanged until the gate; R8 package (code/data DOI, source data,
SI, AI disclosure) runs in parallel. Pass → add one figure + SI; fail → submit water-scoped L1,
report B as boundary/follow-up.

**Per-mode rescale is the wrong granularity.** Both per-mode arms are stable and canonically
correct, but the VDOS collapses:

| Arm | T (K) | Si/O KE (eV) | RDF Si–O/O–O/Si–Si | VDOS O/Si |
|---|---:|---|---:|---:|
| phonon + per-mode rescale, no memory | 299.64 | 0.03856 / 0.03836 | 0.027 / 0.016 / 0.027 | 0.217 / 0.269 |
| phonon + per-mode rescale, AR(10) | 299.79 | 0.03857 / 0.03840 | 0.054 / 0.048 / 0.081 | 0.209 / 0.253 |

Mode KE is exact (100% of modes within 10%; total KE std 0.398/0.401 eV), but the mode ACF error
at 100 fs is 0.21–0.59 vs 0.03–0.04 for the baseline. A per-mode Gamma draw every 5 fs imposes
the right one-time marginal but randomizes each single-dof mode's envelope, destroying phonon
coherence. The rescale granularity must be a **group with many dofs**, not a single mode.

**The positive quartz result is the phonon-band stochastic rescale (no memory).** 100 ps:

| Quantity | Phonon-band stochastic rescale (two seeds) | Reference |
|---|---:|---:|
| T (K) | **299.9** (299.8 / 299.9) | 300.17 ± 17.9 |
| total KE std (eV) | **0.398** (0.397 / 0.399) | **0.4017** |
| Si / O KE per atom (eV) | 0.03846 / 0.03855 | 0.03878 |
| RDF Si–O / O–O / Si–Si | **0.028 / 0.016 / 0.033** | — |
| VDOS O / Si | **0.950 / 0.955** | 1 |
| mode KE within 10% / 20% | 77.2% / 96.8% | 67.7–75.6% / 94.4–97.3%* |
| mode ACF error at 100 fs | 0.02–0.07 | — |

Seed 2 alone: T=299.91 K, KE std 0.3991 eV, RDF Si–O/O–O/Si–Si 0.0312/0.0176/0.0400, VDOS
O/Si 0.947/0.962, mode KE 76.2%/96.7%. The two seeds agree within the reference block-to-block
scatter.

\*Reference values are for a **100 ps** window, not the 2 ns average. The 2 ns average gives 99.4%
within 10%; a 100 ps window has irreducible mode-KE sampling noise. The model's 78%/96.9% is
statistically consistent with the reference at the same window length.

A coarser **species stochastic** partition is a control: T=301.68 K, exact Si/O equipartition
(0.03872/0.03877), total KE std 0.4012, VDOS 0.898/0.955, RDF 0.0245/0.0169/0.0380. The band
partition is slightly better on VDOS and mode KE; both are robust.

The **deterministic species** rescale fixes the mean but kills the variance: T=301.86 K, VDOS
0.866/0.914, RDF 0.078/0.022/0.055, total KE std **0.0036 eV**, 57.8% of modes within 10%.

**Full phonon memory is negative.** The baseline 5 fs backbone is *uniformly* cold: every phonon
band has 0.75–0.85 of \(k_BT/2\), not a spectral bias. Adding the phonon AR(10) memory creates a
spectral bias that was not there:

| Arm | T (K) | total KE std | VDOS O/Si | RDF Si–O |
|---|---:|---:|---:|---:|
| baseline | 235.3 | 0.4355 | 0.918 / 0.944 | 0.271 |
| phonon memory, no rescale | 251.7 | 0.4648 | 0.594 / 0.686 | 0.137 |
| phonon memory + species stochastic rescale | 301.9 | 0.4024 | 0.734 / 0.838 | 0.197 |

The memory arm pumps the high-frequency modes (20–40 THz ratio 1.14–1.27) and drains the
low-frequency modes (0–5 THz ratio 0.28–0.38). The 5 fs backbone's mode ACF is already correct
(100 fs error 0.03–0.04), so the AR innovations are unnecessary and broaden the sharp phonon
lines. **Full-memory claim: NO-GO.**

**Conclusion for the paper:** quartz is a **rescale-only** system. The new main quartz row is the
**phonon-band stochastic rescale** (not deterministic), which gives canonical total KE
fluctuations, VDOS 0.953/0.949, and RDF 0.0245. The species stochastic partition is a coarser
control; the deterministic rescale is an ablation; the full species GLR and full phonon GLR are
negative controls. The phonon-basis audit explains why: the reference is canonically populated,
and the 5 fs backbone's mode dynamics are already correct. The L1 paper is stronger without a
forced full-memory result.











## 23. Stochastic 2.5 ns — canonical KE variance but GK/MSD inconsistency (2026-11-23)

Two 2.5 ns stochastic-rescale runs completed:

| Quantity | val_04 | independent | Reference |
|---|---:|---:|---:|
| T (K) | 299.97 | 300.08 | 300.17 |
| partition (eV) | 2.483 / 2.525 / 2.398 | 2.482 / 2.528 / 2.398 | 2.463 / 2.501 / 2.447 |
| total KE std (eV) | **0.437** | **0.429** | **0.440** |
| RDF O–O/O–H/H–H | 0.028 / 0.054 / 0.024 | 0.028 / 0.048 / 0.027 | — |
| H/O VDOS | 0.852 / 0.949 | 0.853 / 0.950 | 1 |
| MSD D (8-block, 10–100 ps) | 1.435 ± 0.130 | 1.387 ± 0.125 | 1.883 ± 0.306 |
| GK D (100 ps) | **1.206** | — | 1.84 |

The stochastic target restores the canonical KE variance (0.437/0.429 versus 0.440 eV), but the
Green–Kubo integral is **1.16–1.21e-9**, inconsistent with the MSD 1.44e-9. The deterministic
2.5 ns run has GK 1.42–1.45 and MSD 1.406, i.e. GK≈MSD. The extra stochastic fluctuations modify
the negative translation-ACF tail (100 fs 0.000 versus 0.011 deterministic; 500 fs −0.020 versus
−0.019), lowering the GK integral.

**Decision:** the water 2.5 ns production stays **deterministic** because it preserves GK/MSD
consistency; the stochastic 2.5 ns is a thermodynamic ablation, not the main method. Quartz keeps
the stochastic phonon-band rescale because there is no GK/diffusion claim and the deterministic
species rescale suppresses the KE variance to 0.0036 eV. The open problem is a rescale that
restores the KE marginal without changing the translation-ACF integral.

## 24. D slow-mode feasibility — a 0.57% τ=200 fs mode matches the 5 ps ACF integral (2026-11-23)

The realized deterministic-GLR translation ACF (100 ps) has a 5 ps integral of **10.258 fs**;
the reference is **11.341 fs** (a 1.08 fs area deficit). Adding a small independent OU component,

\[
C(t)=(1-w)C_{\rm GLR}(t)+w e^{-t/\tau},
\]

and constraining the integral to the reference gives \(w=0.00571\), \(\tau=200\) fs. The combined
ACF matches the reference at 5/25/100/200/500/1000/2000/5000 fs to RMSE 0.0074 (short-lag max
error 0.016):

| lag (fs) | 5 | 25 | 100 | 200 | 500 | 1000 | 2000 | 5000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0.981 | 0.653 | 0.031 | −0.080 | −0.022 | 0.005 | −0.004 | 0.001 |
| GLR | 0.981 | 0.640 | 0.011 | −0.081 | −0.022 | 0.005 | 0.002 | 0.002 |
| **+ slow mode** | **0.981** | **0.642** | **0.015** | **−0.078** | **−0.022** | **0.005** | **0.002** | **0.002** |

The slow component is only 0.57% of the translation ACF, so its effect on RDF/VDOS should be
small. Implementation: `--glr-slow-tau 200 --glr-slow-weight 0.005706` in `glr_readout.py`; the
slow state is a coherent molecular-COM OU field (no per-atom noise). 100 ps gate running:
check D/GK, RDF, VDOS, mode partition. If D moves toward 1.8 and structure holds, this is a cheap
kernel fix; if not, the full damped-oscillator embedding is still required.
