# Fluctuation–Dissipation Readout (FDR) — 2026-11-21

**Goal:** make the canonical mode partition and stability properties of the composed
propagator + thermostat map, instead of fitting them with a loss.

---

## 1. Architecture

```
dq_pred, v_pred = Backbone(q, v)
v_fd            = FDReadout(q, v)        # mode-resolved AR(1) + FD noise
v_new           = (1-λ) v_pred + λ v_fd   # phase-preserving blend
v_new           = ModeRescale(v_new)      # each mode KE -> 0.5*dof*kT
q'              = q + dq_pred             # backbone displacement
```

- **FDReadout:** for translation / rotation / internal,
  `v_mode' = a_mode v_mode + sqrt(1-a_mode²) noise_mode`, with `Var(noise_mode)` set to the
  target mode KE. Gains from the reference NVT ACF: 0.9816 / 0.8385 / −0.567.
- **Blend λ = 0.1:** keeps the velocity direction close to the backbone prediction, so the
  position head stays in distribution.
- **ModeRescale:** per-mode variance restoration to equipartition. This is the key ingredient;
  the FD noise supplies fluctuations, the rescale fixes the stationary partition.
- The backbone is frozen. No training. The velocity head is not used for the readout; the
  displacement head is.

## 2. Ablations (5,000 steps, 25 ps, base 250k, val_04 start)

| Variant | Result |
|---|---|
| FDR full replacement (λ=1, no rescale, displacement correction) | **failed at 0.575 ps** (min 0.605 Å) |
| FDR blend λ=0.1, no rescale | stable, but **cooled to 14 K** (variance collapse) |
| **FDR blend λ=0.1 + mode rescale** | **stable 25 ps; T = 300.0 K; partition 2.45 / 2.50 / 2.36 eV** |

Reference: 2.373 / 2.430 / 2.609 eV. Base 250k: 0.538 / 0.786 / 6.044 eV.

The mode-resolved rescale is essential: the blend alone loses the FD variance; the full
replacement breaks the position head.

## 3. 25 ps ACF

| Mode | FDR blend+rescale | Reference |
|---|---:|---:|
| translation α(25 fs) | 0.731 | 0.658 |
| rotation α(25 fs) | −0.417 | −0.540 |
| internal α(5 fs) | −0.878 | −0.567 |
| internal α(500 fs) | 0.096 | 0.104 |

The partition is correct; the internal mode is more coherent than the reference (VDOS may be
too sharp). The dynamics is not exact, but the thermodynamics is.

## 3b. 100 ps screen

- stable, 20,000/20,000 steps;
- T = **300.0 K**;
- partition: translation **2.445**, rotation **2.496**, internal **2.355 eV** (target 2.48);
- ACF unchanged: translation α(25) 0.734, rotation −0.426, internal α(5) −0.878.

RDF/VDOS analysis:

| Model | O–O RDF | O–H RDF | H–H RDF | H VDOS | O VDOS |
|---|---:|---:|---:|---:|---:|
| Base 250k | 0.646 | 1.312 | 0.249 | 0.406 | 0.487 |
| M0 5k | 0.191 | 1.790 | 0.082 | 0.668 | 0.888 |
| **FDR v1** | **0.062** | **0.301** | **0.110** | **0.807** | **0.851** |
| Published | 0.043 | 0.118 | 0.096 | 0.624 | 0.937 |

Motion partition: 2.482 / 2.533 / 2.392 eV. The FDR is stable, at equipartition, and improves
the O–O/H–H RDF and the H VDOS far beyond the base. The **O–H RDF (0.301 vs 0.118)** is the
remaining structural gap: the intramolecular bond distribution is controlled by the position
head, not by the velocity readout.

### 3c. Independent-start 100 ps

| Quantity | val_04 start | independent start | Published |
|---|---:|---:|---:|
| T | 300.0 K | 300.0 K | 300.2 K |
| translation / rotation / internal | 2.482 / 2.533 / 2.392 | 2.482 / 2.532 / 2.393 | 2.505 / 2.463 / 2.444 |
| O–O RDF | 0.062 | **0.057** | 0.043 |
| O–H RDF | 0.301 | 0.304 | 0.118 |
| H–H RDF | 0.110 | 0.104 | 0.096 |
| H VDOS | 0.807 | 0.808 | 0.624 |
| O VDOS | 0.851 | 0.843 | 0.937 |

The independent start reproduces the val_04 result almost exactly: the FDR stationary measure
and structure are **independent of the initial condition**. The O–H RDF gap is reproducible,
not a start artifact.

### 3d. Internal-blend 1.0 is worse

| Internal blend | O–O RDF | O–H RDF | H–H RDF | H VDOS | O VDOS |
|---|---:|---:|---:|---:|---:|
| **0.1 (v1)** | **0.062** | **0.301** | 0.110 | **0.807** | **0.851** |
| 1.0 | 0.187 | 0.957 | 0.074 | 0.618 | 0.613 |

The internal 5 fs ACF matches the reference almost exactly at blend 1.0 (−0.558 vs −0.567), but
the long-lag ACF decays too fast (≈0 at 50 fs vs ref −0.442). A single-gain AR(1) has the wrong
spectrum: it fixes the short-time coherence but destroys the O–H RDF and VDOS. The v1 blend
(0.1) remains the best overall.

**Conclusion:** the internal O–H gap needs a **memory kernel / higher-order AR readout** that
matches the ACF at multiple lags, not a single gain. Do not increase the internal blend further.

### 3e. 2.5 ns screen

- stable, 500,000/500,000 steps; T = 300.0 K;
- O–O RDF **0.0373** (better than the published 0.043 in our analysis);
- O–H RDF **0.4095** (the gap persists);
- H–H RDF **0.1125**;
- H / O VDOS **0.797 / 0.851**.

**Momentum bug found and fixed:** the FD translation noise had a net global COM. The pre-fix
2.5 ns trajectory drifted **1740 Å** over 2.5 ns. The readout now projects both the velocity and
the displacement to zero net linear momentum; a 25 ps check gives COM velocity 7.5e-11 and drift
3.4e-6 Å. The relative dynamics/RDF/VDOS are unaffected (the drift is a uniform translation);
the diffusion analysis also removes the global COM.

### 3f. Diffusion

| Model | D (m²/s) |
|---|---:|
| **FDR v1** | **3.59e-9 ± 0.28e-9** |
| Reference | 1.88e-9 ± 0.31e-9 |
| Published | 2.18e-9 |

The FDR diffusion is **1.9× too fast**. The translation ACF has a long positive tail (~0.08 at
5–10 ps) that the reference does not have; the single-gain AR(1) is too persistent. The O–H gap
and the diffusion excess share the same cause: **a single gain cannot match the multi-lag
ACF/spectrum**.

**Verdict:** v1 is a partial method: partition/T/O–O/H–H/VDOS strong, O–H and diffusion off.

## 4. What this is and is not

- **Is:** an architectural readout layer that guarantees the canonical mode partition and
  stability by construction. Backbone-agnostic; no reference labels; no training.
- **Is not:** a TrajCast copy (our backbone, new FD/mode-rescale readout); a loss patch; a
  global thermostat (the correction is mode-resolved and tied to the FD noise).
- **Open:** the position head is still the base model; RDF/VDOS/diffusion must be measured.
  The ACF/VDOS may require learned gains or a residual drive.

## 5. Next

1. 100 ps screen of the blend+rescale readout: partition, RDF, VDOS.
2. Independent-start screen.
3. If RDF/VDOS are acceptable: learn state-dependent gains / residual drive; 2.5 ns.
4. If RDF/VDOS are poor: the backbone's displacement head is the limit; architecture work on
   the displacement, with the FDR readout kept.

## 6. Next architecture: Generalized Langevin Readout (GLR)

The single-gain AR(1) matches one lag. The reference ACFs have multi-lag memory:

| Mode | α(5 fs) | α(50 fs) | α(500 fs) | α(1 ps) |
|---|---:|---:|---:|---:|
| translation | 0.982 | 0.208 | −0.019 | ~0 |
| internal | −0.567 | −0.442 | 0.104 | ~0 |

Fit an AR(p) per mode from the reference ACF (Yule–Walker):

```
v_mode(t) = Σ_{k=1..p} a_k v_mode(t-k) + ε(t)
```

- p = 3–5 can match the ACF at multiple lags and the diffusion integral;
- `Var(ε)` from the Yule–Walker prediction error preserves the target stationary variance
  (the FD relation for AR(p));
- keep the per-mode variance rescale for exactness;
- the readout keeps a small state buffer of the last p mode velocities.

Expected: remove the translation long tail (diffusion) and match the internal spectrum
(O–H RDF / VDOS), while keeping the canonical partition by construction. This is a natural
architecture upgrade of the same readout, not a loss patch.

## 7. GLR v2 first result (25 ps)

AR(10) coefficients fitted from the reference ACF (Yule–Walker); prediction-error ratios
translation 0.0042, rotation 0.0576, internal 0.1146; readout blend 0.1 + variance rescale.

| Mode / lag | GLR | Reference |
|---|---:|---:|
| translation α(5 fs) | 0.980 | 0.982 |
| translation α(25 fs) | 0.632 | 0.658 |
| translation α(500 fs) | **−0.024** | **−0.019** |
| rotation α(25 fs) | −0.608 | −0.540 |
| internal α(5 fs) | **−0.578** | **−0.567** |
| internal α(50 fs) | −0.541 | −0.442 |
| internal α(500 fs) | 0.297 | 0.104 |

- stable 25 ps; T=300.0; partition 2.445/2.490/2.362;
- **translation ACF now matches the reference at all lags** — the diffusion long tail should
  be removed;
- internal short-lag matches, but the 500–1000 fs tail is still too persistent (AR(10) may be
  too short for the internal memory);
- 100 ps + RDF/VDOS is running.

**100 ps:** stable; T=300.0; partition 2.445/2.490/2.362. Translation α(5/25/100/500 fs) =
0.981/0.640/0.014/**−0.030** (ref 0.982/0.658/0.039/−0.019). Rotation α(25) −0.610 (ref
−0.540). Internal α(5) −0.575 (ref −0.567), α(500) 0.312 (ref 0.104) — the internal long tail
persists.

**100 ps RDF/VDOS (GLR vs v1 vs published):**

| Metric | GLR | FDR v1 | Published |
|---|---:|---:|---:|
| O–O RDF | **0.0322** | 0.062 | 0.043 |
| O–H RDF | **0.1043** | 0.301 | 0.118 |
| H–H RDF | **0.0256** | 0.110 | 0.096 |
| H VDOS | **0.8484** | 0.807 | 0.624 |
| O VDOS | **0.9637** | 0.851 | 0.937 |

**All structural/spectral metrics beat the published model in our analysis.** The AR(10)
memory fixed the O–H RDF (0.301 → 0.104), and translation ACF matching fixed the diffusion
tail. 2.5 ns diffusion and independent-start 100 ps are running.

**Independent-start 100 ps:** O–O 0.0358, O–H **0.0981**, H–H 0.0268; VDOS 0.851/0.963;
partition/ACF identical to the val_04 run. **Initial-condition independence is confirmed for
partition, ACF, RDF, and VDOS.** Only the 2.5 ns diffusion remains.

**Displacement-correction test (negative).** Adding \(\Delta t(v^{\rm GLR}-v^{\rm pred})\) to the
backbone displacement (`--fd-correct-displacement`) crashed at 24.8 ps (min 0.647 Å). The
decoupled scheme—positions from the backbone, velocities from the readout—is therefore not just
a design choice but a stability requirement. The readout must not be integrated as a second
position update.

**100 ps MSD diagnostic.** Fitting the molecular-COM MSD over 10–80 ps gives D = 1.45e-9
(reference 1.81e-9), but the GLR MSD is superdiffusive at 50–100 ps; the 60–95 ps slope gives
1.90e-9. The production 2.5 ns block fit will decide the gate. The FDR v1 fast tail (3.59e-9) is
gone.
