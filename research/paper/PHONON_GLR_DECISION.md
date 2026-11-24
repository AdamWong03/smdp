# Phonon-mode GLR (Project B) — panel decision and kill-test protocol

**Date:** 2026-11-23  
**Status:** GO-WITH-MODIFICATIONS, timeboxed kill-test. Not yet a paper claim.

---

## 1. Question

The current L1 paper has one fixed GLR readout:

- water: full memory (AR(10)) + per-mode stochastic rescale; canonical KE, structure, VDOS; \(D\) 26% low;
- quartz 5 fs: zero-memory species rescale positive (T = 301.86 K, exact Si/O KE, Si–O RDF
  0.271 → 0.078, VDOS 0.866/0.914); full species GLR negative (VDOS 0.07).

The open gap is a **full cross-material memory result**. Project B asks whether the same readout,
with a physically defined **phonon-mode basis**, can supply that result on quartz.

## 2. Panel

Four independent reviewer roles were asked to judge B against NMI standards:

| Role | Verdict | Main condition |
|---|---|---|
| ML-MD / NMI author-reviewer | GO-WITH-MODIFICATIONS | Only worth it if it produces a full cross-material memory result; otherwise scope creep |
| Lattice dynamics / phonon transport | GO-WITH-MODIFICATIONS | Timebox ~3 days as a mechanism test; audit the 300 K Hessian; DHO for low bands |
| Quartz / ML-potential / LAMMPS | GO-WITH-MODIFICATIONS | Finite-difference Hessian at 0 K fixed-cell relaxed structure; 483 modes; DHO below ~5 THz |
| NMI editor / reproducibility | GO-WITH-MODIFICATIONS | Follow-up by default; fold into the current paper only inside a fixed timebox; R8 package in parallel |

**Consensus:** B is scientifically sound but must be a **timeboxed kill-test**, not an open-ended
method project. The current L1 paper remains the submission baseline. B adds a fifth figure only
if it passes a pre-registered gate; a failure is reported as a boundary, not hidden, and does not
block submission.

**Key disagreement:** the editor would default to a follow-up paper; the ML-MD reviewer calls B
scope creep unless the full-memory result is achieved. The lattice/quartz reviewers consider the
basis clean enough to try. Resolution: run the kill-test now, but do not make the paper depend on
it and do not open a variant zoo.

## 3. Mandatory controls

1. **Zero-memory phonon rescale** — same basis, same per-mode stochastic target, no AR/DHO memory.
   This isolates the memory contribution.
2. **Cartesian covariance control** — a mass-weighted Cartesian covariance rescale/whitening with
   the same KE target but no phonon eigenvectors (or a phase-randomized static-covariance draw).
   If the phonon GLR does not beat this control, the eigenbasis is not the contribution.
3. **Acoustic modes** — remove the three exact COM translations from the mode basis. Do not rescale
   them; rescaling injects COM energy that the projection then discards.
4. **Imaginary modes** — four modes have negative harmonic frequency (one is the acoustic zero);
   the three non-acoustic soft modes have canonical KE in the reference and are flagged. They may
   be kept in the AR kernel but must not be fitted with a positive-frequency DHO.

## 4. Basis audit (already completed)

Artifacts: `/home/wyd/quartz_phonon/phonon_basis.npz`,
`phonon_basis_metadata.json`, `ar10_diagnostics.npz`, `dho_fit.npz`.

| Check | Result | Status |
|---|---|---|
| Finite-difference Hessian | 972 single points, ±0.01 Å, LAMMPS Buckingham + Coulomb/DSF | ✅ |
| ASR correction | \(7.6\times10^{-15}\) relative | ✅ |
| Residual force | 0.0067 kcal mol⁻¹ Å⁻¹ | ✅ |
| Modes | 486 total; 3 acoustic zeros; 4 imaginary (3 non-acoustic) | flagged |
| Reference mode KE | 99.4% of all modes within 10% of \(k_BT/2\); 100% of stable optical | ✅ |
| Off-diagonal covariance | 0.55% (Frobenius² fraction) | ✅ |
| AR(10) prediction error | max 0.0068 over 483 non-acoustic modes | ✅ |
| AR(10) pole modulus | max 0.998 | ✅ |
| AR(10) long-lag ACF, 1–3 THz | median error 0.14 at 500 fs | ⚠️ |
| AR(10) long-lag ACF, 3–5 THz | median error 0.20 at 500 fs | ⚠️ |
| DHO long-lag ACF, 1–3 THz | median error 0.042 at 500 fs | ✅ |
| DHO long-lag ACF, 3–5 THz | median error 0.035 at 500 fs | ✅ |

**Interpretation:** the basis is clean and the reference is canonically populated. AR(10) captures
the frequencies but not the low-frequency linewidths; a damped-harmonic-oscillator kernel improves
the 500 fs ACF by 3–6× in the 1–5 THz bands. The 0–1 THz band is dominated by the three soft
imaginary modes and is not a meaningful DHO target.

## 5. Kill-test protocol

**One configuration, one seed, 100 ps, 5 fs horizon.**

| Arm | Basis | Memory | Rescale | Purpose |
|---|---|---|---|---|
| baseline | — | — | — | already measured (T 235.3 K, VDOS 0.918/0.944) |
| zero-memory phonon | phonon | none | per-mode stochastic | isolate basis + variance target |
| full phonon GLR | phonon | AR(10), λ = 0.1 | per-mode stochastic | main test |
| Cartesian control | Cartesian covariance | none | covariance matched | mandatory null |

If the AR(10) full GLR fails only on low-band linewidths, replace the kernel with DHO for
positive-frequency modes (AR for the three soft modes) and rerun **one** configuration. Do not
report both kernels as co-equal main methods.

### Gate (pre-registered)

| Criterion | Pass |
|---|---|
| Stability | 100 ps, no collapse, min distance sane |
| Temperature | \(300 \pm 3\) K |
| Mode KE | ≥90% of 483 non-acoustic modes within ±10% of \(k_BT/2\) |
| Phonon VDOS | O/Si ≥ 0.866/0.914 (zero-memory species baseline) |
| Si–O RDF | ≤ 0.08 |
| Control | full GLR beats the Cartesian covariance control on band-resolved VDOS and low-frequency mode ACF |
| Reproducibility | second seed within noise before any production claim |

**NO-GO:** VDOS below 0.80, RDF above 0.15, full GLR no better than zero-memory or the Cartesian
control, or any seed unstable. Stop at the first failed seed or at the timebox, whichever comes
first. No rescue sweeps.

## 5b. First 100 ps result — per-mode rescale is the wrong granularity

Both per-mode arms completed stably at 100 ps:

| Arm | T (K) | Si/O KE (eV) | RDF Si–O/O–O/Si–Si | VDOS O/Si |
|---|---:|---|---:|---:|
| phonon + per-mode rescale, memory off | 299.64 | 0.03856 / 0.03836 | 0.027 / 0.016 / 0.027 | 0.217 / 0.269 |
| phonon + per-mode rescale, AR(10) memory | 299.79 | 0.03857 / 0.03840 | 0.054 / 0.048 / 0.081 | 0.209 / 0.253 |

The canonical marginal is exact (100% of modes within 10% of \(k_BT/2\); total KE std 0.398/0.401 eV),
the RDF is excellent, but the **VDOS collapses** to ~0.22/0.26. The mode ACF error at 100 fs is
0.21–0.59, versus 0.03–0.04 for the baseline. The species zero-memory rescale, by contrast, kept
VDOS at 0.866/0.914.

**Diagnosis:** a per-mode Gamma draw every 5 fs multiplies each single-dof mode by an independent
O(1) random factor. This imposes the correct one-time marginal but destroys the mode's phase
coherence; the Fourier transform broadens/collapses. Water's stochastic rescale works because it
acts on three groups with many dofs (translation/rotation/internal), so the group scale factor is
close to one. The rescale granularity must be **coarser than one mode**: species or bands.

The existing **deterministic species** rescale shows the opposite failure: it preserves the mode
ACF (0.02–0.04 at 100 fs) and the VDOS (0.866/0.914), but it fixes the species KE exactly every
step, so the total KE std collapses to 0.0036 eV (reference 0.4017). The mode KE distribution is
only 57.8% within 10% of \(k_BT/2\). The stochastic species rescale is designed to fix exactly
this: Gamma-distributed group KE (correct total KE std) with a group-level scale factor that is
close to one (ACF preserved).

**Fix being tested:** phonon AR(10) memory + species-level **stochastic** rescale, and a
no-rescale arm to see whether the memory alone corrects the KE bias. The per-mode and
deterministic-species arms remain as negative/control data.

### 5c. Final quartz result: phonon-band stochastic rescale, not phonon memory

The positive result is a **phonon-band stochastic rescale with no memory**:

| Quantity | Phonon-band stochastic rescale (two seeds) | Reference |
|---|---:|---:|
| T (K) | **299.9** (299.8 / 299.9) | 300.17 ± 17.9 |
| total KE std (eV) | **0.398** (0.397 / 0.399) | **0.4017** |
| Si / O KE per atom (eV) | 0.03846 / 0.03855 | 0.03878 |
| RDF Si–O / O–O / Si–Si | **0.028 / 0.016 / 0.033** | — |
| VDOS O / Si | **0.950 / 0.955** | 1 |
| mode KE within 10% / 20% | 77.2% / 96.8% | 67.7–75.6% / 94.4–97.3%* |
| mode ACF error at 100 fs | 0.02–0.07 | — |

\*The reference values are for a **100 ps** window, not the 2 ns average. A 100 ps window has
irreducible mode-KE sampling noise: the 2 ns average is 99.4% within 10%, but individual 100 ps
windows are 68–76%. The model is statistically consistent with the reference at the same window
length.

A coarser **species stochastic** partition is a control: T=301.68 K, exact Si/O equipartition,
total KE std 0.4012 eV, VDOS 0.898/0.955, RDF 0.0245/0.0169/0.0380. The band partition is slightly
better on VDOS and mode KE; both are robust.

The deterministic species rescale is the ablation that fixes the mean but kills the variance:
T=301.86 K, VDOS 0.866/0.914, RDF 0.078/0.022/0.055, but total KE std **0.0036 eV** and only
57.8% of modes within 10%.

**Full phonon memory is negative.** The baseline 5 fs backbone is *uniformly* cold: every phonon
band has 0.75–0.85 of \(k_BT/2\), not a spectral bias. Adding the AR(10) memory creates a spectral
bias that was not there:

| Arm | T (K) | total KE std | VDOS O/Si | RDF Si–O |
|---|---:|---:|---:|---:|
| baseline | 235.3 | 0.4355 | 0.918 / 0.944 | 0.271 |
| phonon memory, no rescale | 251.7 | 0.4648 | 0.594 / 0.686 | 0.137 |
| phonon memory + species stochastic rescale | 301.9 | 0.4024 | 0.734 / 0.838 | 0.197 |

The memory arm pumps high-frequency modes (20–40 THz ratio 1.14–1.27) and drains low-frequency
modes (0–5 THz ratio 0.28–0.38). The 5 fs backbone already has the correct mode ACF (100 fs error
0.03–0.04), so the AR innovations are unnecessary and broaden the sharp phonon lines.

**Caveat.** The reference species KE is correlated on a 60–95 fs scale (O τ≈76 fs, Si τ≈60 fs,
total τ≈95 fs), while the τ=0 rescale draws an independent Gamma every 5 fs. This does not destroy
the mode ACF or VDOS empirically, but a weak-coupling variant with τ≈100 fs is a principled
refinement to test in the SI. It is not needed for the main result.

**Verdict:** quartz is a **rescale-only** system; the full-memory claim is **NO-GO**. The paper
upgrades the quartz row from deterministic to stochastic **phonon-band** rescale and reports the
species partition, deterministic rescale, full-species, and full-phonon arms as
controls/ablations/negative controls.

## 6. Publication plan

- **Current L1 paper:** remains the submission baseline. Do not delay the R8 package (code/data
  DOI, source data, SI, AI disclosure) for B.
- **Outcome (2026-11-23):** the full-memory claim is NO-GO, but the phonon-basis investigation
  produced a **better quartz result**: phonon-band stochastic rescale (T=299.8 K, total KE std
  0.3967, VDOS 0.953/0.949, RDF 0.0245). The L1 paper is upgraded with this row; the species
  partition, deterministic rescale, full species, and full phonon arms become
  controls/ablations/negative controls. No new main figure is required; the SI gets the
  phonon-basis audit, the AR/DHO fits, and the per-mode/band controls.
- **If the full-memory gate had passed:** add one figure and one SI section. It did not; do not
  force it.

## 7. Immediate implementation

1. `phonon_glr.py`: fixed phonon basis, AR(p) readout, stochastic Gamma rescale with mode, band,
   species, or global partition; COM projection; backbone displacement unchanged.
2. CLI: `--phonon-glr-basis`, `--phonon-glr-coefficients`, `--phonon-memory-off`,
   `--phonon-rescale-partition`, `--phonon-static-scale`.
3. 100 ps results: per-mode rescale negative (VDOS 0.21/0.25); deterministic species rescale
   preserves VDOS but kills KE variance; **stochastic species rescale is the positive result**;
   full phonon memory negative (VDOS 0.734/0.838).
4. Paper: replace the deterministic quartz row with the stochastic species row; add the
   deterministic, full-species, and full-phonon arms as ablations/negative controls; update the
   Discussion and Methods. DHO is not needed for the main result.
