# Phase 2 — Evidence Map

**Date:** 2026-11-22
**Purpose:** tie every manuscript claim to a figure/table, an experiment, and a data artifact.
**Status key:** ✅ complete · 🟡 running / partial · ⬜ missing

---

## 1. Claims → evidence

### C1 — Pointwise one-step accuracy is neither necessary nor sufficient for long-time equilibrium

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| Published TrajCast test MAE | 6.88e-5 Å (position), 3.41e-5 Å/fs (velocity) | `experiments/published_trajcast_repro_20261120` | ✅ |
| 250k pointwise error vs published | ~15× smaller on val transitions | diagnostic log / `CURRENT_METHOD_STATUS.md` §3 | ✅ |
| 250k 100 ps NVT | partition fails: 0.54 / 0.79 / 6.04 eV | `nvt_screen_250k_full_20261120/analysis` | ✅ |
| Published 100 ps NVT | partition passes: 2.505 / 2.463 / 2.444 eV vs 2.5 ns reference 2.459 / 2.502 / 2.450 eV | `nvt_screen_published_20261120/analysis` | ✅ |
| **Conclusion** | MSE ranking is anti-correlated with ensemble fidelity in this pair | — | ✅ |

**Figure:** Fig. 1a (scatter: pointwise MAE vs partition error), Fig. 1b (bar: partition per model).

### C2 — NVE energy stability is a red herring

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| Published 25 ps NVE | 302 → 476 K (heats strongly) | `nve_diagnostic_20261121` | ✅ |
| M0 25 ps NVE | 302 → 387 K | `nve_diagnostic_20261121` | ✅ |
| 250k base 25 ps NVE | 302 → 270 K (most stable) | `nve_diagnostic_20261121` | ✅ |
| NVT outcome | published passes; base fails | C1 artifacts | ✅ |
| **Conclusion** | NVE stability does not predict NVT fidelity; the composed stationary measure does | — | ✅ |

**Figure:** Fig. 1c (NVE temperature traces), Fig. 1d (NVT partition vs NVE drift).

### C3 — The failure is a composed propagator + thermostat stationary-measure problem

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| v0 stable 2.75 ns but partition 0.111 / 0.808 / 6.338 | collective cold, internal hot | `phase_space_20261120/water5_nvt_replica_01` | ✅ |
| 250k / Current-Only 150 ps | same failure direction | `nvt_screen_*_20261120` | ✅ |
| Native ep145 150 ps | opposite bias (3.256 / 3.105 / 1.041) | `nvt_screen_native_trajcast_20261120` | ✅ |
| M0 overshoot 3.373 / 3.270 / 1.000; independent start blow-up at 130 ps | short-horizon fix destabilizes | `nvt_pairs_20261121` | ✅ |
| CSRV moment loss | blow-up at 4.3 ps | `csrv_rollout_20261121` | ✅ |
| **Conclusion** | stability ≠ correct ensemble; mode partition is the discriminating variable | — | ✅ |

**Figure:** Fig. 1e (partition ternary/bar across all methods), Table 1.

### C4 — The GLR enforces the canonical mode partition by construction

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| Derivation | per-mode target KE = ½ N_dof k_B T; variance rescale each step | `FD_READOUT_20261121.md` §5 | ✅ |
| 100 ps val_04 | 2.482 / 2.526 / 2.399 eV; T=300.0 | `glr_100ps_20261122/analysis/motion.json` | ✅ |
| 100 ps independent start | 2.445 / 2.490 / 2.362 eV; T=300.0 | `glr_independent_100ps_20261122/analysis` | ✅ |
| 25 ps / 100 ps ACF | identical across starts | `mode_gain/glr_*.json` | ✅ |
| Velocity marginal | component kurtosis 2.93/5.49/5.93 (GLR) vs 3.04/5.62/5.57 (reference) | `mode_gain` velocity check | ✅ |
| Rescale-off ablation | variance collapses (FDR blend 0.1 cooled to 14 K) | `fd_rescale_ablation_20261121` | ✅ |
| **Conclusion** | **mean** partition is guaranteed and observed on two starts; mode-KE fluctuations are suppressed (std 0.002 vs ref 0.24 eV) | — | ✅ (rescale-off re-run with GLR pending) |

**Figure:** Fig. 2 (architecture), Fig. 3b (partition bars), Table 1.

### C5 — The GLR matches multi-lag mode dynamics

| Evidence | GLR | Reference | Artifact | Status |
|---|---:|---:|---|---|
| translation α(5/25/100/500 fs) | 0.981 / 0.640 / 0.014 / −0.030 | 0.982 / 0.658 / 0.039 / −0.019 | `mode_gain/glr_100ps.json` | ✅ |
| rotation α(25 fs) | −0.610 | −0.540 | same | ✅ |
| internal α(5 fs) | −0.575 | −0.567 | same | ✅ |
| internal α(500 fs) | 0.312 | 0.104 | same | 🟡 known residual (AR(10) too short) |
| AR(10) prediction-error ratios | 0.0042 / 0.0576 / 0.1146 | — | `glr_fit_20261122` | ✅ |
| p=1/5/20 fits | running | — | `glr_fit_p*_20261122` | 🟡 |
| **Conclusion** | short/mid-lag ACF matched; internal long tail is a stated limitation | — | — | 🟡 |

**Figure:** Fig. 3c (ACF panels translation/rotation/internal, GLR vs reference).

### C6 — The GLR reproduces water structure and VDOS at or beyond the published checkpoint

| Metric | GLR 2.5 ns | GLR independent 100 ps | Published | Status |
|---|---:|---:|---:|---|
| O–O RDF RMSE | **0.0307** (100 ps 0.0322) | **0.0358** | 0.043 | ✅ |
| O–H RDF RMSE | **0.0376** (100 ps 0.1043) | **0.0981** | 0.118 | ✅ |
| H–H RDF RMSE | **0.0282** (100 ps 0.0256) | **0.0268** | 0.096 | ✅ |
| H VDOS overlap | **0.8469** (100 ps 0.8484) | **0.8506** | 0.624 | ✅ (caveat below) |
| O VDOS overlap | **0.9526** (100 ps 0.9637) | **0.9628** | 0.937 | ✅ |

Artifacts: `glr_2p5ns_20261122/analysis`, `glr_100ps_20261122/analysis`,
`glr_independent_100ps_20261122/analysis`.

**Caveat:** H VDOS overlap depends on our reference H-stretch frequency (published scores 0.624
here while the paper reports ~0.99 with its own reference). O VDOS and the RDFs are the robust
comparisons; H VDOS is reported with the caveat.

**Figure:** Fig. 3d–f (RDF panels), Fig. 3g (VDOS).

### C7 — The GLR recovers the diffusion coefficient (**gate**)

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| Reference D | 1.84–1.88e-9 m²/s | `nvt_2p5ns/production.h5` | ✅ |
| FDR v1 D (2.5 ns) | 3.59e-9 (1.9× fast; AR(1) long tail) | `nvt_screen_fd_rescale_2p5ns_20261121` | ✅ |
| Published TrajCast D (paper) | 2.18e-9 | paper | ✅ |
| GLR 2.5 ns | **1.406 ± 0.155e-9 (25% low)**; T = 299.99 K, stable | `glr_2p5ns_20261122/analysis/diffusion.json` | ❌ |
| GLR Green–Kubo | **1.420e-9** vs reference 1.836e-9 | `green_kubo.json` | ❌ |
| GLR independent 2.5 ns | **1.393 ± 0.079e-9 (26% low)**; RDF 0.028/0.044/0.027; VDOS 0.847/0.953 | `glr_independent_2p5ns_20261122/analysis` | ✅ (D still low) |
| Mixed p20 translation | GK 1.61e-9; MSD 1.46e-9 (improves but does not close) | `glr_mixedp20_100ps_20261122` | 🟡 |
| **Gate** | D ≈ 1.9e-9, both starts | — | **not passed** |

**Verdict:** partition/RDF/VDOS pass; diffusion is 25% low. The Green–Kubo integral agrees with
the MSD, so the position update is consistent with the readout velocity; the translation ACF
integral is the bottleneck. Blend, translation-displacement, and COM-scaling fixes do not close
the gap without breaking consistency or structure. The paper reports the 25% deviation as a
limitation rather than claiming exact diffusion recovery.

**Figure:** Fig. 3h (MSD), Fig. 3i (D bar with reference band).

### C8 — The GLR is initial-condition independent

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| Two starts (275 K NVE vs 301.8 K NVT) | partition/ACF/RDF/VDOS agree within noise | 100 ps artifacts above | ✅ |
| Independent 2.5 ns | partition 2.482/2.527/2.398; RDF 0.028/0.044/0.027; VDOS 0.847/0.953; D 1.393e-9 | `glr_independent_2p5ns_20261122/analysis` | ✅ |
| **Conclusion** | stationary measure is not an artifact of val_04, including the 26% diffusion deficit | — | ✅ at 2.5 ns |

**Figure:** Fig. 3 overlays (two starts), Supplementary Table.

### C9 — Each GLR component is necessary

| Ablation | Configuration | Result | Status |
|---|---|---|---|
| AR order | p = 1, 5, 10, 20 | p=1 O–H 0.301 vs p10 0.104; p≥10 matches; p20 refines 100–200 fs | ✅ p1/p10; p5/p20 runs running |
| Internal blend | 0.1 vs 1.0 | 1.0 single-gain spectrum failure (O–H 0.957 in FDR v1) | ✅ FDR v1; GLR re-run pending |
| Variance rescale | on vs off | off → **T=172 K**, partition 0.76/0.65/2.84, RDF 0.398/0.474/0.279, VDOS 0.432/0.608 | ✅ GLR negative control |
| FD noise | on vs off | off → structure/D unchanged at λ=0.1 (RDF 0.034/0.098/0.026; VDOS 0.849/0.962; D 1.55e-9) | ✅ **not necessary at λ=0.1** |
| Seeds | 3 | partition identical; O–H 0.104/0.111/0.110; O VDOS 0.964/0.966/0.959 | ✅ |
| Frozen vs end-to-end | frozen vs fine-tuned backbone | scope / boundary | ⬜ (optional) |
| Displacement correction | full / translation-only / COM scale | full crashes at 24.8 ps; partial/COM do not close D and break consistency or RDF | ✅ negative controls |

**Figure:** Fig. 4 (four panels: RDF/VDOS vs p; partition vs rescale; O–H vs internal blend; seed
error bars).

### C10 — The readout is backbone-agnostic and training-free

| Evidence | Value | Artifact | Status |
|---|---|---|---|
| Frozen 250k backbone | no backbone gradients; readout fitted from reference ACF | code | ✅ |
| Readout cost | negligible vs backbone (185 MB peak, ~15 ns/day end-to-end) | `trajectory.summary.json` | ✅ |
| **v0 phase-space backbone + GLR** | baseline 0.111/0.808/6.338, RDF 0.944/1.409/0.229, VDOS 0.356/0.236 → GLR 2.482/2.529/2.396, RDF 0.042/0.109/0.019, VDOS 0.855/0.969 | `glr_v0_100ps_20261122/analysis` | ✅ |
| v0 + GLR independent start | partition 2.482/2.527/2.398; RDF 0.050/0.131/0.023; VDOS 0.857/0.977 | `glr_v0_independent_100ps_20261122/analysis` | ✅ |
| **Quartz phonon-band stochastic rescale (5 fs)** | baseline T=235.3, RDF 0.271, VDOS 0.918/0.944; **phonon-band stochastic rescale positive, two seeds**: T=299.8/299.9, KE std 0.397/0.399 vs 0.4017, Si/O KE 0.03846/0.03855, RDF 0.028/0.016/0.033, VDOS 0.950/0.955; species-partition control T=301.68, KE std 0.4012, exact Si/O KE, VDOS 0.898/0.955; deterministic ablation VDOS 0.866/0.914 but KE std 0.0036; full species GLR negative (VDOS 0.07); full phonon GLR negative (VDOS 0.734/0.838) | `quartz5_phonon_bands_rescale_only_100ps_20261123/analysis_5fs`, `quartz5_phonon_bands_rescale_seed2_100ps_20261123/analysis_5fs`, `quartz5_phonon_species_rescale_only_100ps_20261123/analysis_5fs`, `quartz5_species_rescale_100ps_20261122/analysis_5fs`, `quartz5_glr_100ps_20261122/analysis_5fs` | ✅ positive + negative controls |
| **Scope** | one water model, two architectures; quartz phonon-band stochastic rescale positive at 5 fs; full memory negative even with a phonon basis; memory useful only when the backbone mode ACF is wrong | — | 🟡 |

**Figure:** Table 3 (two-backbone comparison); Fig. 5 optional.

---

## 2. Figure plan (5 main figures)

| Figure | Panels | Claims | Data status |
|---|---|---|---|
| **Fig. 1 — Diagnostic** | a pointwise MAE vs partition error; b partition bars; c NVE T traces; d NVT partition vs NVE drift; e stability/partition table | C1, C2, C3 | ✅ |
| **Fig. 2 — GLR architecture** | a schematic (backbone + readout); b AR(p) equations; c algorithm box; d variance-rescale derivation | C4, C5 | ✅ (schematic to draw) |
| **Fig. 3 — Water NVT** | a protocol; b partition; c ACF (3 modes); d–f RDF; g VDOS; h MSD; i diffusion | C4–C8 | ✅ (2.5 ns complete) |
| **Fig. 4 — Ablations** | a RDF/VDOS vs p; b rescale on/off; c internal blend; d seeds | C9 | 🟡 |
| **Fig. 5 — Scope/cost** | a ns/day and cost; b water+quartz scope; c limitations | C10 | 🟡 |

**Supplementary:** full stability table, all ACF lags, per-seed numbers, analysis details, H VDOS
caveat.

---

## 3. Experiment queue (ordered by gate value)

| # | Experiment | Length | Why | Status |
|---|---|---|---|---|
| 1 | GLR 2.5 ns val_04 | 2.5 ns | **diffusion gate** | 🟡 276k/500k |
| 2 | GLR 2.5 ns independent | 2.5 ns | gate + initial-condition independence | 🟡 started |
| 3 | GLR p=1/5/20 fits | CPU | ablation coefficients | 🟡 running |
| 4 | GLR p=1/5/20 100 ps | 100 ps each | memory-kernel necessity | ⬜ queued |
| 5 | GLR rescale-off 100 ps | 100 ps | partition guarantee | ⬜ needs CLI flag |
| 6 | GLR FD-noise-off 100 ps | 100 ps | noise necessity | ⬜ needs CLI flag |
| 7 | GLR seeds 2/3 100 ps | 100 ps each | error bars | ⬜ |
| 8 | Published TrajCast 2.5 ns | 2.5 ns | fair baseline diffusion | ⬜ queued |
| 9 | Second backbone (ep145 + GLR) | 100 ps | agnosticism (optional) | ⬜ |
| 10 | Quartz / second system | — | scope (optional; else water-only) | ✅ phonon-band stochastic rescale positive, two seeds; full memory negative |

---

## 4. Manuscript claim boundaries

**Supported now**

- pointwise accuracy ≠ ensemble fidelity (water, two models);
- NVE stability ≠ NVT fidelity (water, three models);
- mode-resolved partition is the discriminating diagnostic;
- GLR guarantees the canonical partition and matches multi-lag ACFs;
- GLR reproduces water RDF/VDOS better than the published checkpoint in our analysis;
- initial-condition independence at 100 ps.

**Not yet supported / must not claim**

- exact diffusion recovery (D is 26% low on two starts);
- universal backbone-agnosticism (two architectures tested; more needed);
- full cross-material memory kernel (quartz stochastic species rescale is positive; full species and full phonon memory are negative; memory is useful only when the mode ACF is wrong);
- long-range electrostatics / large-scale physics (local 6 Å backbone; not addressed);
- temperature transfer, other water models, other liquids;
- end-to-end training benefit (readout is frozen-backbone);
- "first" / "only" claims without a broader benchmark.

---

## 5. Open risks

| Risk | Impact | Mitigation |
|---|---|---|
| Diffusion still fast/slow at 2.5 ns | blocks the main claim | p=20 / internal long memory; check MSD linearity |
| Internal ACF long tail (0.312 vs 0.104) | reviewer questions memory kernel | report as limitation; p=20 ablation |
| H VDOS analysis mismatch vs paper | unfair-looking comparison | report O VDOS as primary; state the reference caveat |
| "Just a GLE thermostat" | novelty rejection | diagnostic + exact rescale + structure results + ablations |
| Bigi/Ceriotti action map overlap | novelty rejection | conceptual comparison; run if code released |
| Single seed / single system | weak statistics | water: two starts, three 100 ps seeds; quartz: two seeds; stochastic water GK/MSD trade-off reported |
