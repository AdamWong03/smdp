# Diffusion Gate — GLR 2.5 ns

**Status:** running; fill after `experiments/glr_2p5ns_20261122` and
`experiments/glr_independent_2p5ns_20261122` finish.
**Decision rule:** D in the reference band (roughly 1.4–2.5e-9 m²/s) on both starts, with
stable T = 300 K and no structural collapse. If D is outside the band, iterate (partial
displacement coupling, AR order, blend) before locking the paper positioning.

---

## 1. Main run (val_04 start)

| Quantity | Value |
|---|---|
| Run | `experiments/glr_2p5ns_20261122` |
| Steps | 500,000 (2.5 ns) |
| Stable | **yes** |
| T | 299.994 K |
| Minimum distance | > 0.93 Å |
| Partition (trans/rot/int) | **2.482 / 2.526 / 2.399 eV** (2.5 ns ref 2.463 / 2.501 / 2.447) |
| O–O / O–H / H–H RDF | **0.0307 / 0.0376 / 0.0282** |
| H / O VDOS | 0.847 / 0.953 |
| D (m²/s) | **1.406 ± 0.155e-9** |
| D reference | 1.883 ± 0.306e-9 |
| D published | 2.18e-9 |
| D FDR v1 | 3.59e-9 |
| Deviation | **−25.3%** vs reference; published is +15.8% |

**Green–Kubo cross-check.** The mass-weighted translation ACF integral gives
D_GK = 1.420e-9 (GLR) and 1.836e-9 (reference), in agreement with the MSD values. The
position update is therefore consistent with the readout velocity; the bottleneck is the
translation ACF integral. The GLR translation ACF at 100 fs is 0.011 versus the reference 0.039;
the AR(20) fit matches this lag (0.039). A mixed readout with translation AR(20) and
rotation/internal AR(10) is the targeted fix under test.

**Blend sweep (p10 translation).** Raising the translation blend makes diffusion worse:

| translation blend | D (MSD 10–80 ps) | D_GK | ACF(100 fs) |
|---|---:|---:|---:|
| 0.1 | 1.45e-9 | 1.42e-9 | 0.011 |
| 0.3 | 0.97e-9 | 1.09e-9 | −0.052 |
| 0.5 | 1.05e-9 | 1.02e-9 | −0.116 |

The reason is that the AR(10) translation ACF itself is −0.032 at 100 fs (wrong sign), so a
larger admixture pulls the trajectory further from the reference. The AR(20) translation ACF is
+0.039 at the same lag. The next test is the mixed p20-translation readout with a larger
translation blend (0.3/0.5), which should move the trajectory ACF toward the reference instead of
away from it.

**Mixed p20 + blend 0.3 (negative).** The mixed readout improves D_GK to 1.61e-9 at blend 0.1,
but raising the translation blend to 0.3 gives D = 1.34e-9 and a negative 100 fs ACF (−0.034).
The blend lever is not sufficient because the closed-loop trajectory ACF is not a linear blend of
the backbone and readout velocities.

**Translation-only displacement correction (negative).** A translation-only correction
\(q_{\rm COM}\leftarrow q_{\rm COM}+\alpha\Delta t(v^{\rm trans}_{\rm GLR}-v^{\rm trans}_{\rm pred})\)
was stable at \(\alpha=0.5,1.0\) but did not help: D = 1.36/1.43e-9 (MSD) and
1.25/1.36e-9 (GK). The COM displacement correction is not the right lever.

**COM displacement scaling (negative).** Scaling each molecule's COM displacement by
\(\beta\) leaves the internal geometry unchanged but directly changes the MSD. \(\beta=1.05\)
gives MSD D = 2.33e-9 and GK D = 1.74e-9; \(\beta=1.15\) gives MSD 3.04e-9 and GK 2.22e-9.
The MSD and GK values disagree, so the scaling breaks velocity--position consistency, and the
RDF degrades at \(\beta=1.15\) (O--O 0.096, O--H 0.127). We do not adopt it.

**Verdict.** The decoupled position update is the limiting factor for diffusion. None of the
simple blend, translation-displacement, or COM-scaling fixes closes the 25% gap without breaking
consistency or structure. The paper therefore reports the 25% deviation as a limitation and does
not claim exact diffusion recovery. A position-consistent readout or an end-to-end COM head is
future work.

Note: the first 100 ps reference block has partition 2.373/2.430/2.609 eV and is an outlier;
the 2.5 ns mean 2.459/2.502/2.450 eV is the long-time reference for the partition and diffusion
comparisons.

## 2. Independent run

| Quantity | Value |
|---|---|
| Run | `experiments/glr_independent_2p5ns_20261122` |
| Steps | 500,000 (2.5 ns) |
| Stable | **yes** |
| T | 300.0005 K |
| Minimum distance | > 0.94 Å |
| Partition (trans/rot/int) | **2.482 / 2.527 / 2.398 eV** (ref 2.463 / 2.501 / 2.447) |
| O–O / O–H / H–H RDF | **0.028 / 0.044 / 0.027** |
| H / O VDOS | 0.847 / 0.953 |
| D (m²/s) | **1.393 ± 0.079e-9** (26% low) |

## 3. Decision

- [x] Partition/RDF/VDOS stable at 2.5 ns, both starts
- [ ] Main D in band — **1.406e-9, −25.3%** (structure passes; diffusion does not)
- [ ] Independent D in band — **1.393e-9, −26.0%**
- [x] Independent start reproduces the stationary measure (partition/RDF/VDOS/D)
- [ ] Lock the paper positioning: **backbone structure + mode-resolved generalized-Langevin
      thermodynamics**
- [ ] Proceed to Phase 4 `full` drafting

**Final gate verdict (2026-11-22):** the 2.5 ns GLR passes partition, RDF, VDOS, stability, and
initial-condition independence on two starts, but diffusion is robustly 26% low. The Green–Kubo
integral confirms the translation ACF is the bottleneck. Mixed AR(20) translation, larger blends,
translation-only displacement correction, and COM scaling do not close the gap without breaking
consistency or structure. The paper should lock the **mode-resolved thermodynamic consistency and
structure** claim and report the 26% diffusion deviation as a quantified limitation. Do not claim
exact diffusion recovery. A position-consistent readout or an end-to-end COM head is future work.

## Stochastic rescale and kernel-order follow-up (2026-11-22)

**Stochastic per-mode rescale** (Gamma/CSVR target) passes the 100 ps screen:
T = 300.16 ± 17.84 K; total KE std 0.4405 vs reference 0.4417 eV; mode-KE std
0.253/0.257/0.255 vs reference 0.239/0.234/0.250 eV; partition 2.483/2.531/2.397; RDF
0.036/0.114/0.022; VDOS 0.854/0.957; D = 1.342e-9. The KE-marginal hole is fixed; D is still
~1.34e-9. The deterministic rescale is now an ablation.

**Kernel-order feasibility for B3.** The reference translation ACF has a negative dip
(−0.078 at 200 fs) and a small positive tail (500 fs −0.019). A positive Prony series cannot fit
the negative dip. Raw AR orders do not fix D monotonically: p10 gives D_GK 1.42e-9, p20 1.61e-9,
p40 4.20e-9 (the p40 AR tail is too positive). A proper Green–Kubo-constrained Markovian
embedding—positive memory kernel with oscillatory modes and an explicit zero-frequency D
constraint—is a real method-development project, not a one-day fix. For the current paper, keep
the 26% D as a quantified limitation and report the kernel-order analysis as the mechanistic
explanation.

## 4. Artifacts

- Analysis script: `method_development/run_glr_2p5ns_analysis.sh`
- Analysis output: `experiments/glr_2p5ns_20261122/analysis/`
- Report: `analysis/glr_2p5ns_report.json`
- Figure 3 updates automatically when `analysis/msd.csv` exists.
- Standalone cross-check: `code/h1h2/quick_diffusion_2p5ns.py --candidate ... --reference ...`
  (8 blocks, 50 fs sampling, 10--100 ps fit).


## Stochastic 2.5 ns — canonical KE variance vs GK/MSD consistency (2026-11-23)

Two 2.5 ns stochastic-rescale runs give T=299.97/300.08 K, partition 2.483/2.525/2.398 and
2.482/2.528/2.398 eV, total KE std **0.437/0.429 eV** (reference 0.440), and MSD diffusion
1.435/1.387e-9. The Green–Kubo integral is **1.16–1.21e-9**, inconsistent with the MSD; the
deterministic 2.5 ns run has GK 1.42–1.45 and MSD 1.406 (consistent). The stochastic target fixes
the KE marginal but adds a negative translation-ACF tail. The water production therefore stays
deterministic; the stochastic run is a thermodynamic ablation. The transport deficit remains a
translation-kernel limitation.
