# Methodology Review (R1)

**Recommendation:** Major Revision

## Summary

The methodology is generally well designed: a controlled diagnostic, multiple baselines, two
initial conditions, a 2.5 ns production run, seed and ablation studies, and a clear separation
between the readout fit and the structural observables. The authors are unusually honest about
the negative results. The main methodological concerns are (i) the 26% diffusion deficit and its
relationship to the central claim, (ii) the reference-data split and potential leakage, (iii) the
FD-noise ablation, and (iv) statistical reporting for the long runs.

## Strengths

1. **Controlled diagnostic.** The published checkpoint vs 250k comparison uses the same held-out
   transitions and the same NVT protocol. The NVE control is a clean falsification test.
2. **Multiple baselines.** 250k, Current-Only, native ep145, M0, CSRV, FDR v1, published
   TrajCast. The negative controls are informative.
3. **Ablations.** AR order, rescale, FD noise, blend, displacement coupling. The rescale-off and
   displacement-correction results are strong.
4. **Two-start replication.** The 100 ps and 2.5 ns independent-start results are a meaningful
   robustness check.
5. **Reporting.** The paper reports block std, fit \(R^2\), and the exact diffusion protocol.

## Weaknesses

### CRITICAL

**C1. The diffusion deficit is a central methodological failure, not a side note.**  
Location: Section 2.3, Table 1, Discussion. The paper’s stated goal is faithful long-time water
physics, and \(D\) is 26% low on both 2.5 ns starts. The Green–Kubo and MSD estimates agree
(1.42 vs 1.41), so this is not a sampling artifact; the position update is consistent with the
readout velocity, but the velocity ACF integral is low. The paper reports this honestly, but the
title/abstract still imply a broader success. A methodological reviewer will not accept “within
26%” as a pass for a transport coefficient without either (a) a position-consistent fix, or
(b) a reframing of the claim to exclude exact transport.

### MAJOR

**M1. Reference-data split and potential leakage.**  
Location: Methods, “Reference data and force-free backbone.” The readout is fitted on 500–600 ps
of the same 2.5 ns reference trajectory used for evaluation. The paper says RDF/VDOS/D are not
used in the fit, which is true, but the reference ACF and the reference observables come from the
same trajectory. A reviewer will ask for a held-out check: fit on 0–100 ps and evaluate on
500–600 ps, or fit on the independent trajectory and evaluate on the main reference. This is a
cheap, high-value revision.

**M2. FD noise is not necessary at λ = 0.1, but the paper still sells it as a component.**  
Location: Section 2.7, abstract, Introduction, Discussion. The noise-off run gives essentially the
same partition, RDF, VDOS, and a slightly higher \(D\) (1.55 vs 1.45). The paper says this
honestly, but then the “fluctuation–dissipation-consistent noise” remains in the abstract and the
method name. Either (a) show a regime where the noise matters (larger λ, a different system, or
the one-point velocity distribution), or (b) reframe the method as an AR(\(p\)) memory readout with
optional FD noise, and move the noise to a theoretical detail.

**M3. Statistical reporting for the 2.5 ns runs.**  
Location: Methods, Table 1, Fig. 3h. The 2.5 ns \(D\) is one seed per start. The block std is
reported, but the seed-to-seed \(D\) is not. The 100 ps seed study shows the partition is identical,
but \(D\) at 100 ps is noisy. Report the seed-to-seed \(D\) at 100 ps or run a second 2.5 ns seed.
Also state the block autocorrelation / effective sample size for the diffusion fit.

**M4. The AR order selection is tuned on the same observable used to claim success.**  
Location: Section 2.7. p10 is chosen because it gives the best 100 ps O–H RDF; p20 has the better
long-lag ACF but a slightly worse 100 ps O–H. This is a legitimate engineering choice, but the
paper should justify p10 using a pre-specified criterion (e.g., ACF match + stability + diffusion),
not only O–H. Otherwise it looks like selecting the order on the test metric.

**M5. Baseline fairness.**  
Location: Methods, Table 1. The published checkpoint is run with its native CSVR and float64,
while the GLR has no external thermostat. The comparison is fair as a method comparison, but the
paper should state the different thermostat treatment and the fact that the published 2.5 ns
\(D\) is taken from the paper, not re-run under our protocol. The native ep145 baseline is 10%
trained; it should be clearly marked as diagnostic, not as a fair architecture baseline.

### MINOR

**m1. The H VDOS comparison** is sensitive to the reference H-stretch frequency; the caveat is
present, but the table should flag it (it does). Keep it.

**m2. The “15×” pointwise-error ratio** depends on the held-out set and the checkpoint; report the
confidence interval or at least the per-graph distribution.

**m3. Code/data availability** is missing. NMI will require a repository/DOI and a reproducibility
statement.

## Questions for the authors

1. What is the seed-to-seed \(D\) at 100 ps? Does the 2.5 ns \(D\) change with a second seed?
2. What happens if the readout is fitted on 0–100 ps and evaluated on 500–600 ps, or on the
   independent trajectory?
3. Is there any regime where the FD noise changes the structure, spectra, or diffusion?
4. Why is p10 preferred over p20 if p20 has the better long-lag ACF? State the selection rule.

## Recommendation

**Major Revision.** The diagnostic and structural results are publishable, but the diffusion
deficit and the reference-data split must be addressed. The minimum path is: (i) reframe the claim
around mean partition + structure, (ii) add the held-out reference-fit check, (iii) report
seed-to-seed \(D\), (iv) reframe or justify the FD noise, and (v) provide code/data.
