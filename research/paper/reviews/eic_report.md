# Journal-Fit Review (EIC)

**Recommendation:** Major Revision

## Summary

The manuscript makes a clear and useful point: force-free propagators should be judged by their
long-time stationary measure, not by one-step error or short NVE stability. The diagnostic is
well constructed and the water results are strong: the GLR matches the canonical mean mode
partition, improves the RDFs and O VDOS over the published TrajCast checkpoint, and is stable and
initial-condition independent at 2.5 ns. The paper is well written and the figures are clear.

The main obstacle to acceptance is not the quality of the diagnostic but the breadth of the
claim. The title and abstract promise “thermodynamically consistent” molecular dynamics, yet the
position dynamics are deliberately decoupled from the velocity readout and the diffusion
coefficient is 26% low on both 2.5 ns starts. The paper is honest about this in Results and
Discussion, but the title/abstract still read as a stronger claim than the evidence supports. For
NMI, the paper also needs a sharper answer to “why is this a conceptual advance rather than a
mode-resolved thermostat?”

## Strengths

1. **The diagnostic is the best part of the paper.** The pointwise-vs-ensemble inversion
   (Section 2.1, Fig. 1) is simple, surprising, and useful. The NVE red herring (Section 2.2) and
   the mode-partition heat map (Fig. 1c) make the mechanism visible. This is a genuine
   contribution to how the field evaluates learned propagators.
2. **The structural results are strong and reproducible.** Two 2.5 ns starts give nearly
   identical partition, RDF, and VDOS; O–H RDF 0.038/0.044 versus published 0.118; O VDOS
   0.953 versus 0.937. This is a meaningful empirical advance over the published checkpoint in
   the authors’ analysis.
3. **The negative controls are valuable.** The full displacement correction crash, the rescale-off
   collapse, and the AR(1) failure are informative and honestly reported.
4. **The paper is readable and well organized.** The separation of structure and thermodynamics is
   a good narrative device.

## Weaknesses

### CRITICAL

**C1. The title/abstract overclaim “thermodynamic consistency.”**  
Location: title, abstract, Section 2.3, Discussion. The readout guarantees the **mean** mode kinetic
partition and the velocity ACF, not the full joint position–velocity distribution. The per-step
rescale suppresses mode-KE fluctuations (reference std ≈0.24 eV versus ≈0.002 eV in the readout),
and the position diffusion is 26% low. “Thermodynamically consistent” is therefore true only in a
narrow, first-moment sense. Either define the term explicitly in the abstract (mean mode partition
+ multi-lag ACF) and add “transport limitation” to the title/abstract, or narrow the title to
“mode-resolved thermodynamic readout for force-free water MD.” The current framing invites the
wrong reviewer.

### MAJOR

**M1. NMI fit: one water model, one backbone, no second system.**  
Location: entire Results; Discussion. The paper claims the readout is “backbone-agnostic” but only
tests the 250k relational backbone. For NMI’s broad readership, either add a second system
(another water model, a simple liquid, or a different backbone) or explicitly frame the paper as a
water method paper and remove “general recipe” language from the Introduction.

**M2. Novelty relative to GLE thermostats and Bigi–Ceriotti needs a sharper, empirical
differentiation.**  
Location: Introduction, Discussion. The current distinction is conceptual (“thermostat for an MD
integrator” vs “readout for a frozen propagator”). A reviewer will ask: if the readout is fitted to
the reference ACF, why is this not a mode-resolved GLE thermostat applied post hoc? The paper needs
one paragraph that states the precise delta (the diagnostic, the exact mean-partition rescale on a
blended backbone velocity, the structure/VDOS results beyond the published checkpoint) and,
ideally, a direct comparison to the action-map approach.

**M3. The diffusion limitation is not reflected in the paper’s main claim.**  
Location: abstract, Section 2.3, Table 1, Discussion. The 26% deficit is reported, which is good,
but the abstract still says “within 26%” in a way that reads as a success. For NMI, lead with the
limitation: the method reproduces structure/spectrum and the mean partition, but not the transport
coefficient. This is a framing issue, not a data issue.

### MINOR

**m1. “First configuration in this study”** (Section 2.3) is fine, but avoid anything that sounds
like a literature priority claim.

**m2. The H VDOS comparison** is caveated in the Results but the abstract only uses O VDOS; good.
Keep the caveat visible in Table 1.

**m3. Author list, code/data availability, and SI** are missing. NMI will require them.

## Questions for the authors

1. What is the falsifiable definition of “thermodynamically consistent” used in the title? Which
   observables would have to fail for the claim to be false?
2. Why is a 26% diffusion error acceptable for a paper that promises faithful long-time water
   physics? If it is not acceptable, what is the minimal position-consistent fix?
3. What happens with a second backbone or a second water model? Is the readout still
   backbone-agnostic?
4. How much reference NVT data is needed, and how sensitive are the results to the fitting block?

## Recommendation

**Major Revision.** The diagnostic and structural results are strong enough to justify a revised
paper. The revision should (i) recalibrate the title/abstract/claims around the mean-partition
guarantee and the transport limitation, (ii) add a precise prior-art delta, and (iii) either add a
second system/backbone or explicitly scope the paper to water. The diffusion deficit should be
treated as a central limitation, not a footnote.
