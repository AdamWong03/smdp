# Perspective Review (R3)

**Recommendation:** Minor–Major Revision

## Summary

The paper’s central idea—separating structure (backbone) from thermodynamics (readout)—is
conceptually attractive and could be useful beyond water. The diagnostic is the most transferable
contribution. The main perspective concerns are the gap between the broad framing and the
single-system evidence, the practical reference-data requirement, and the interpretation of a
velocity field whose fluctuations are deliberately constrained.

## Strengths

1. **The structure/thermodynamics separation is a good design principle.** It is easy to explain
   and could be adopted by other force-free propagators.
2. **The readout is lightweight.** No backbone retraining, no forces, no new labels; the cost is
   small compared with the backbone.
3. **The diagnostic is transferable.** Pointwise-vs-ensemble decoupling and mode-resolved
   evaluation apply to other learned dynamical maps.
4. **The negative results are valuable.** The field benefits from knowing which fixes do not work.

## Weaknesses

### MAJOR

**M1. “Backbone-agnostic” is asserted, not tested.**  
Location: abstract, Introduction, Discussion. The paper tests one backbone (250k relational) and
one published checkpoint. A second backbone (even the native ep145 or a small MLP) would turn the
claim into evidence. Otherwise, soften “backbone-agnostic” to “applied here to a frozen relational
backbone” and make it a future-work item.

**M2. The reference-NVT data requirement is a practical barrier.**  
Location: Methods, Discussion. The readout needs 100 ps of reference NVT data. For a new system,
this may be cheap, but the paper does not quantify the sensitivity to the reference length, block
choice, or thermostat. A short sensitivity analysis (e.g., fit on 20/50/100 ps, evaluate on a
held-out block) would make the method much more adoptable. Without it, the “training-free” framing
oversells the data efficiency.

**M3. The velocity field is not fully canonical.**  
Location: Section 2.2, Discussion. The per-step rescale fixes the mean mode KE but suppresses its
fluctuations, and the position dynamics are decoupled from the velocity. Applications that use
velocities directly (reactive flux, transition-path shooting, dynamical corrections) may be
sensitive to this. The paper should state the limitation explicitly and, if possible, test one
velocity-sensitive observable beyond VDOS.

### MINOR

**m1. Cost accounting.** The paper reports 15.4 ns/day for the GLR rollout, but not the cost of
generating the reference NVT data or fitting the readout. Add a small cost table: reference MD,
fit, rollout, analysis.

**m2. Scope.** NPT, long-range electrostatics, temperature transfer, and non-water liquids are not
addressed. State this explicitly in the Discussion and avoid “general recipe” language.

**m3. Stakeholders.** The paper’s broad title may attract readers working on solids, electrolytes,
or biomolecules, but the evidence is water-only. A short “who is this for / who is left out”
paragraph would set expectations.

## Questions for the authors

1. How sensitive are the results to the reference block (20/50/100 ps) and to fitting on the
   independent trajectory?
2. Can the readout be fitted on-the-fly or from NVE data, removing the NVT reference requirement?
3. What happens with a much worse backbone (e.g., native ep145)? Does the readout still help, or
   does it fail gracefully?
4. Does the suppressed KE fluctuation affect any velocity-sensitive observable?

## Recommendation

**Minor–Major Revision.** The perspective issues are mostly addressable by scoping and a small
sensitivity analysis. The “backbone-agnostic” claim should be softened unless a second backbone is
added; the reference-data requirement should be quantified; the velocity-field limitation should be
stated explicitly.
