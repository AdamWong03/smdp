# Domain Review (R2)

**Recommendation:** Major Revision

## Summary

The paper addresses a real problem in force-free water MD and the mode-resolved readout is
physically motivated. The water results are strong, and the negative controls are informative.
The main domain concerns are (i) a length-mismatched comparison with the published checkpoint,
(ii) the precise meaning of “thermodynamic consistency” when the mode-KE fluctuations are
suppressed, and (iii) the prior-art differentiation from GLE thermostats and the Bigi–Ceriotti
action map.

## Strengths

1. **The diagnostic is physically meaningful.** The mode partition (translation/rotation/internal)
   is the right low-dimensional projection for water, and the heat map (Fig. 1c) makes the failure
   mechanism visible.
2. **The readout is interpretable.** AR(\(p\)) coefficients from the reference ACF, FD noise, and
   per-mode rescaling are physically transparent, unlike a black-box velocity correction.
3. **The water results are strong.** The 2.5 ns O–H RDF 0.038 and O VDOS 0.953 are good; the
   initial-condition independence is convincing.
4. **The negative controls are useful.** The rescale-off collapse and the displacement-correction
   crash are physically informative.

## Weaknesses

### CRITICAL

**C1. The RDF/VDOS comparison with the published checkpoint is not length-matched.**  
Location: abstract, Table 1, Section 2.3, Fig. 3. The GLR RDF/VDOS values are from 2.5 ns runs,
while the published checkpoint values are from 100 ps. The 2.5 ns GLR O–H RDF (0.038) is compared
with the 100 ps published value (0.118), which exaggerates the improvement. At 100 ps, the GLR
O–H is 0.104 versus published 0.118—still better, but a much smaller margin. The paper must either
(a) compare at the same length (100 ps vs 100 ps), (b) run the published checkpoint for 2.5 ns, or
(c) report the 2.5 ns GLR values as a separate long-run convergence result and not as a head-to-head
with the 100 ps published model. This is a fairness issue, not a physics issue, but it must be
fixed before submission.

### MAJOR

**M1. “Thermodynamic consistency” is not defined precisely enough.**  
Location: title, abstract, Section 2.2, Discussion. The readout fixes the **mean** mode kinetic
energy and the velocity ACF, but suppresses the mode-KE fluctuations and does not guarantee the
position marginal. The paper should state the definition explicitly (e.g., “canonical mean mode
partition + reference velocity ACF”), explain that the joint position–velocity distribution is
not guaranteed, and discuss whether the suppressed fluctuations matter for RDF/VDOS/diffusion.
The current phrasing invites a stronger interpretation than the evidence supports.

**M2. The GLE-thermostat prior art needs a more concrete mathematical comparison.**  
Location: Introduction, Discussion. Ceriotti–Bussi–Parrinello and the GLE4MD literature already
provide memory kernels, colored noise, and frequency-resolved thermostats. The paper’s distinction
is the object (a readout on a frozen force-free propagator) and the mode-resolved exact rescale,
but the stationary-distribution math is close. A domain reviewer will ask: what is the stationary
distribution of the GLR, and how does it differ from a GLE thermostat with the same memory kernel?
A short analytical paragraph or a comparison of the Fokker–Planck/AR stationary distributions would
strengthen the paper.

**M3. The Bigi–Ceriotti action map is cited but not compared.**  
Location: Introduction, Discussion. The action-map approach is the closest deterministic
alternative: it enforces symplectic/time-reversible structure and improves equipartition. The paper
should explain why a deterministic structure-preserving map does not already solve the NVT mode
partition problem, and whether an action map + GLR would be complementary. If code is available, a
water comparison would be very strong; if not, a careful conceptual comparison is needed.

**M4. The internal ACF long tail remains a physical residual.**  
Location: Section 2.3, Fig. 3c, Section 2.7. AR(10) gives internal ACF 0.312 at 500 fs versus
reference 0.104; AR(20) improves the ACF but not the 100 ps O–H. The paper should discuss whether
this residual is a limitation of the AR basis, the mode decomposition, or the reference statistics,
and whether a longer memory kernel or a different basis (e.g., exponentials/Markovian embedding)
would help.

### MINOR

**m1. SPC/Fw only.** The paper should scope to SPC/Fw and discuss transfer to SPC/E,
TIP4P/2005, and rigid water models.

**m2. The H VDOS caveat** is handled well in the Results; keep it visible in the table.

**m3. The mode decomposition** uses molecular translation/rotation/internal, not normal modes or
phonons. For water this is reasonable; state why this basis is the right one and how it would
generalize to solids.

## Questions for the authors

1. At 100 ps, what are the GLR and published RDF/VDOS values side by side? Can you provide a
   length-matched table?
2. What is the stationary distribution of the GLR, and how does it differ from a GLE thermostat
   with the same memory kernel?
3. Why does the AR(20) long-lag improvement not translate into a better O–H RDF?
4. Would an action map + GLR be complementary, and can you test it on water?

## Recommendation

**Major Revision.** The physics is interesting and the water results are strong, but the
length-mismatched comparison must be fixed, the “thermodynamic consistency” claim must be defined
precisely, and the GLE/action-map prior art needs a sharper comparison.
