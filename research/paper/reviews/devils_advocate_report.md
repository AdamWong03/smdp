# Devil’s Advocate Report

**Recommendation:** Major Revision (two CRITICAL issues; both addressable, neither fatal)

## Strongest counter-argument

The paper’s headline claim—“thermodynamically consistent force-free molecular dynamics”—is
stronger than the evidence. The readout fixes the **mean** mode kinetic energy, but it suppresses
the mode-KE fluctuations, does not guarantee the position marginal, and leaves the diffusion
coefficient 26% low on both 2.5 ns starts. The position update is deliberately decoupled from the
velocity readout; when the authors try to make it consistent, the full correction crashes and the
partial/COM variants either fail to close the gap or break the MSD–Green–Kubo consistency. So the
one thing the title promises—a thermodynamically consistent dynamics—is precisely the thing the
architecture does not deliver. The paper is honest about this in the Results and Discussion, but
the abstract still reads as a success story, and the headline RDF/VDOS comparison is not
length-matched: the GLR values are from 2.5 ns, the published values from 100 ps. At 100 ps the
O–H improvement is 0.104 vs 0.118, not 0.038 vs 0.118. The diagnostic is genuinely useful, but the
method may be a mode-resolved deterministic AR filter plus a variance rescale, with the GLE noise
playing no empirical role; the novelty relative to GLE thermostats, Bigi–Ceriotti’s action map, and
MAMD is therefore narrower than the paper suggests.

## Issue list

### CRITICAL

**DA-C1. Claim–evidence mismatch: “thermodynamically consistent” is not defined falsifiably, and
the evidence contradicts the strong reading.**  
Dimension: argument validity / claim calibration.  
Location: title, abstract, Section 2.2, Section 2.3, Discussion.  
Evidence: D = 1.39–1.41 vs reference 1.88 (26% low); mode-KE std ≈0.002 vs reference ≈0.24 eV;
position update decoupled from the velocity readout.  
Required fix: define “thermodynamically consistent” in the abstract as **mean mode partition +
reference velocity ACF**, explicitly exclude exact transport and full KE fluctuations, and either
narrow the title or add “transport limitation” to it. A reader must be able to say which
observable would falsify the claim.

**DA-C2. The headline RDF/VDOS comparison is not length-matched.**  
Dimension: evidence fairness.  
Location: abstract, Table 1, Section 2.3, Fig. 3.  
Evidence: GLR RDF/VDOS from 2.5 ns; published checkpoint from 100 ps. At 100 ps, GLR O–H is 0.104
vs published 0.118, not 0.038 vs 0.118.  
Required fix: compare at the same length (100 ps vs 100 ps), run the published checkpoint for
2.5 ns, or report the 2.5 ns GLR as a separate long-run convergence result and do not present it
as a head-to-head. This is a fairness issue, not a physics issue, but it must be fixed.

### MAJOR

**DA-M1. The diagnostic is an existence proof from two confounded models, not a controlled
relation.**  
Location: Section 2.1, Fig. 1. The published checkpoint and the 250k model differ in architecture,
training data, objective, and pointwise error. The inversion is striking, but it does not establish
that pointwise accuracy is generally uninformative; it establishes that it is neither necessary nor
sufficient in this pair. The paper says this, but the Introduction’s “the ranking can be inverted”
could be read as a general claim. Add a controlled sweep or state the existence-proof scope more
prominently.

**DA-M2. The FD noise is not necessary at λ = 0.1, so the “generalized Langevin” framing is
partly decorative.**  
Location: Section 2.7, abstract, Introduction. The noise-off run gives the same partition, RDF,
VDOS, and a slightly higher D. The paper reports this honestly, but the abstract still lists
“fluctuation–dissipation-consistent noise” as a core component. Either show a regime where the
noise matters or reframe the method as an AR(\(p\)) memory readout with optional FD noise.

**DA-M3. Single water model, single backbone; “backbone-agnostic” and “general recipe” are
unsupported.**  
Location: abstract, Introduction, Discussion. Add a second backbone/system or soften the claim to
“demonstrated on SPC/Fw water with a frozen relational backbone.”

**DA-M4. The reference ACF is used both to fit and to validate the dynamics.**  
Location: Methods, Section 2.3. The ACF match is by construction, not an independent test. The
RDF/VDOS/diffusion are independent, but the paper should state the circularity explicitly and add
a held-out reference block/independent trajectory for the fit.

### MINOR

**DA-m1.** The H VDOS comparison is caveated; keep the caveat in the abstract/table.
**DA-m2.** The internal ACF 500 fs tail is a residual; state whether it matters for any observable.
**DA-m3.** Code/data/SI are missing; NMI will require them.

## Ignored alternative explanations / paths

1. **The structural gains may be due to the longer 2.5 ns sampling, not the readout.** The 100 ps
   GLR O–H is 0.104; the 2.5 ns is 0.038. The length-matched comparison is less dramatic. This is
   partially addressed by the 100 ps published comparison, but the abstract uses the 2.5 ns value.
2. **The variance rescale alone** could be responsible for the partition; FDR v1 shows that the
   rescale alone does not fix O–H (0.301), so the AR memory is doing real work. This alternative is
   largely ruled out, but the paper should say so explicitly.
3. **A position-consistent backbone fine-tune** could fix D; the paper tests displacement
   corrections but not a learned COM head. This is the obvious next path and should be discussed.
4. **The diagnostic could be a checkpoint artifact.** The published checkpoint may be unusually
   good or bad; a second published model would strengthen the claim.

## Missing stakeholder perspectives

- Users without a 100 ps reference NVT trajectory (the method’s data requirement).
- Researchers who need transport coefficients (D is 26% low).
- Solids/phonons, long-range electrostatics, and non-water liquids (outside scope but implied by
  the broad title).
- Groups with limited compute (the reference MD + backbone training cost).

## Observations (non-defects)

- The negative results (rescale-off, displacement-correction crash, FD-noise insensitivity) are a
  strength; they make the paper more credible.
- The writing is clear and the figures are well designed.
- The two-start 2.5 ns agreement is convincing for the stationary measure.
