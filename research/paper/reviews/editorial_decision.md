# Editorial Decision

**Manuscript:** *Thermodynamically consistent force-free molecular dynamics via a mode-resolved
generalized Langevin readout*
**Decision:** **Major Revision**

## Consensus

All five seats agree on the following:

1. **The diagnostic is a genuine contribution.** The pointwise-vs-ensemble inversion, the NVE red
   herring, and the mode-partition heat map are clear, surprising, and useful. This is the
   strongest part of the paper.
2. **The structural results are strong and reproducible.** Two 2.5 ns starts give nearly identical
   partition, RDF, and VDOS; the O–H RDF 0.038/0.044 and O VDOS 0.953 are meaningful improvements
   over the published checkpoint at 100 ps (0.104 vs 0.118 O–H; 0.964 vs 0.937 O VDOS).
3. **The diffusion deficit is the central limitation.** \(D=1.39\)–\(1.41\times10^{-9}\) versus
   \(1.88\times10^{-9}\) on both starts. The paper reports it honestly, but the title/abstract
   still need to be calibrated to it.
4. **The FD-noise ablation is a negative result.** At \(\lambda=0.1\), removing the noise does not
   change the partition, RDF, VDOS, or \(D\). The memory kernel and the variance rescale are the
   load-bearing components.
5. **The single-system/single-backbone scope needs to be explicit.** The paper demonstrates water
   with one frozen backbone; “general recipe” and “backbone-agnostic” should be softened or
   supported by a second backbone.

## Disagreement and adjudication

### DA-C1 — “Thermodynamically consistent” is overclaimed

**Status: VALIDATED.** The strong reading (full joint canonical distribution, exact transport) is
not supported. The paper now defines “thermodynamically consistent” as the **mean mode kinetic
partition plus the reference velocity ACF**, states that the full joint distribution and exact
transport are not guaranteed, and reports the 26% diffusion deficit as a limitation. This is
sufficient for revision; the title may remain if the definition stays visible in the abstract.
The authors should consider a subtitle in the final version if the editor requests it.

### DA-C2 — The headline RDF/VDOS comparison is length-mismatched

**Status: VALIDATED.** The GLR 2.5 ns values were compared with the published 100 ps values. The
revision now uses a length-matched 100 ps comparison in the abstract and adds a separate 2.5 ns
GLR row to Table 1; the 2.5 ns O–H 0.038 is reported as a long-run convergence result, not as a
head-to-head with the 100 ps published checkpoint. This resolves the fairness issue.

### DA-M1 — The diagnostic is an existence proof from two confounded models

**Status: PARTIALLY VALIDATED.** The paper now states explicitly that the two checkpoints differ
in architecture and training data, so the inversion is an existence proof (“neither necessary nor
sufficient”), not a controlled anti-correlation. A controlled sweep is suggested but not required
for the revised claim.

### DA-M2 — The FD noise is not necessary

**Status: VALIDATED.** The abstract no longer lists the FD noise as a core component; the Results
and Discussion state the negative ablation and identify the memory kernel and rescale as the
load-bearing components. The paper should not claim the noise is necessary.

### DA-M3 — Single system / single backbone

**Status: VALIDATED.** The text now says “demonstrated here on water” and “architecture-agnostic
in principle, but only one backbone is tested.” A second backbone/system is a suggested revision,
not a requirement for the water-scoped claim.

### DA-M4 — Reference ACF is used for fit and validation

**Status: PARTIALLY VALIDATED.** A block-sensitivity check (0–100 ps vs 500–600 ps fit; coefficient
correlation 0.999) is now in Methods, and the Results state that the ACF match is by construction,
not an independent validation. The RDF/VDOS/diffusion are the independent tests.

## Required revisions

| ID | Revision | Status after this round |
|---|---|---|
| R1 | Define “thermodynamically consistent” precisely; state mean-partition/ACF scope and the transport limitation in the abstract. | **Done** |
| R2 | Make the RDF/VDOS comparison length-matched; separate the 2.5 ns GLR convergence result. | **Done** |
| R3 | Add a held-out reference-fit check and state the ACF-fit circularity. | **Done** |
| R4 | Reframe the FD noise: report the negative ablation, remove it from the abstract’s core claims. | **Done** |
| R5 | Add a precise prior-art comparison (GLE thermostats, Bigi–Ceriotti, MAMD). | **Done** (Table 2 + Discussion) |
| R6 | Soften “general recipe/backbone-agnostic” or add a second backbone/system. | **Partially done** (text scoped; experiment optional) |
| R7 | Report seed-to-seed \(D\) and the diffusion uncertainty. | **Done** (100 ps seed range 1.20–1.45; 2.5 ns 1.39–1.41) |
| R8 | Provide code/data/SI, author list, and AI-use disclosure. | **Pending (author input)** |
| R9 | Minor: H VDOS caveat, native ep145 label, 15× CI, internal ACF discussion. | **Mostly done** |

## Suggested revisions (not required for the water-scoped claim)

1. A second backbone (e.g., native ep145 or a small MLP) to test architecture-agnosticism.
2. A second system (SPC/E, TIP4P/2005, or a simple liquid) to test transfer.
3. A published-checkpoint 2.5 ns RDF/VDOS run (we stopped ours at 3k/510k because it was too slow;
   a 500 ps run would already improve the comparison).
4. A direct action-map (Bigi–Ceriotti) comparison on water, if code is available.
5. A stochastic rescale or a position-consistent COM head to fix the 26% diffusion deficit.

## Recommendation

**Major Revision.** The paper has a strong diagnostic and strong water structure/spectrum results.
The revision path is primarily claim calibration, length-matched comparisons, and explicit
scoping; the experimental additions (second backbone/system, position-consistent diffusion) would
strengthen the paper but are not strictly required for the water-scoped claim. With R1–R8
addressed, the paper is a credible NMI method paper.

## Post-review addendum (2026-11-22)

After the review, we added a **second-backbone test**: the same GLR coefficients were applied to
the v0 phase-space operator (3.04M parameters), a different architecture that fails NVT much more
strongly than the 250k model. Without the readout: partition 0.111/0.808/6.338, RDF
0.944/1.409/0.229, H/O VDOS 0.356/0.236. With the GLR: T=300.0 K, partition
2.482/2.529/2.396, RDF 0.042/0.109/0.019, H/O VDOS 0.855/0.969; the independent start gives
2.482/2.527/2.398, RDF 0.050/0.131/0.023, VDOS 0.857/0.977. **The readout rescues a completely
different backbone**, which directly addresses the “backbone-agnostic” concern (DA-M3/EIC M1) and
strengthens the general-framework claim.

We also tested a **quartz species-resolved basis** at the available 30 fs horizon. The 30 fs ACF
is nearly white (prediction error 0.91), and the full species GLR degrades the phonon VDOS (0.48 vs
0.82 baseline) even at λ=0.02. However, the **zero-memory limit** (species variance rescale only)
enforces exact Si/O equipartition at 301.9 K, preserves the phonon VDOS (0.838/0.835), and
slightly improves the Si–O RDF (0.149 vs 0.175 baseline). This is a positive cross-material result
for the variance-rescale component, with the memory kernel limited by the coarse horizon. A 5 fs
quartz backbone is future work.

**Effect on the decision:** the backbone-agnostic claim is now empirically supported, and the
paper is stronger. The remaining major gap is a second **system** (not a second backbone) and the
26% diffusion deficit. The decision remains **Major Revision**, but the path is now:
(i) keep the two-backbone result, (ii) either add a second molecular system (paracetamol/SPC/E) or
scope explicitly to water, (iii) keep the diffusion limitation. With these, the paper is a strong
NMI candidate.
