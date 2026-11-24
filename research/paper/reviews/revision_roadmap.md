# Revision Roadmap

**Decision:** Major Revision
**Manuscript:** `paper/latex/main.pdf` (17 pages after Table 2)
**Status key:** ✅ done · 🟡 partial · ⬜ pending

## Priority 1 — Claim calibration (must)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R1.1 | Define “thermodynamically consistent” | abstract, Results 2.2, Discussion | Mean mode partition + reference ACF; exclude full joint distribution, exact transport, KE fluctuations | ✅ |
| R1.2 | State the 26% diffusion limitation in the abstract | abstract | “within 26% of the reference” + “transport limitation” framing | ✅ |
| R1.3 | Avoid “recovers diffusion” language | whole manuscript | Use “within 26%”, “26% low”, “transport limitation” | ✅ |
| R1.4 | Title check | title | Keep title if the abstract definition is visible; add a subtitle if the editor requests | 🟡 |

## Priority 2 — Fair comparison (must)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R2.1 | Length-matched RDF/VDOS | abstract, Table 1, Results 2.3 | 100 ps GLR vs 100 ps published; 2.5 ns GLR as separate long-run result | ✅ |
| R2.2 | Separate 100 ps and 2.5 ns GLR rows | Table 1 | Added 100 ps row with seed D range; 2.5 ns row with D 1.39–1.41 | ✅ |
| R2.3 | Figure caption lengths | Fig. 3 caption | Panels a–g 100 ps; panel h 2.5 ns | ✅ |
| R2.4 | Published 2.5 ns RDF/VDOS (optional) | — | 500 ps published run would strengthen the long-run comparison; our 2.5 ns run was stopped (too slow) | ⬜ optional |

## Priority 3 — Reference-fit and circularity (must)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R3.1 | Held-out block fit | Methods | 0–100 ps vs 500–600 ps fit; coefficient correlation 0.999 | ✅ |
| R3.2 | ACF circularity statement | Results 2.3 | “The ACF match is by construction; RDF/VDOS/diffusion are the independent tests” | ✅ |
| R3.3 | Independent-trajectory fit (optional) | — | Fit on independent NVT and compare; requires converting the LAMMPS dump | ⬜ optional |

## Priority 4 — FD noise and component claims (must)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R4.1 | Report the negative ablation | Results 2.7 | Noise-off: partition/RDF/VDOS/D unchanged | ✅ |
| R4.2 | Remove noise from abstract’s core claims | abstract | Abstract now lists memory kernel + rescale, not noise | ✅ |
| R4.3 | Discussion sentence | Discussion | “Noise retained for theoretical consistency, not required by data” | ✅ |
| R4.4 | Larger-λ noise test (optional) | — | Test whether noise matters at λ=0.3/0.5 | ⬜ optional |

## Priority 5 — Prior art (must)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R5.1 | Prior-art table | Discussion, Table 2 | TrajCast / Bigi–Ceriotti / MAMD / GLE / GLR | ✅ |
| R5.2 | Bigi–Ceriotti PRL citation | refs.bib | PRL 136, 237301 (2026) + DOI | ✅ |
| R5.3 | GLE stationary-distribution comparison | Discussion | Conceptual paragraph; a math comparison is optional | 🟡 |
| R5.4 | Action-map experiment (optional) | — | Compare on water if code available | ⬜ optional |

## Priority 6 — Scope and generality (must/soft)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R6.1 | Soften “general recipe” | Introduction | “practical route, demonstrated here on water” | ✅ |
| R6.2 | Soften “backbone-agnostic” | abstract, Discussion | “architecture-agnostic in principle; one backbone tested” | ✅ |
| R6.3 | Second backbone/system (optional) | — | Native ep145 + GLR or another water model | ⬜ optional |

## Priority 7 — Statistics (must)

| ID | Item | Location | Action | Status |
|---|---|---|---|---|
| R7.1 | Seed-to-seed D | Results 2.7 | 100 ps seeds 1.20/1.21/1.45; 2.5 ns 1.39–1.41 | ✅ |
| R7.2 | Block uncertainty | Methods/Table 1 | Block std and \(R^2\) reported | ✅ |
| R7.3 | Second 2.5 ns seed (optional) | — | Same-start seed replication | ⬜ optional |

## Priority 8 — Submission package (must)

| ID | Item | Action | Status |
|---|---|---|---|
| R8.1 | Author list/affiliations | User input | ⬜ |
| R8.2 | Code repository/DOI | Package `code/h1h2`, fit/rollout/analysis scripts | ⬜ |
| R8.3 | Data availability | Reference/rollout trajectories, published checkpoint link | ⬜ |
| R8.4 | Supplementary Information | Notes S1–S7, Figs S1–S7, Tables S1–S4 (outline exists) | 🟡 |
| R8.5 | AI-use disclosure | Run `academic-paper` `disclosure` mode for NMI | ⬜ |
| R8.6 | Cover letter | Optional; not written | ⬜ |

## Priority 9 — Minor (nice to have)

| ID | Item | Status |
|---|---|---|
| R9.1 | H VDOS caveat in table | ✅ |
| R9.2 | Native ep145 marked diagnostic | ✅ |
| R9.3 | 15× ratio confidence interval | ⬜ optional |
| R9.4 | Internal ACF 500 fs discussion | ✅ |
| R9.5 | Remove line numbers for final PDF | ✅ `main_clean.pdf` |

## Current assessment after this revision round

- **Scientific core:** strong diagnostic + strong water structure/spectrum; diffusion 26% low,
  honestly reported.
- **Claim calibration:** much improved; the paper no longer overclaims exact diffusion or full
  thermodynamic consistency.
- **Remaining blockers:** author list, code/data/SI, AI disclosure, and a final title/abstract
  check. The optional experiments (second backbone/system, published 2.5 ns RDF/VDOS,
  position-consistent COM head) would strengthen the paper but are not required for the
  water-scoped claim.
