# Submission Checklist — NMI Method Paper

**Manuscript:** *Thermodynamically consistent force-free molecular dynamics via a mode-resolved
generalized Langevin readout*
**Target:** Nature Machine Intelligence
**Date:** 2026-11-22

## Manuscript files

| Item | Path | Status |
|---|---|---|
| Clean manuscript PDF | `paper/latex/main_clean.pdf` (17 pages) | ✅ |
| Review PDF with line numbers | `paper/latex/main.pdf` (17 pages) | ✅ |
| LaTeX source | `paper/latex/main.tex`, `sections/`, `refs.bib` | ✅ |
| Figures (vector PDF) | `paper/figures/fig1_diagnostic.pdf`, `fig2_architecture.pdf`, `fig3_water.pdf`, `fig4_ablations.pdf` | ✅ |
| Figure scripts | `paper/figures/fig{1,2,3,4}_*.py` | ✅ |
| Tables | `paper/tables/table1.tex`, `table2_prior_art.tex` | ✅ |
| Bibliography | `paper/latex/refs.bib` (10 cited works) | ✅ |
| AI-use disclosure | `paper/AI_DISCLOSURE_DRAFT.md` + placeholder in `main.tex` | 🟡 author input |
| Author list / affiliations | placeholder in `main.tex` | ⬜ author input |
| Cover letter | not written | ⬜ optional |
| Supplementary Information | outline: `paper/SUPPLEMENTARY_OUTLINE.md` | 🟡 not assembled |

## Figures and tables

| Figure | Content | Data source |
|---|---|---|
| Fig. 1 | pointwise-vs-ensemble diagnostic; NVE; partition heat map; stability | `paper/figures/fig1_diagnostic.py` |
| Fig. 2 | GLR architecture | `paper/figures/fig2_architecture.py` |
| Fig. 3 | water NVT: partition, ACFs, RDFs, VDOS, MSD | `paper/figures/fig3_water.py` |
| Fig. 4 | ablations: AR order, O–H RDF, displacement crash, component bars | `paper/figures/fig4_ablations.py` |
| Table 1 | full method comparison, length-matched 100 ps + 2.5 ns GLR | `paper/tables/table1.tex` |
| Table 2 | prior-art comparison | `paper/tables/table2_prior_art.tex` |

## Data and code

| Item | Path / DOI | Status |
|---|---|---|
| Reference NVT trajectory | `/data1/.../nvt_2p5ns/production.h5` | local |
| Independent NVT trajectory | `experiments/independent_nvt_500ps_20261121/` | local |
| GLR 2.5 ns trajectories | `experiments/glr_2p5ns_20261122/`, `glr_independent_2p5ns_20261122/` | local |
| Ablation trajectories | `experiments/glr_*_100ps_20261122/` | local |
| Published checkpoint | IBM Hugging Face `ibm-research/trajcast.models-arxiv2025` | public |
| Code | `code/h1h2/` (`fit_glr.py`, `glr_readout.py`, `fd_readout.py`, `rollout_relational.py`, analysis) | to package |
| Repository/DOI | [to be supplied] | ⬜ |

## Reproducibility

- Seeds: 2026110401 (val_04), 2026110402 (independent), 2026110403 (seed 3).
- Checkpoint SHA-256: `368697746f4cc48e27ee3953c3b7e253aef0f69abce9e7164e38b2a2cff0e0c1`.
- Reference readout fit: `experiments/glr_fit_20261122/glr_coefficients.json`.
- Held-out block fit: `experiments/glr_fit_heldout0_20261122/glr_coefficients.json`.
- Analysis: `method_development/run_glr_2p5ns_analysis.sh`; ablations:
  `method_development/analyze_glr_ablations.sh`.
- Environment: `/data1/wyd/envs/mlip/bin/python`; PyTorch + e3nn + TrajCast upstream
  `trajcast-c52f107`; LAMMPS runtime for energy verification.
- No reference force is used during propagation.

## Claim boundaries (for the cover letter / abstract)

- **Supported:** mean mode kinetic partition, multi-lag velocity ACF, stable 2.5 ns NVT, RDF/VDOS
  at or beyond the published checkpoint at 100 ps, initial-condition independence.
- **Limited:** diffusion is 26% below the reference on both 2.5 ns starts; the full joint
  position–velocity distribution and mode-KE fluctuations are not guaranteed.
- **Out of scope:** long-range electrostatics, NPT, solids/phonons, non-water liquids, temperature
  transfer, second backbone (not tested).

## Open items before submission

1. Author list, affiliations, corresponding author, CRediT contributions.
2. Code/data repository and DOI; license.
3. AI-use disclosure final text.
4. Supplementary Information assembly (Notes S1–S7, Figs S1–S7, Tables S1–S4).
5. Optional: second backbone/system, published 2.5 ns RDF/VDOS, position-consistent COM head.
6. Final Nature-style reference formatting and figure sizing.
7. Cover letter (optional).
