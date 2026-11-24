# Paper Configuration Record

**Status:** confirmed by the author, 2026-11-22
**Skill mode:** `academic-paper` `plan` now → `full` after the diffusion gate

| Item | Value |
|---|---|
| Paper type | Original method paper |
| Discipline | Machine learning for molecular simulation / computational chemistry |
| Target journal | **Nature Machine Intelligence** |
| Positioning | **L1**: one fixed GLR readout, per-system frozen backbones, water + quartz, two water backbones. Not a foundation model (L3). |
| NMI reference | TrajCast, NMI 8, 764–776 (2026): structure/benchmark alignment, not copied. |
| Citation style | Nature (numbered, superscript) |
| Output format | LaTeX + PDF (DOCX optional) |
| Manuscript language | English |
| Main text | ~4,000–4,500 words |
| Figures | 4–5 main figures |
| Methods | ~2,000 words |
| Working title | *Stationary-measure correction for force-free molecular dynamics via a mode-resolved generalized Langevin readout* |
| Authors / affiliations | to be supplied by the author |
| Data/code availability | code + trajectories; statement to be drafted |
| AI disclosure | use `academic-paper` `disclosure` mode before submission |

## Positioning (provisional until the diffusion gate)

> A frozen force-free propagator learns the structure; a mode-resolved generalized-Langevin
> readout enforces the canonical partition and the multi-lag dynamics by construction.

Novelty to verify in Phase 1:

- mode-resolved AR(p) memory kernel as a readout on a learned propagator;
- FD-consistent noise + per-mode variance rescaling;
- decoupling structure (backbone displacement) from thermodynamics (readout);
- diagnostic: pointwise one-step accuracy is neither necessary nor sufficient for long-time
  ensemble fidelity.

## Gate

- GLR 2.5 ns diffusion ≈ 1.9e-9 m²/s, independent start, seeds, ablations.
- If the gate fails, iterate the GLR (higher p / internal memory) before Phase 4 drafting.

## Current status (2026-11-22)

- Phase 0 config: confirmed.
- Phase 1 novelty matrix: complete.
- Phase 2 outline + evidence map: complete.
- Phase 3 argument blueprint: complete.
- Phase 4 draft: complete LaTeX draft (Introduction, Results, Discussion, Methods), 4 figures,
  Table 1, 16-page PDF. All numerical placeholders filled.
- Figures: Fig. 1 diagnostic, Fig. 2 architecture, Fig. 3 water (both 2.5 ns starts), Fig. 4
  ablations. Fig. 5 optional; four main figures are sufficient.
- Gate: **structure/thermodynamics pass on two 2.5 ns starts; diffusion is 26% low** (1.39–1.41
  vs 1.88e-9). The simple fixes (mixed p20 translation, blend, translation displacement, COM
  scaling) do not close the gap. The paper locks the **mode-resolved thermodynamic consistency and
  structure** claim and reports diffusion as a quantified limitation; it does not claim exact
  diffusion recovery.
- Phase 5 citation check: complete; TrajCast DOI/pages and Bigi–Ceriotti PRL metadata corrected.
- Phase 6 mock review: complete (`paper/reviews/`), decision **Major Revision**; the required
  textual revisions (claim calibration, length-matched comparison, held-out fit, FD-noise
  reframing, prior-art table, seed D range) are implemented. Remaining: author list, code/data,
  SI, AI disclosure, optional second backbone/system.
- Phase 7 format: clean PDF `paper/latex/main_clean.pdf` (17 pages); review PDF
  `paper/latex/main.pdf` with line numbers; submission checklist in `paper/SUBMISSION_CHECKLIST.md`.
