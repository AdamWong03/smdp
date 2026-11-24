# R8 package checklist — NMI submission

**Status:** in progress, 2026-11-23. This is the submission-critical path; experiments do not block it.

## 1. Code

- [ ] Public repository (GitHub/GitLab) with the GLR readout, rollout, fitting, and analysis code.
- [ ] `environment.yml` / `requirements.txt` with exact versions (Python 3.12, PyTorch, e3nn,
      h5py, numpy/scipy, LAMMPS runtime).
- [ ] Checkpoint hashes for the 250k water, v0 water, and 5 fs quartz backbones.
- [ ] `README` with the three reproduction commands:
      water 100 ps, water 2.5 ns, quartz 100 ps.
- [ ] License (MIT/BSD-3) and citation file.

## 2. Data and artifacts

- [ ] Water reference NVT (2.5 ns), water NVE initial frames, independent NVT initial frame.
- [ ] Quartz reference NVT (15 ns), quartz NVE initial frame, LAMMPS input/potential.
- [ ] Phonon basis: `phonon_basis.npz`, `phonon_basis_metadata.json`, AR(10) diagnostics,
      DHO fit, mode analysis.
- [ ] GLR coefficients: water `glr_coefficients.json`; quartz AR diagnostics.
- [ ] Trajectories for all main-text runs (or reduced source data if size is prohibitive).
- [ ] Zenodo/figshare DOI; data availability statement.

## 3. Source data for figures and tables

- [ ] Fig. 1 diagnostic data.
- [ ] Fig. 2 architecture (schematic; no data).
- [ ] Fig. 3 water panels: partition, ACFs, RDF, VDOS, MSD/diffusion.
- [ ] Fig. 4 ablations: AR order, FD noise, seeds, v0.
- [ ] Table 1 long-time comparison.
- [ ] Table 2 prior-art comparison (no data).
- [ ] Table 3 backbone comparison.
- [ ] Table 4 quartz band/species/deterministic/full-memory rows.
- [ ] CSV/JSON per panel, with README mapping.

## 4. SI sections

- [ ] Methods details: mode decomposition, Yule-Walker, FD noise, blend, rescale, COM projection.
- [ ] AR order ablation (1/5/10/20) and the internal long-tail limitation.
- [ ] FD-noise ablation (negative at \(\lambda=0.1\)).
- [ ] Seed study (three 100 ps seeds) and 2.5 ns starts.
- [ ] v0 phase-space backbone rescue.
- [ ] Quartz 30 fs vs 5 fs, species basis negative, deterministic species control.
- [ ] Phonon-basis audit: finite-difference Hessian, ASR, acoustic/imaginary modes, AR/DHO fits.
- [ ] Full species GLR negative; full phonon GLR negative; per-mode rescale negative.
- [ ] Stochastic water 2.5 ns: canonical KE variance but GK/MSD inconsistency.
- [ ] Diffusion gate: MSD/GK, kernel-order tests, 26% limitation.
- [ ] Compute/cost table.

## 5. Manuscript metadata

- [ ] Authors, affiliations, corresponding author, ORCID.
- [ ] Author contributions, competing interests, acknowledgements, funding.
- [ ] AI/LLM disclosure (writing/analysis assistance; all scientific claims verified by authors).
- [ ] Cover letter: one fixed readout, two materials, positive and negative results.
- [ ] Data/code availability statements.
- [ ] Reporting summary (NMI).

## 6. Reproducibility checks

- [ ] Fresh environment install on a clean machine.
- [ ] Re-run one 1 ps smoke test for water and quartz.
- [ ] Re-run one 100 ps analysis and compare to the archived JSON.
- [ ] Verify checkpoint SHA256 and coefficient hashes.
- [ ] Verify figure scripts regenerate the PDFs from source data.

## 7. Open gaps

- [ ] Water 2.5 ns seeds 2–3 per start (Stage 1).
- [ ] Quartz 2.5 ns seeds (Stage 1).
- [ ] Final PDF/cover letter after Stage 1.
- [ ] DOI minting after the repository is frozen.
