# Running Jobs — 2026-11-20

**Working root:** `/data2/wyd/research/foundation_mlip_adaptation`
**Data/code root:** `/data1/wyd/research/foundation_mlip_adaptation` (read-only)

---

## 0. Completed campaign — 2026-11-23

| Job | GPU | Status | Note |
|---|---|---|---|
| `glr_stoch_2p5ns_20261122` | 3 | completed | val_04: T 299.97, KE std 0.437, MSD D 1.435, GK 1.21 |
| `glr_stoch_independent_2p5ns_20261122` | 0 | completed | independent: T 300.08, KE std 0.429, MSD D 1.387, GK 1.16 |
| `quartz5_phonon_bands_rescale_seed2_100ps_20261123` | 1 | completed | second seed: T 299.91, KE std 0.3991, VDOS 0.947/0.962 |
| `quartz5_phonon_bands_rescale_only_100ps_20261123` | 1 | completed | **positive**: T 299.8, KE std 0.3967, VDOS 0.953/0.949 |
| `quartz5_phonon_species_rescale_only_100ps_20261123` | 1 | completed | species control: VDOS 0.898/0.955 |
| `quartz5_phonon_species_100ps_20261123` | 1 | completed | full phonon memory: negative, VDOS 0.734/0.838 |
| `quartz5_phonon_norescale_100ps_20261123` | 1 | completed | full phonon memory only: negative, VDOS 0.594/0.686 |
| `quartz5_phonon_glr_100ps_20261123` | 1 | completed | per-mode rescale + memory: negative, VDOS 0.209/0.253 |
| `quartz5_phonon_rescale_100ps_20261123` | 1 | completed | per-mode rescale only: negative, VDOS 0.217/0.269 |
| `v0_stoch_*_100ps_20261123` | 1 | aborted | not needed; water production stays deterministic |
| `dho_fit.py` | CPU | completed | DHO linewidth fit; log `quartz_phonon/dho_fit.log` |

**Outcome:** water production stays deterministic because the stochastic 2.5 ns variant, while
restoring the KE variance (0.437/0.429 versus 0.440 eV), gives GK 1.16–1.21e-9 inconsistent with
MSD 1.39–1.44e-9. Quartz uses the phonon-band stochastic rescale (two seeds, VDOS 0.950/0.955).
Phonon basis and coefficients: `/home/wyd/quartz_phonon/phonon_basis.npz`,
`ar10_diagnostics.npz`. Decision memo: `paper/PHONON_GLR_DECISION.md`.

---

## 1. H1 selection screening — completed (null at 20k)

| Arm | Selection | Selected epoch | Test position RMSE | Test velocity RMSE |
|---|---|---:|---:|---:|
| A | endpoint only | 32 | 0.0013679 Å | 0.0004874 Å/fs |
| B | endpoint + 500 fs rollout | 32 | 0.0013920 Å | 0.0004899 Å/fs |

Both arms selected the final epoch. The 0.001 rollout weight was too small; best rollout epoch
was 27, but the combined score still chose epoch 32. The rollout proxy was noisy at 4 × 100
steps. **H1 is a null at 20k; do not scale blindly.** See `H1_SELECTION_PROXY.md` §6.

---

## 2. Phase-space v0 Water NVT 2.5 ns — completed, negative

| Item | Value |
|---|---|
| Checkpoint | `data1/.../experiments/phase_space_20261120/water5/best.pt` (epoch 398) |
| Protocol | 50,000 thermalization + 500,000 production steps, CSVR 300 K, τ=50 fs |
| Output | `data2/.../experiments/phase_space_20261120/water5_nvt_replica_01/trajectory.h5` |
| Analysis | `.../analysis/` |
| Result | stable, 16.5 ns/day; **physical failure**: translation 0.111 vs 2.459, internal 6.338 vs 2.450, VDOS 0.356/0.236, diffusion 2.81e-13 vs 1.88e-9 |

The 20 ps NVE gain did not survive NVT. The bias is already visible in the first 100 ps of
NVT, so use a **100 ps NVT screen** for method iteration.

---

## 3. E1 motion-balanced loss — completed, negative

| Arm | Endpoint pos RMSE | 20 ps NVE translation | rotation | internal | total |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.001396 Å | 2.421 | 1.820 | 2.873 | 7.114 eV |
| Motion | 0.001263 Å | 2.689 | 2.081 | 5.396 | 10.166 eV |

- Motion improves position RMSE but heats to ~400–420 K and is worse on internal/RDF/VDOS.
- Likely confounds: proposal loss inflation (0.387 vs 0.017) and equal mode weights.
- Do not scale E1. See `E1_MOTION_LOSS_RESULT.md`.
- Unexpected signal: 20k baseline is physically better than 250k Full on this NVE start;
  controlled training-time sweep is now the priority.

## 3b. H-C controlled 20k prefix — completed, unstable

| Item | Value |
|---|---|
| Run | `experiments/hc_20k_seed20261103_20261120` |
| Seed | `20261103` (same as 250k Full) |
| Budget | 32 epochs × 625 = 20,000 steps; same 250k warmup-cosine schedule prefix |
| Endpoint | 0.001163 Å / 0.000401 Å/fs |
| NVT screen | **unstable** at ~50 ps production: min 0.852 Å, preT 3338 K |
| Conclusion | early stopping/20k prefix is not a valid method; 20 ps NVE was misleading again |

The training-time/early-stopping line is not supported. The immediate control is Current-Only
250k 100 ps NVT.

## 3c. Current-Only 250k 100 ps NVT screen — completed, negative

| Item | Value |
|---|---|
| Run | `experiments/nvt_screen_current_only_250k_20261120` |
| Result | stable, but translation 0.646 vs 2.373; rotation 0.535 vs 2.430; internal 6.384 vs 2.609; H/O VDOS 0.348/0.600 |
| Conclusion | removing future conditioning does **not** fix NVT; the shared pipeline is now the suspect |

Next: `method_development/ROOT_CAUSE_AUDIT_PLAN.md`.

## 3d. Native TrajCast 100 ps NVT control — completed, diagnostic

| Item | Value |
|---|---|
| Run | `experiments/nvt_screen_native_trajcast_20261120` |
| Checkpoint | native validation-best epoch 145 (not a completed native protocol) |
| Result | stable; translation 3.256, rotation 3.105, internal 1.041; O–O RDF 0.180; H/O VDOS 0.683/0.905 |
| Conclusion | **opposite bias** to our models; incomplete native protocol, diagnostic only |

This is not a SOTA claim. It shows the task is not forcing one failure direction;
architecture/training controls the partition bias direction.

## 3e. Mode-aware readout fine-tune — completed, null

| Item | Value |
|---|---|
| Run | `experiments/mode_aware_finetune_20k_20261121` |
| Base | 250k Full 6 Å (`dsrt_radius6/best.pt`) |
| Correction RMS | 9.60e-6 (0.08% of velocity RMS) |
| Endpoint | 0.000412 Å / 0.000150 Å/fs (unchanged) |
| NVT screen | not run; correction too small to change partition |

Endpoint loss cannot train the correction. Next: rollout/partition-aware training, or a
backbone change. See `MODE_AWARE_READOUT.md`.

## 3f. Rollout/partition-aware fine-tune — completed, null

| Item | Value |
|---|---|
| Run | `experiments/rollout_physical_finetune_5k_20261121` |
| Base | 250k Full 6 Å (frozen) |
| Correction RMS | 9.61e-6 (0.08% of velocity) |
| Validation partition | ~1e-4 at epoch 1, flat |
| Conclusion | 2-step partition is already good; short-horizon loss cannot see the NVT failure |

Next: long-horizon/on-policy correction, or a structural/backbone change. See
`MODE_AWARE_READOUT.md` §9.

## 3g. Mode-gain diagnosis — completed (2026-11-21)
| Item | Value |
|---|---|
| Scripts | `code/h1h2/diagnose_mode_gain.py`, `diagnose_one_step_gain.py` |
| Outputs | `method_development/mode_gain/*.json` |
| Trajectories | existing 100 ps NVT screens; reference 100 ps block |
| Result | total KE correct, partition wrong; internal ACF 0.72 at 500 fs vs 0.10 |
| One-step on data | mode gain correct to ~1e-4; residual ~1e-4 of variance |
| On-policy pilot | 160 autonomous states, SPC/Fw one-step labels; gain still ~1e-3, signed internal-KE surplus +0.0062 eV/step |
| Conclusion | closed-loop distribution shift with a slow signed mode-KE bias |
| Write-up | `LONG_TIME_TRAINING_LITERATURE.md`, GAP_LOG G14 |

No training job is running. The next method step (on-policy correction) is blocked on an
explicit research-lead decision because it would use SPC/Fw one-step labels for training.

## 3h. Published TrajCast water control — completed, PASS

| Item | Value |
|---|---|
| Checkpoint | IBM Hugging Face `water/state_dict_e3nn.pt` (2,258,856 params) |
| Screen | `experiments/nvt_screen_published_trajcast_20261121` |
| Protocol | same as our native: 50 ps thermalization + 100 ps production, val_04 frame 0 |
| Stability | 30,000/30,000 steps, no safety stop |
| Partition | trans 2.505, rot 2.463, int 2.444 eV (total 7.411, T=300.2 K) |
| ACF | trans(25 fs) 0.662 vs ref 0.658; rot(25 fs) −0.539 vs −0.540 |
| O–O RDF RMSE | 0.043 (native 0.180) |
| Pointwise on own NVT states | published pos 6.9e-3 Å, vel 2.5e-3 Å/fs, signed KE +0.083 eV/step |
| 250k pointwise on own states | pos 4.9e-4 Å, vel 1.7e-4 Å/fs, +0.0062 eV/step; NVT fails |
| Conclusion | **pointwise accuracy is neither necessary nor sufficient; ensemble fidelity is the objective** |

The published model has a ~15× larger pointwise error than the 250k yet the correct ensemble.
It behaves generatively (different trajectory, same distribution). The failure is a
distribution/training-objective problem, not a one-step accuracy or on-policy problem.

Next: **M0 ensemble-matched training** — fine-tune 250k Full on 5 fs pairs from the reference
NVT trajectory (offline, no new labels), then 100 ps NVT.

## 3i. M0 ensemble-matched fine-tune — completed, negative

| Item | Value |
|---|---|
| Run | `experiments/m0_nvt_finetune_5k_20261121` |
| Data | 5 fs pairs from reference NVT `production.h5`: 20k train / 5k val / 5k test |
| Base | 250k Full `dsrt_radius6/best.pt` |
| Steps | 5,000; batch 8; lr 3e-5; 10k train / 2k val |
| Val endpoint | 6.4e-4 (epoch 1) → 2.06e-4 (epoch 4); best = last |
| 100 ps screen | stable; **partition 3.37 / 3.27 / 1.00 eV** (ref 2.37/2.43/2.61) |
| Base 250k | 0.54 / 0.79 / 6.04 eV |
| Verdict | **negative**; the bias flipped sign and overshot; total 7.69 eV / 311.6 K |

The model fitted the NVT pairs but its autonomous rollout did not reach equipartition. One-step
endpoint MSE on NVT pairs is not sufficient to control the stationary measure.

## 3j. Independent NVT trajectory — completed

| Item | Value |
|---|---|
| Run | `experiments/independent_nvt_500ps_20261121` |
| Purpose | break same-trajectory leakage for M0 validation |
| Protocol | SPC/Fw, 300 K, CSVR τ=50 fs, seed 26104102, 100 ps thermalization + 500 ps production |
| Output | 100,001 frames at 5 fs; first-frame HDF5 `initial.h5` (301.8 K) |
| Status | completed (21:39 wall time) |

## 3k. M0 independent-start screen — completed, failed

| Item | Value |
|---|---|
| Run | `experiments/nvt_screen_m0_independent_20261121` |
| Model | M0 5k `best.pt` |
| Initial | independent NVT frame 0 (`initial.h5`), 301.8 K |
| Result | **failed at 130 ps**: min 0.965 Å, pre-thermostat T 3798.7 K |
| Valid production | trans 3.41, rot 3.21, int 1.01 eV; T 313.8 K (same bias as val_04 start) |
| Verdict | bias is robust to the initial condition, but the model is **unstable**; M0 endpoint NVT training is not viable |

## 3l. M0 NVE 25 ps diagnostic — running

| Item | Value |
|---|---|
| Run | `experiments/nve_m0_25ps_20261121` |
| Model | M0 5k `best.pt` |
| Initial | independent NVT frame 0 (301.8 K) |
| Purpose | double-thermostat / energy-injection diagnostic: does the model heat without CSVR? |

## 3m. Composed CSVR rollout — negative (failed at 4.3 ps)

| Item | Value |
|---|---|
| Checkpoint | `experiments/csrv_rollout_finetune_500_20261121/best.pt` (step 100) |
| Training signal | val moment 3.25 → **0.028**; endpoint 7.8e-5 → 0.041 |
| Screen | `experiments/nvt_screen_csrv_full_s100_20261121` |
| Result | **failed at 855 steps (4.3 ps)**: min 0.647 Å, preT 1473 K |
| Verdict | short-rollout moment loss does not constrain stability; endpoint collapse signaled leaving the stable basin |

Next: trust-region/stability-constrained ensemble training or architecture change. See
`method_development/CSRV_ROLLOUT_20261121.md`.

## 3n. FD readout — 100 ps positive

| Item | Value |
|---|---|
| Readout | mode-resolved AR(1) + FD noise, blend λ=0.1, per-mode variance rescale |
| Backbone | 250k Full (frozen) |
| 100 ps | **stable; T = 300.0 K; partition 2.482 / 2.533 / 2.392 eV** |
| O–O / O–H / H–H RDF | **0.062 / 0.301 / 0.110** |
| H / O VDOS | **0.807 / 0.851** |
| Base | 0.646 / 1.312 / 0.249; VDOS 0.406 / 0.487 |
| Published | 0.043 / 0.118 / 0.096; VDOS 0.624 / 0.937 |
| Verdict | **first viable method lead**; O–H RDF is the remaining gap (position head) |

Next: independent-start 100 ps; then 2.5 ns and diffusion. See
`method_development/FD_READOUT_20261121.md`.

## 3o. FDR independent-start 100 ps — completed, same stationary measure

| Item | Value |
|---|---|
| Run | `experiments/nvt_screen_fd_rescale_independent_100ps_20261121` |
| Initial | independent NVT frame 0 (301.8 K) |
| 100 ps | **stable; T = 300.0 K; partition 2.445 / 2.496 / 2.355 eV** |
| val_04 FDR | 2.482 / 2.533 / 2.392 eV |
| ACF | translation 0.7345, rotation −0.4314, internal −0.8788 (same as val_04) |
| Verdict | **the FDR stationary measure is independent of the initial condition**; RDF/VDOS running |

## 3p. FDR 2.5 ns diffusion — completed, analysis running

| Item | Value |
|---|---|
| Run | `experiments/nvt_screen_fd_rescale_2p5ns_20261121` |
| Length | 500,000 steps = 2.5 ns; stable; T=300.0 K |
| O–O / O–H / H–H RDF | **0.0373** / 0.4095 / 0.1125 |
| H / O VDOS | 0.797 / 0.851 |
| COM bug | pre-fix global drift 1740 Å; fixed by projecting v and dq to zero net momentum |
| Diffusion | `analysis_diffusion` (global COM removed by the task) |

## 3q. FDR internal-blend 1.0 — 100 ps completed, negative

| Item | Value |
|---|---|
| Run | `experiments/nvt_screen_fd_int10_100ps_20261121` |
| 25 ps signal | internal ACF(5 fs) −0.558 vs ref −0.567; partition 2.445/2.451/2.400; stable |
| 100 ps RDF | O–O **0.187**, O–H **0.957**, H–H 0.074 |
| 100 ps VDOS | H **0.618**, O **0.613** |
| v1 (int 0.1) | O–O 0.062, O–H 0.301, H–H 0.110, VDOS 0.807 / 0.851 |
| Verdict | **worse**; the single-gain AR(1) has the wrong long-lag spectrum |

The internal 5 fs ACF match is not sufficient. The O–H gap needs a memory-kernel/higher-order
AR readout, not a higher blend.

## 3r. GLR AR(10) — 100 ps positive

| Item | Value |
|---|---|
| Fit | `experiments/glr_fit_20261122/glr_coefficients.json` (AR(10), Yule–Walker) |
| 100 ps | stable; T=300.0; partition 2.445/2.490/2.362 |
| Translation ACF | 0.981/0.640/0.014/**−0.030** at 5/25/100/500 fs (ref 0.982/0.658/0.039/−0.019) |
| O–O / O–H / H–H RDF | **0.0322 / 0.1043 / 0.0256** |
| H / O VDOS | **0.8484 / 0.9637** |
| Verdict | **all structural/spectral metrics beat published**; diffusion pending |

## 3s. GLR 2.5 ns — completed

| Item | Value |
|---|---|
| Run | `experiments/glr_2p5ns_20261122` |
| Protocol | GLR AR(10) + blend 0.1 + rescale, 500k steps |
| Result | stable; T=299.99 K; partition 2.482/2.526/2.399; RDF 0.031/0.038/0.028; VDOS 0.847/0.953; **D 1.406e-9 (25% low)** |
| Verdict | structure/thermodynamics pass; diffusion is a quantified limitation |

## 3t. GLR independent-start 100 ps — completed, independent

| Item | Value |
|---|---|
| Run | `experiments/glr_independent_100ps_20261122` |
| Initial | independent NVT frame 0 (301.8 K) |
| Result | stable; T=300.0; partition 2.445/2.490/2.362 |
| ACF | translation 0.981/0.642/0.011/−0.024; rotation −0.607; internal −0.577/0.307 — **identical to val_04** |
| RDF | O–O **0.0358**, O–H **0.0981**, H–H **0.0268** |
| VDOS | H **0.8506**, O **0.9628** |
| Verdict | **initial-condition independence confirmed for partition/ACF/RDF/VDOS** |

## 3u. GLR order fits p=1/5/20 — completed

| Order | trans pred | rot pred | int pred | Internal AR100 / 200 fs | Reference |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0365 | 0.2970 | 0.6785 | 0.103 / −0.019 | 0.147 / 0.072 |
| 5 | 0.0047 | 0.0597 | 0.1406 | 0.031 / 0.013 | 0.147 / 0.072 |
| 10 | 0.0042 | 0.0576 | 0.1146 | 0.197 / 0.103 | 0.147 / 0.072 |
| **20** | **0.0039** | **0.0566** | **0.1098** | **0.147 / 0.068** | **0.147 / 0.072** |

p=20 also matches translation/rotation 100 fs exactly. **p=20 is the best ACF match so far.**

## 3v. GLR independent-start 2.5 ns — completed

| Item | Value |
|---|---|
| Run | `experiments/glr_independent_2p5ns_20261122` |
| Initial | independent NVT frame 0 (301.8 K) |
| Protocol | GLR AR(10) + blend 0.1 + rescale, 500k steps |
| Result | stable; T=300.0005 K; partition 2.482/2.527/2.398; RDF 0.028/0.044/0.027; VDOS 0.847/0.953; **D 1.393e-9 (26% low)** |
| Verdict | initial-condition independence confirmed, including the diffusion deficit |

## 3w. GLR order ablation p=20/5/1 — completed

| Order | O–H RDF (100 ps) | O VDOS | T |
|---|---:|---:|---:|
| p=1 | 0.312 | 0.896 | 300 |
| p=5 | 0.131 | 0.965 | 300 |
| p=10 | **0.104** | **0.964** | 300 |
| p=20 | 0.117 | 0.964 | 300 |

All runs stable. p=10 is the best 100 ps O–H; p=20 has the better long-lag ACF but not the better
short-window O–H. Artifacts: `glr_p{1,5,20}_100ps_20261122`.

## 3x. Published TrajCast 2.5 ns — stopped (too slow)

| Item | Value |
|---|---|
| Run | `experiments/nvt_2p5ns_published_20261122` |
| Checkpoint | published IBM water e3nn checkpoint |
| Protocol | native CSVR, 50 ps thermalization + 2.5 ns production |
| Result | stopped at 3,000/510,000 steps after 1,000 s; projected ~47 h on the contended GPU |
| Decision | cite the paper's 2.18e-9; our 100 ps screen already validates the protocol; do not spend 20+ h on this baseline |
| Artifact | incomplete `trajectory.h5` kept, summary absent |

## 3y. Paper Phase 0–3 artifacts — completed

| Item | Value |
|---|---|
| Config | `paper/PAPER_CONFIG.md` (NMI, Nature style, LaTeX+PDF) |
| Novelty | `paper/NOVELTY_MATRIX.md` |
| Outline | `paper/OUTLINE.md` |
| Evidence map | `paper/EVIDENCE_MAP.md` |
| Argument blueprint | `paper/ARGUMENT_BLUEPRINT.md` |
| Methods draft | `paper/METHODS_DRAFT.md` |
| Figures | `paper/figures/fig1_diagnostic.{py,pdf,png}`, `fig2_architecture.{py,pdf,png}` |

## 3z. GLR 2.5 ns analysis script — ready

| Item | Value |
|---|---|
| Script | `method_development/run_glr_2p5ns_analysis.sh` |
| Action | tags `ensemble='nvt'`, runs motion/rdf/vdos/diffusion, writes `analysis/glr_2p5ns_report.json` |
| Trigger | after `glr_2p5ns_20261122/trajectory.summary.json` exists and `stable=true` |

## 3aa. GLR displacement correction — negative (crash at 24.8 ps)

| Item | Value |
|---|---|
| Run | `experiments/glr_cd_100ps_20261122` |
| Change | `--fd-correct-displacement`: add $\\Delta t(v^{\\rm GLR}-v^{\\rm pred})$ to the backbone displacement |
| Result | **failed at 4,969 steps (24.8 ps)**: min 0.647 Å, preT 299.7 K |
| Verdict | the decoupled scheme (positions from backbone, velocities from readout) is required; direct displacement correction injects an unstable velocity/displacement mismatch |

## 3ab. GLR 100 ps MSD diagnostic — diffusion consistent with reference at long lag

| Fit window | GLR D (m²/s) | Reference D (m²/s) |
|---|---:|---:|
| 10–50 ps | 1.40e-9 | 1.66e-9 |
| 10–80 ps | 1.45e-9 | 1.81e-9 |
| 10–95 ps | 1.54e-9 | 1.76e-9 |
| 40–95 ps | 1.69e-9 | 1.75e-9 |
| 60–95 ps | **1.90e-9** | 1.52e-9 |

The GLR MSD is superdiffusive at 50–100 ps (slope rising from 0.84 to 1.14), so the
short-window fit underestimates D. The 2.5 ns block fit (10–100 ps per 312 ps block) will
give the production number. No evidence for the FDR v1 fast-diffusion tail.

## 3ac. GLR blend sweep — stopped (L40S stall)

| Item | Value |
|---|---|
| Runs | `glr_blend0.2_100ps_20261122`, `glr_blend0.3_100ps_20261122` |
| Protocol | GLR AR(10) + rescale, val_04, seed 2026110401 |
| Result | both stalled at step 0 on GPU 4; killed |
| Decision | do not run on the contended L40S cards; rerun on a free RTX GPU if the λ ablation is needed |
| Baseline | blend 0.1: O–O 0.032, O–H 0.104, H–H 0.026; 100 ps D 1.45–1.90e-9 depending on fit window |

## 3ad. Final gate verdict — structure passes, diffusion 26% low

| Item | Value |
|---|---|
| Main 2.5 ns | stable; partition 2.482/2.526/2.399; RDF 0.031/0.038/0.028; VDOS 0.847/0.953; D 1.406e-9 |
| Independent 2.5 ns | stable; partition 2.482/2.527/2.398; RDF 0.028/0.044/0.027; VDOS 0.847/0.953; D 1.393e-9 |
| Reference | partition 2.463/2.501/2.447; D 1.883e-9 |
| Published | partition 2.505/2.463/2.444; D 2.18e-9 |
| Rescale off | T=172 K; partition 0.763/0.649/2.839; RDF 0.398/0.474/0.279; VDOS 0.432/0.608 |
| Noise off | structure/D unchanged at λ=0.1; not necessary |
| Seeds 1/2/3 | partition identical; O–H 0.104/0.111/0.110; O VDOS 0.964/0.966/0.959 |
| Diffusion fixes | mixed p20, blend, translation displacement, COM scale — none close the gap |
| Paper | LaTeX draft with 4 figures + Table 1, 16 pages; claim locked around thermodynamic/structure consistency, diffusion reported as a limitation |

## 3ae. 5 fs quartz — transitions/cache built; backbone training

| Item | Value |
|---|---|
| Root | `experiments/quartz5fs_20261122` |
| Transitions | `quartz/transitions/5fs/{train,val,test}.h5` (5000/1250/1000) |
| Cache | `cache/quartz5_{train,val,test};P:64.pt` (1.4 GB) |
| Training run | `experiments/quartz5_relational_20261122` |
| Model | RelationalTransition, 4.5 Å, 5 fs horizon, 2.36M params |
| Progress | epoch 1/400, 58 s/epoch, ETA ~6.5 h on L40S; val pos RMSE 8.4e-4 Å at epoch 1 |
| Purpose | true cross-material positive system (Si/O species modes at a resolved horizon) |

## 3af. 5 fs quartz species GLR fit

| Item | Value |
|---|---|
| Fit | `experiments/quartz5_glr_fit_20261122/species_coefficients.json` |
| Reference | quartz NVT 15 ns, 5–6 ns block, 5 fs stride |
| Prediction error | O 0.0137, Si 0.0150 (vs 0.91 at 30 fs) |
| Interpretation | the 5 fs ACF is resolved; the memory kernel is informative, unlike the 30 fs case |

## 4. What not to touch

- `data1/.../experiments/phase_space_20261120/water5/best.pt` is the frozen v0 checkpoint;
  do not overwrite it.
- The NVT output path must not be overwritten; the rollout script refuses existing files.
- The active H1 worktree `code/h1h2` is separate from `data1/.../smdp-future-control`; do not
  edit the active worktree.
- The H1 screening, E1 A/B and NVT use different GPUs; do not launch an additional heavy job
  on GPUs 1/2/4 until one finishes.

---

## 5. After jobs finish

1. Compare H1 arm A vs B at the same epoch / budget:
   - endpoint RMSE;
   - 500 fs and 1 ps rollout partition error;
   - whether the selected checkpoints differ.
2. Compare E1 baseline vs motion-balanced:
   - endpoint RMSE;
   - 20 ps NVE translation/rotation/internal KE;
   - RDF and VDOS;
   - whether the translation gain appears without an internal/rotation regression.
3. Analyze v0 NVT:
   - temperature, translation/rotation/internal KE;
   - RDF, VDOS, MSD/diffusion;
   - compare with Full/Current-Only on the same protocol.
4. Fix the pipeline topology path bug before any automatic production analysis.
5. Decide whether to scale H1/E1 to 50k/250k and whether to run a second v0 NVT seed.
