# Claim boundary — backbone vs readout

**Date:** 2026-11-27  
**Purpose:** prevent the two project routes from being conflated in the paper, README, or talks.

## The two routes

| | Route A — raw force-free backbone | Route B — fixed GLR readout |
|---|---|---|
| Object | a trained direct propagator (relational or phase-space) | a statistical velocity readout on a **frozen** backbone |
| What it predicts | displacement and velocity from the current state | velocity correction from the reference mode ACF + variance target |
| Short-time evidence | water 5 fs pos RMSE 4.9e-4 Å; quartz 30 fs 1.9e-3 Å | applied to frozen checkpoints; no retraining |
| Long-time water | **fails**: v0 NVT VDOS 0.356/0.236, \(D=2.8\times10^{-13}\) m²/s | partition/RDF/VDOS restored; **\(D\) 22–25% low**; decoupled position/velocity |
| Long-time quartz | 30 fs NVE drift +0.904/−0.330 eV (inconsistent) | 5 fs, 100 ps, two seeds: \(T=299.9\) K, VDOS 0.950/0.955, Si–O RDF 0.028 |
| Method status | not a long-time method | **this is the paper's method** |

## What the paper claims

> A frozen force-free backbone supplies the displacement; a fixed mode-resolved GLR readout
> supplies the velocity and corrects the **stationary measure**. The readout is the method; the
> backbone is a plug-in.

The paper **does not** claim:

- that the raw phase-space or relational network alone reproduces long-time water dynamics;
- that the 30 fs quartz backbone is improved by the 5 fs phonon-band result;
- that the GLR recovers \(D\) exactly (water is 26% low);
- that the deterministic rescale reproduces the kinetic-energy variance (it is mean-only);
- that the stochastic rescale is free (it restores the KE variance but increases the
  position–velocity kinematic residual; the 8-block GK/MSD ratio is 1.06, so the earlier
  first-block "GK/MSD break" is not a systematic 2.5 ns statement);
- that quartz is validated at 2.5 ns (only 100 ps, two seeds);
- that the new GLE readout improves autonomous dynamics (its offline gate failed);
- that FDT alone guarantees Gibbs invariance when composed with frozen R3-Opt (the composed
  process was not validated);
- that the 100 ps official-TrajCast control is the 2.5 ns result (the 2.5 ns run is pending).

## What is locked

- Unified method definition (same algorithm, material-specific basis): choose an orthogonal,
  physically defined mode basis (molecular translation/rotation/internal for water; phonon bands
  for quartz); fit each mode ACF on a reference NVT trajectory; apply AR(\(m\)) memory only where
  the backbone ACF is wrong; rescale the mode KE to the canonical target; blend \(\lambda=0.1\);
  project out the COM. The basis is material-specific; the algorithm and parameter-selection rule
  are shared.
- The readout algorithm: physics-defined mode basis, AR(10) memory where the backbone ACF is
  wrong, group-resolved variance rescale, blend \(\lambda=0.1\), COM projection,
  \(q'=q+\Delta q\).
- Water production: molecular modes + memory + deterministic rescale (GK/MSD agree to ~4%;
  D is still 22–25% low).
- Quartz: phonon-band partition + stochastic rescale + memory off.
- The negative controls: full species GLR, full phonon GLR, per-mode rescale.

## What is not locked

- Whether the 22–25% water \(D\) deficit is a readout failure, a calibration-window bias, or the
  AR(10) 0–50 fs local fit. The 25×100 ps reference ACF integral is 14.45 ± 1.86 fs; independent
  100/200 ps calibration gives the same AR(10) and a 2.3–2.5× AR integral, so the
  calibration-trajectory length is not the bottleneck. A longer fit window / D-constrained AR
  fit would be a method change, not authorized here.
- Whether a single rescale can satisfy both the KE marginal and GK/MSD consistency.
- Quartz at 2.5 ns (Stage 1).
- Temperature transfer, larger box, third material, action-map baseline (Stage 2).

## Position–velocity and calibration audit (2026-11-27)

- Production GLR is a **closed-loop velocity-feedback readout**, not a post-hoc correction: the
  current-step displacement is the backbone displacement, but `v_new` is the next velocity state
  and drives the next backbone call (and the AR history). The one-step probe changes the next
  backbone displacement by **~62%** of `dq0` from a ~21% velocity correction. The local
  `q_{t+1}-q_t = h v_new` equality does not hold (midpoint residual 1.3%/3.8%), but that is a local
  kinematic discretization issue, not proof of no closed-loop dynamics. `--fd-correct-displacement`
  crashed at 24.8 ps and was not used.
- Eight 100 ps blocks: reference D_MSD 1.890 ± 0.410, D_GK 1.943 ± 0.253, GK/MSD 1.052;
  deterministic GLR 1.476 ± 0.228 / 1.392 ± 0.127, ratio 0.958; stochastic GLR 1.350 ± 0.243 /
  1.403 ± 0.136, ratio 1.058. The midpoint kinematic residual is 1.0% (reference), 1.3%
  (deterministic), 3.8% (stochastic) of `h|v|`. The D deficit is a **common low-D bias**, not a
  GK/MSD inconsistency; the earlier stochastic first-block difference is not systematic over
  2.5 ns.
- The Yule–Walker fit uses **0–50 fs**; `--maxlag 40` only extends the reported ACF to 200 fs.
  The AR(10) is stable (radius 0.853–0.855) but overpredicts the 5 ps integral by 2.3–2.5×. The
  offline AR D ~4.4×10⁻⁹ vs autonomous GLR D ~1.4–1.5×10⁻⁹; offline ACF improvements do not
  transfer to the coupled rollout.
- Raw 250k backbone: D_MSD ~0; GLR restores D to ~1.4–1.5×10⁻⁹. This is a real improvement over
  the raw backbone, but not reference-level.

## Velocity-feedback intervention (2026-11-27)

- One-step probe: a ~21% `v_new−v_pred` correction changes the next backbone displacement by
  ~62% of `dq0` and the next velocity by ~113% of `v_pred`; the correction is nearly orthogonal to
  the current velocity. The loop is closed; do not call it a post-hoc correction.
- 5 ps separate controls: full and thermostat-only hold 300.000 K; no_feedback/all_no_feedback
  heat to ~307–308 K and distort the translation/rotation/internal partition. The corrected/
  rescaled velocity feedback is the thermostat.
- **Caveat:** two independent full runs from the same state diverged 5.18 Å by 5 ps (CUDA
  non-determinism/chaos). Separate-process position divergence is not a clean feedback measure.
  A same-process paired 200-step control gives a full-vs-full floor of **0.00047 Å / 0.000045
  Å/fs**; no_feedback and thermostat_only diverge by **0.70/0.71 Å** (~1,500× the floor). The
  feedback/readout effect is real. The 500-step paired T log shows full 300.000 K vs no_feedback
  277–311 K.
- `no_feedback` is not thermostatted (T 287.9 K at 1 ps, partition distorted); `thermostat_only`
  holds 300 K but differs from full by 0.714 Å and a different rotation/internal partition. The
  AR/blend readout itself, not only the feedback path, changes the trajectory at matched T.
- The 5 ps early MSD is a mechanism check, not a long-time D. At matched T, thermostat-only has a
  larger early slope than full; AR/blend reduces early diffusion. Do not report a 5 ps D, and do
  not infer the AR as the sole cause of the long-time D deficit from this 1 ps result.

## Existing-ablation necessity audit (2026-11-27)

No rollout was rerun; the existing 100 ps trajectories were re-analyzed with the same MSD/GK
metrics.

- **Rescale is necessary:** without it, T = 172 K, partition 0.76/0.65/2.84 eV, O–O RDF 0.398,
  VDOS 0.431/0.608, D ≈ 0. This is a strong, unambiguous ablation.
- **Memory is necessary for water:** AR(1) gives D 4.25/4.55 ×10⁻⁹ (2.5× too high), O–O RDF
  0.089, O–H 0.312, VDOS 0.809/0.896. AR(5) is close to AR(10); AR(20) is not better than AR(10).
  The p=1 result supports the memory claim; p=10 vs p=20 is not distinguishable.
- **FD noise is not necessary for D at λ=0.1:** deterministic (noise-off) D 1.62/1.47; stochastic
  1.42/1.38; the difference is within seed/block variability. It **is** necessary for the KE
  variance: deterministic partition std 0.002 eV vs stochastic 0.253 eV.
- **Mode blend is only partially supported:** translation blend 0.1 gives D 1.62–1.71, O–O RDF
  0.032–0.034, VDOS 0.849/0.964; blend 0.3/0.5 gives D 0.88/0.97, O–O RDF 0.061/0.095, VDOS
  0.848/0.911 and 0.853/0.865, with larger kinematic residuals. The 0.1 translation blend is
  supported; the internal-mode 1.0 ablation (`glr_int10_100ps`) is missing, so a full
  "mode decomposition is necessary" claim is not established.
- **Seed variability:** 100 ps D_MSD is 1.26–1.71 ×10⁻⁹ across seeds, ~27% spread. The 22–25%
  deficit is within this seed/block variability; it should be reported with uncertainty, not as a
  precise method failure.

## Fair no-AR-memory baseline (2026-11-27)

`rescale_only`: frozen backbone + same mode basis + same per-mode canonical target + COM
projection, but no AR coefficients, no ACF calibration and no readout history. It is not AR(1)
and not rescale-off. Two independent 100 ps starts:

| run | D_MSD (10⁻⁹) | ACF integral 0–5 ps (fs) | RDF O–O/O–H | VDOS O/H |
|---|---:|---:|---:|---:|
| reference block 0 | 1.692 | 11.341 | — | — |
| full deterministic | 1.62–1.71 | 10.8–11.0 | 0.032–0.036 / 0.098–0.104 | 0.963 / 0.848–0.851 |
| `rescale_only` | **2.67–2.82** | **17.3–17.5** | 0.038–0.042 / 0.122–0.131 | 0.972–0.975 / **0.941** |

- The simple rescale **over-diffuses** and has a 100 fs ACF shoulder that is too high. AR memory
  is necessary for the 100 ps ACF/D and O–O/O–H RDF.
- The AR memory is **not uniformly better**: `rescale_only` has a better H VDOS (0.941 vs
  0.848–0.851) and lower H VACF error (0.024 vs 0.073–0.076). The method is a trade-off, not a
  free improvement.
- **Stochastic no-AR baseline (100 ps, two starts):** full ACF 9.99–10.59 vs no-AR 14.53–16.28
  (reference block 0 11.34); full GK 1.38–1.47 vs no-AR 2.01–2.25 (reference 1.50). The AR
  ACF/GK benefit persists under stochastic. The MSD benefit is clear on val_04 (1.42 vs 1.93) but
  nearly tied on independent (1.90 vs 1.95), within single-100-ps block noise. The H/O VDOS cost
  persists: no-AR H 0.946–0.951, O 0.976–0.981; full H 0.854–0.857, O 0.957–0.964. O–H RDF is
  better with AR; O–O/H–H RDF is better without. The stochastic rescale restores the canonical KE
  distribution in both; the AR memory does not create the KE variance.
- A matching 2.5 ns deterministic `rescale_only` baseline is running on val_04 (one long start;
  two 100 ps starts cover initial-state independence). A stochastic 2.5 ns no-AR baseline was not
  run; its long-time benefit is unmeasured. Do not fill a 2.5 ns baseline number before it exists.
  If the deterministic baseline is closer to the reference than the full GLR, the AR memory's
  long-time necessity is not supported and the paper must say so.
- The official TrajCast 2.5 ns is still running; no 2.5 ns D/VDOS number.

## GLE gate (2026-11-27)

The bounded 4–6 mode DHO+slow GLE study was executed and **failed the offline gate**:

- The exact 2×2 complex-OU realization is correct (Monte Carlo ACF RMSE 0.0025).
- The unconstrained 3-mode fit matches the 5/25 fs ACF but its discrete D is **1.90×** the
  reference; the unconstrained 6-mode is 1.52×. The near-Nyquist ω≈1.25 rad/fs mode is aliased
  at the 5 fs cadence.
- The frequency-bounded 5-mode fit restores D (+1.0%) and the long-lag ACF (RMSE 0.047) but its
  5 fs ACF is 0.726 vs 0.982 (error 0.255); the 6-mode does not fix it.
- The independent `train_nvt_200ps` reference shows the same 5/25 fs shape, so this is not a
  calibration-block artifact.

**Classification:** kernel identification at 5 fs, not discrete implementation, not neural
coupling. No 100 ps or 2.5 ns GLE rollout was started. A different kernel family or a 1 fs
cadence would be needed; neither is authorized.

## Official TrajCast 2.5 ns (pending)

The official IBM published checkpoint is running under the same initial state, CSVR thermostat,
5 fs horizon, and 50 ps thermalization + 2.5 ns production protocol. The 100 ps control has
excellent temperature/total-KE fluctuations and O VDOS 0.937, with H VDOS 0.624 and ~5–6%
mode-partition differences. Do not report a 2.5 ns number until
`experiments/published_trajcast_2p5ns_20261126/analysis/trajcast_2p5ns_report.json` exists.

## Version-control state

- Local reproducible snapshot: `reproducibility/snapshot_20261127/`, git branch
  `research/glr-gle-20261127`, tag `research-snapshot-20261127` (local-only; not pushed).
- GitHub branch `research/h1h2-selection-20261120` still stops at `4399ccd`; the current paper,
  experiments, and GLR coefficients are local-only.
- The snapshot includes the GLR/GLE code, coefficients, fit/audit results, logs, `MANIFEST.json`,
  and `CHECKSUMS.sha256`. Large trajectories are referenced, not copied.
- README and RUNNING_JOBS were stale; this claim boundary and `SMDP_ICML2027_STATUS_AUDIT.md`
  are the authoritative documents.

**Action:** push the snapshot branch/tag and freeze a submission tag only after the official
TrajCast 2.5 ns result is analyzed and the GLR provenance/refit decision is recorded.
