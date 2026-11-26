# SMDP/DSRT — ICML 2027 reproducible snapshot (2026-11-27)

## 0. Bottom line (中文)

- **Frozen R3-Opt + GLR remains the only ICML method candidate.** No new propagation backbone.
- **The exact discrete GLE offline gate failed.** The 2×2 complex-OU realization is correct
  (Monte Carlo ACF error 0.0025), but no 3–6 mode DHO+slow kernel simultaneously fits the
  reference 5/25 fs ACF and the discrete Green–Kubo integral. The failure is **kernel
  identification at the 5 fs cadence**, not the discrete implementation and not the neural
  coupling. **No 100 ps GLE rollout was started.**
- **Official IBM published TrajCast 2.5 ns is still running.** A watcher will run the same
  `analyze_water.py` protocol when it finishes. Do not report a 2.5 ns number before
  `trajectory.summary.json` and `analysis/trajcast_2p5ns_report.json` exist.
- **The GLR 22–25% D deficit is a common low-D bias, not a GK/MSD inconsistency.** The 25×100 ps
  reference ACF integrals span 10.94–18.00 fs (mean 14.45, std 1.86). The production GLR's
  Yule–Walker fit uses only **0–50 fs**; `--maxlag 40` only extends the reported ACF to 200 fs.
  A 100/200 ps independent calibration gives the same AR(10) and the same 2.3–2.5× integral
  overprediction, so calibration length is not the bottleneck.
- **The production GLR is a closed-loop velocity-feedback readout.** The current-step displacement
  is the backbone displacement, but `v_new` is the next velocity state and drives the next
  backbone call (and the AR history). A one-step probe with a ~21% velocity correction changes the
  next backbone displacement by **~62%** of `dq0`. The local `q_{t+1}-q_t = h v_new` equality does
  not hold (midpoint residual 1.0% reference, 1.3% deterministic, 3.8% stochastic), but that is a
  local kinematic discretization issue, not proof of no closed-loop dynamics.

## 1. What is in this snapshot

```
code/h1h2/               core GLR/GLE/fit/analysis code (selected)
configs/                 run scripts and the TrajCast 2.5 ns watcher
parameters/              GLR coefficients, GLE fit parameters, reference ACF blocks
results/                 fit results, audits, 2.5 ns GLR/stochastic reports, 100 ps TrajCast control
logs/                    fit/audit/run logs (the 2.5 ns run log is a partial snapshot)
data_sources.json        large read-only data paths and checksums
MANIFEST.json            every copied file: original path, size, SHA256
CHECKSUMS.sha256         sha256sum-compatible checksum list
```

The snapshot deliberately **does not copy** the 22 GB reference NVT trajectory, the 2.5 GB GLR
trajectory, or the 7 GB TrajCast 2.5 ns trajectory. Their paths, sizes, and (where practical)
checksums are in `data_sources.json`.

## 2. Environment

- Main environment: `/data1/wyd/envs/mlip/bin/python`
  - Python 3.12.3, numpy 2.2.6, h5py 3.16.0, torch 2.13.0+cu130, LAMMPS 20250722, ASE 3.29.0.
  - LAMMPS needs `LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib`.
- CPU-only fit environment used in this round:
  `/data2/wyd/tmp_opencode/env_mace` (Python 3.11, numpy/scipy). This is a temporary path; any
  Python with numpy + scipy can run the offline fits.
- TrajCast upstream: `/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107`.

## 3. Data sources

| role | path | size | notes |
|---|---|---:|---|
| reference NVT 2.5 ns | `/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5` | 23,205,934,562 B | calibration + evaluation; 0.5 fs storage, 5,000,001 frames |
| independent NVT 200 ps | `.../water/physical_reference/train_nvt_200ps/production.h5` | 2,492,837,498 B | independent trajectory; used to confirm the ACF failure is systematic |
| independent NVT 500 ps dump | `.../water/.../experiments/independent_nvt_500ps_20261121/production.lammpstrj.gz` | 714,695,507 B | LAMMPS NVT, independent start |
| topology | `.../water/equilibration/snapshots/water_5.data` | 30,310 B | SPC/Fw, 64 molecules, 192 atoms |
| official TrajCast checkpoint | `reference/trajcast_published_water/published_checkpoint.pt` | 18,719,751 B | SHA256 `a7f18954…15a62ea`; IBM published water state |
| GLR coefficients | `experiments/glr_fit_20261122/glr_coefficients.json` | 10,823 B | SHA256 `a6e65965…15682d`; **no calibration metadata** |

## 4. Reproduce

Set `ROOT` to the project root and `DATA` to the reference-data root.

```bash
# 1. 25-block reference ACF / D uncertainty
PYTHONPATH=code/h1h2 python code/h1h2/reference_acf_blocks.py

# 2. Continuous 3–6 mode DHO+slow fits on the 25-block mean ACF
PYTHONPATH=code/h1h2 python code/h1h2/gle_kernel_study_mean.py

# 3. Generalized DHO+slow fit (underdamped + overdamped, discrete-D constraint)
PYTHONPATH=code/h1h2 python code/h1h2/fit_gle_general.py \
  --output experiments/fit_gle_general_overdamped_20261126.json \
  --modes 4 5 6 --n-starts 32 --omega-max 0.5

# 4. Monte Carlo test of the exact A_d / Q_d realization
PYTHONPATH=code/h1h2 python code/h1h2/verify_gle_discrete.py \
  --fit parameters/fit_gle_general_overdamped_20261126.json --mode-index 2

# 5. Audit the production GLR coefficients (stability, extrapolated ACF, D)
PYTHONPATH=code/h1h2 python code/h1h2/audit_glr_coefficients.py \
  --coefficients parameters/glr_coefficients.json

# 6. Official TrajCast 100 ps control analysis (already complete)
cd code/h1h2
LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib \
PYTHONPATH=$DATA/upstream/trajcast-c52f107:$PWD \
python -u analyze_water.py \
  --candidate $ROOT/experiments/nvt_screen_published_trajcast_20261121/trajectory.h5 \
  --reference $DATA/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5 \
  --topology $DATA/data/smdp_reference_md_v1/water/equilibration/snapshots/water_5.data \
  --output $ROOT/experiments/nvt_screen_published_trajcast_20261121/analysis \
  --tasks motion rdf vdos --frames 2000 --model-label "Published TrajCast water (IBM, 5 fs)"

# 7. Official TrajCast 2.5 ns: wait for the run, then analyze
bash configs/experiments/published_trajcast_2p5ns_20261126/analyze_when_done.sh
```

Seeds: GLR production rollouts `2026110401` (seed 1), `2026110402` (seed 2), `2026110403`
(seed 3); GLE fit `20261126 + 1000*modes`; Monte Carlo `20261126`; TrajCast 2.5 ns `2026110401`.

## 5. Results

### 5.1 Reference ACF / D statistical uncertainty

25 consecutive 100 ps blocks of the 2.5 ns reference NVT; translation ACF integral (0–5 ps):

| quantity | value |
|---|---:|
| mean | **14.4479 fs** |
| std (ddof=1) | **1.8636 fs** |
| SEM | 0.3727 fs |
| min | **10.9356 fs** (block 4, 400–500 ps) |
| max | **18.0034 fs** (block 2, 200–300 ps) |
| block 0 (0–100 ps) | **11.3405 fs** |
| block 1 (100–200 ps) | 14.4674 fs |
| block 5 (500–600 ps) | 13.3812 fs |

The reference D from the 2.5 ns MSD fit is **1.883 ± 0.306 ×10⁻⁹ m²/s**. The block-to-block
CV is ~13%, so a single 100 ps block is not a D target.

### 5.2 GLR calibration provenance and AR audit

- `glr_coefficients.json` records **no reference path, start time, duration, or ACF window**.
- `fit_glr.py` defaults: start 500 ps, duration 100 ps, stride 5 fs, maxlag 40 frames (200 fs),
  order 10. **The Yule–Walker fit actually uses only ACF lags 0..10 (0–50 fs at 5 fs stride);
  `maxlag=40` only extends the reported ACF to 200 fs, it does not constrain the coefficients.**
  The production fit is inferred to be block 5 (500–600 ps), not recorded.
- The translation AR(10) is stable (spectral radius 0.855) and matches 5/25 fs
  (0.9816/0.6578), but it misses the 100–500 fs negative dip:
  - AR: 100 fs −0.0316, 200 fs +0.0016
  - reference: 100 fs +0.0391, 200 fs −0.0781
- Extrapolating the AR recursion to 5 ps gives a translation integral of **33.24 fs**, 2.30× the
  25-block mean. The 0–50 fs Yule–Walker window cannot constrain the 100–500 fs dip or D.
- Independent `train_nvt_200ps` calibration (never the 2.5 ns test trajectory):

| window | AR radius | AR integral (fs) | actual 5 ps integral (fs) | ratio | local RMSE (≤200 fs) | long RMSE (≥500 fs) |
|---|---:|---:|---:|---:|---:|---:|
| 100 ps, 0–100 | 0.8554 | 32.63 | 12.95 | 2.52× | 0.055 | 0.015 |
| 100 ps, 100–200 | 0.8526 | 33.47 | 14.67 | 2.28× | 0.052 | 0.013 |
| 200 ps, 0–200 | 0.8531 | 33.07 | 13.82 | 2.39× | 0.053 | 0.014 |
| production | 0.8552 | 33.24 | — | 2.30× vs 25-block | — | — |

100 vs 200 ps calibration does not change the AR(10) coefficients, stability, or the 2.3–2.5×
integral overprediction; the **50 fs local fit** is the bottleneck, not the calibration-trajectory
length. The production fit is reproducible from independent data; its missing metadata is a
documentation gap, not evidence of a bad fit.
- **Offline vs autonomous:** AR-implied D ~4.4×10⁻⁹ m²/s; independent-window D ~1.8–2.0×10⁻⁹;
  autonomous GLR D ~1.4–1.5×10⁻⁹. The AR overpredicts D, the coupled rollout underpredicts it;
  improving the offline AR ACF alone does not transfer to the autonomous simulation.
- Calibration/evaluation overlap: the production fit likely used the 2.5 ns trajectory. The
  independent 200 ps gives the same coefficients, so leakage is not driving the low D, but the
  paper should record the calibration window and use a held-out trajectory.

### 5.3 GLE offline gate — failed

The exact 2×2 complex-OU embedding is correct: Monte Carlo sampling of the 5-mode state-space
matches the analytic ACF at 5–5000 fs with **RMSE 0.0025** (max error 0.0067). The failure is the
kernel, not the discrete implementation.

| model | discrete integral (fs) | ratio to 14.4479 | key RMSE | max key error | 5 fs ACF | 25 fs ACF | long-lag RMSE (≥25 fs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| unconstrained 3-mode (continuous fit) | **27.4596** | **1.90×** | 0.038 | 0.088 | 0.968 | 0.572 | 0.041 |
| generalized 3-mode | 14.547 | 1.007 | 0.141 | 0.375 | 0.606 | 0.528 | 0.050 |
| generalized 4-mode | 14.566 | 1.008 | 0.115 | 0.296 | 0.662 | 0.510 | 0.057 |
| generalized 5-mode | **14.591** | **1.010** | **0.100** | **0.255** | 0.726 | 0.536 | **0.047** |
| generalized 6-mode | 14.592 | 1.010 | 0.097 | 0.247 | 0.734 | 0.539 | 0.046 |
| reference (25-block mean) | 14.4479 | 1 | — | — | **0.982** | **0.659** | — |

The unconstrained 3-mode fit reaches the 5/25 fs values only with an ω≈1.25 rad/fs mode
(period ≈5.04 fs). At the 5 fs rollout cadence that mode is aliased: its discrete ACF integral is
1.90× the reference. Bounding the frequency to ≤0.5 rad/fs (period ≥12.6 fs) restores D but
cannot reproduce the 5/25 fs shoulder. The 4/5/6-mode fits do not fix the 5 fs point; the 6-mode
adds 3 parameters and 0.003 key RMSE. **Mode count is not the bottleneck.**

The independent `train_nvt_200ps` reference shows the same ACF (5 fs 0.9816/0.9818, 25 fs
0.6573/0.6606, 100 fs 0.0314/0.0408, 200 fs −0.0822/−0.0728), so the failure is systematic, not
a calibration-block artifact. The exact discrete 5-mode process's long-time ACF (RMSE 0.047)
and D (+1.0%) are correct; the failure is the **short-time (5 fs) kernel shape**.

PSD/realizability/stability: the 5-mode ACF's real FFT is non-negative (min **0.500** at 0.5 fs,
**0.508** at 5 fs); `Q_d` min eigenvalue **1.4e-7 > 0**; `A_d` spectral radius **0.991 < 1**.
The exact discrete FDT is satisfied. The failure is the short-time shape, not realizability or
stability.

**Decision: the offline gate did not pass. No 100 ps GLE rollout was started.** The failure is
classified as **kernel identification at 5 fs**; the discrete implementation is verified, and
the neural coupling was not exercised.

### 5.4 Official TrajCast control

**100 ps control (official IBM published checkpoint, same initial state / CSVR / 5 fs):**

| metric | TrajCast | reference |
|---|---:|---:|
| temperature (K) | 300.19 ± 17.68 | 300.19 ± 17.89 |
| total KE (eV) | 7.4113 ± 0.4366 | 7.4114 ± 0.4417 |
| translation KE (eV) | 2.5050 | 2.3725 |
| rotation KE (eV) | 2.4626 | 2.4302 |
| internal KE (eV) | 2.4437 | 2.6087 |
| RDF RMSE O–O / O–H / H–H | 0.0431 / 0.1180 / 0.0957 | — |
| VDOS overlap O / H | 0.9370 / 0.6241 | — |
| D | not reported (100 ps < frozen fit limit) | — |

The total temperature/KE and fluctuations are excellent. The mode-resolved partition differences
are ~0.5–0.7 block σ at 100 ps; the H VDOS is the weakest metric. The **2.5 ns** run is the
formal comparison and is still running. The 100 ps numbers are context, not a 2.5 ns result.

**2.5 ns status:** `experiments/published_trajcast_2p5ns_20261126/`, PID 3840811 on GPU 3,
50 ps thermalization + 500,000 production steps. At 2026-11-27 00:xx it was at step ~75,000/
510,000, ETA ~20 h. `configs/experiments/published_trajcast_2p5ns_20261126/analyze_when_done.sh`
waits for `trajectory.summary.json`, runs `analyze_water.py` with the same protocol, and writes
`analysis/trajcast_2p5ns_report.json`.

### 5.5 GLR 2.5 ns vs stochastic GLR vs reference

| run | T (K) | total KE (eV) | translation KE std (eV) | D (10⁻⁹ m²/s) | RDF RMSE O–O | VDOS overlap O/H |
|---|---:|---:|---:|---:|---:|---:|
| reference 2.5 ns | 300.17 | 7.4108 | 0.2534 | 1.883 ± 0.306 | — | — |
| deterministic GLR | 299.99 | 7.4064 | **0.0017** | **1.406 ± 0.155** | 0.0307 | 0.953/0.847 |
| stochastic GLR | 299.97 | 7.4058 | **0.2520** | **1.435 ± 0.130** | 0.0276 | 0.949/0.852 |
| published TrajCast 100 ps | 300.19 | 7.4113 | 0.2582 | pending | 0.0431 | 0.937/0.624 |
| official TrajCast 2.5 ns | pending | pending | pending | pending | pending | pending |

Deterministic GLR recovers the mean partition but suppresses the KE variance. Stochastic GLR
restores the variance; both leave D 22–25% low. The earlier "stochastic GLR breaks GK/MSD" claim
is **not systematic over 2.5 ns**: the 8-block GK/MSD ratio is 0.958 (deterministic), 1.058
(stochastic), 1.052 (reference). The stochastic rescale increases the kinematic residual
(1.3% → 3.8%), but not a 20% GK/MSD offset. The GLE was intended to fix both; its offline gate
failed.

### 5.6 Position–velocity consistency

The production GLR is a **closed-loop velocity-feedback readout**, not a post-hoc correction. The
current-step displacement is the backbone displacement; `v_new` is written to
`output["velocities"]`, becomes `v_{t+1}`, is fed into the next backbone call, and is appended to
the GLR AR history. The `--fd-correct-displacement` test crashed at 24.8 ps (min distance
0.647 Å), so the current-step displacement was kept decoupled for stability; this is a local
kinematic discretization choice, not absence of closed-loop dynamics.

One-step probe (production checkpoint/coefficients, deterministic readout, matched state):

| state | correction / v_pred | next displacement change / dq0 | next v_pred change / v_pred |
|---|---:|---:|---:|
| val_04 frame 0 | 0.218 | **0.632** | **1.131** |
| GLR 2.5 ns frame 20,000 | 0.209 | **0.617** | **1.143** |

A ~21% velocity correction changes the next backbone displacement by ~62% and the next velocity by
~113%; the feedback is strong and nearly orthogonal to the current velocity.

Eight 100 ps blocks (0, 300, …, 2100 ps), same 5 fs discrete rule:

| trajectory | D_MSD (10⁻⁹ m²/s) | D_GK (10⁻⁹ m²/s) | GK/MSD | midpoint residual / h\|v\| |
|---|---:|---:|---:|---:|
| reference 2.5 ns | 1.890 ± 0.410 | 1.943 ± 0.253 | 1.052 | 0.010 |
| deterministic GLR 2.5 ns | 1.476 ± 0.228 | 1.392 ± 0.127 | 0.958 | 0.013 |
| stochastic GLR 2.5 ns | 1.350 ± 0.243 | 1.403 ± 0.136 | 1.058 | 0.038 |
| raw 250k backbone 100 ps | ~0 (no diffusive regime) | 0.012 | — | 0.031 |

The reference block-to-block D std is 22%; a single 100 ps block cannot resolve a 20–25%
difference. The deterministic GLR's GK and MSD agree; its D deficit is a **common low-D bias**,
not a GK/MSD inconsistency. The 1.3% deterministic kinematic residual cannot explain a 22% D
deficit. The stochastic residual is 3× larger, but its 8-block GK/MSD ratio is still ~1.06. The
raw backbone has no diffusion; the GLR restores D to ~1.4–1.5×10⁻⁹, a real improvement but still
below the reference.

Matched 5 ps separate-process feedback controls (same checkpoint/coefficients/initial state):

| mode | T mean ± std (K) | translation/rotation/internal KE (eV) | early D 1–5 ps (10⁻⁹ m²/s) |
|---|---:|---:|---:|
| full | **300.000 ± 0.000014** | 2.481/2.521/2.401 | 1.333 |
| no_feedback | **308.18 ± 20.40** | 2.099/2.124/3.375 | 0.721 |
| all_no_feedback | **307.38 ± 21.07** | 2.155/2.180/3.268 | 0.696 |
| thermostat_only | **300.000 ± 0.000014** | 2.481/2.541/2.381 | 2.224 |

The full GLR and thermostat-only control hold T=300 K; removing the feedback of the corrected/
rescaled velocity heats to ~307–308 K and distorts the partition. The feedback is the thermostat.
At matched T, thermostat-only has a larger early MSD slope than full; the AR/blend readout reduces
early diffusion. **Caveat:** two independent full-feedback processes from the same state diverged
by 5.18 Å in 5 ps from CUDA non-determinism/chaos; the separate-process position divergence is not
a clean feedback effect.

Same-process paired control (200 steps, 1 ps; `full_control` is the chaos floor):

| branch | T mean ± std (K) | trans/rot/int KE (eV) | final RMS position from full (Å) |
|---|---:|---:|---:|
| full | 300.000015 ± 0.000014 | 2.4818/2.5970/2.3278 | 0 |
| full_control | 300.000014 ± 0.000015 | 2.4818/2.5968/2.3280 | **0.00047** |
| no_feedback | 287.92 ± 13.87 | 2.0704/2.3089/2.4026 | **0.7005** |
| thermostat_only | 300.000010 ± 0.000013 | 2.4818/2.6493/2.2755 | **0.7142** |

The same-process floor is 0.00047 Å; the no_feedback and thermostat-only branches diverge by
0.70/0.71 Å (~1,500× the floor). The feedback/readout effect is real, not CUDA chaos. The
no_feedback branch is not thermostatted; thermostat-only holds 300 K but differs from full, so
the AR/blend readout itself changes the trajectory at matched T. The 500-step paired T log is
retained as the longer thermodynamic evidence; the 200-step COM-MSD field was not usable due to
an interpolation bug, so the 5 ps early MSD above is the available (non-formal-D) indicator.

### 5.7 Existing-ablation necessity audit

No rollout was rerun; the existing 100 ps trajectories were re-analyzed with the same MSD/GK
metrics.

| run | D_MSD (10⁻⁹) | D_GK (10⁻⁹) | RDF O–O/O–H/H–H | VDOS O/H | comment |
|---|---:|---:|---:|---:|---|
| deterministic | 1.711 | 1.528 | 0.032/0.104/0.026 | 0.964/0.848 | partition std 0.002 |
| deterministic noise-off | 1.623 | 1.473 | 0.034/0.098/0.026 | 0.962/0.849 | partition std 0.002 |
| stochastic | 1.417 | 1.382 | 0.036/0.114/0.022 | 0.957/0.854 | partition std 0.253 |
| rescale off | 0.001 | 0.021 | 0.398/0.474/0.279 | 0.608/0.432 | T=172 K; collapse |
| translation blend 0.3 | 0.883 | 1.092 | 0.061/0.105/0.027 | 0.911/0.848 | residual 0.018 |
| translation blend 0.5 | 0.969 | 1.024 | 0.095/0.102/0.033 | 0.865/0.853 | residual 0.030 |
| AR(1) | 4.249 | 4.545 | 0.089/0.312/0.105 | 0.896/0.809 | no memory; 2.5× D |
| AR(5) | 1.837 | 1.655 | 0.041/0.131/0.022 | 0.965/0.867 | close to AR(10) |
| AR(20) | 1.549 | 1.465 | 0.035/0.117/0.021 | 0.964/0.859 | not better than AR(10) |
| seed 2 | 1.257 | 1.386 | 0.034/0.111/0.026 | 0.966/0.854 | seed spread |
| seed 3 | 1.252 | 1.277 | 0.033/0.110/0.026 | 0.959/0.846 | seed spread |

**Supported:** rescale is necessary (removing it collapses T/partition/RDF/VDOS/D); memory is
necessary for water (AR(1) is 2.5× too diffusive and structurally worse; AR(10) is a reasonable
choice); FD noise is not necessary for D but is necessary for the KE variance (partition std
0.002 vs 0.253); the 0.1 translation blend is better than 0.3/0.5.

**Not supported:** a full mode-decomposition necessity claim (the internal-mode 1.0 ablation is
missing); a precise 22–25% D deficit (the 100 ps seed spread is 1.26–1.71×10⁻⁹, ~27%, comparable
to the deficit).

## 6. Five questions

**Q1. What is the official TrajCast 2.5 ns really doing?**
Pending. The 100 ps control is strong on temperature/total KE and O VDOS; it has a ~5.6%
translation-KE excess, ~6.3% internal-KE deficit, O–O RDF RMSE 0.043, and H VDOS overlap 0.624.
These are 100 ps numbers; the 2.5 ns run is still in progress and must not be pre-empted.

**Q2. Does GLR need extra long-time reference statistics?**
Yes, but calibration *length* is not the main bottleneck. The 25-block reference ACF integral is
14.45 ± 1.86 fs. The production Yule–Walker fit uses only **0–50 fs**; independent 100 ps and
200 ps fits give the same AR(10) (radius 0.853–0.855) and the same 2.3–2.5× integral
overprediction. A longer calibration trajectory alone will not fix D; a longer fit window and a
D-constrained fit are needed. The readout needs no reference forces at inference; extra MD is
offline only.

**Q3. Does the new GLE improve full autonomous dynamics, not just the ACF fit?**
Not tested. The offline gate failed: the 5-mode generalized kernel matches D (+1.0%) and the
long-lag ACF (RMSE 0.047) but misses 5 fs by 0.255; the unconstrained 3-mode matches 5 fs but
has 1.90× the discrete D. The exact discrete implementation is verified (Monte Carlo RMSE 0.0025),
so this is a **kernel-identification/sampling** failure, not an implementation or coupling result.
No 100 ps GLE rollout was authorized or started.

**Q4. What is the ML contribution versus existing methods?**
Not mode decomposition, AR, or FDT by themselves. The contribution is a **training-free,
mode-resolved readout on a frozen neural propagator**: it imposes reference-NVT mode partition,
RDF/VDOS, and time correlations without retraining the backbone, and it transfers across
backbones (v0 + GLR) and materials (water; quartz 5 fs phonon bands). The diagnostic — pointwise
one-step accuracy is neither necessary nor sufficient for long-time ensemble fidelity; NVE energy
stability is a red herring for NVT — is the paper's conceptual contribution. The GLR is
"training-free for the backbone" but **not data-free**: it needs an offline reference NVT ACF.
The new GLE attempt does not add a new ML architecture; it is a physics-based readout whose
5 fs kernel is not identifiable. The GLR is a **closed-loop velocity-feedback readout**: the
current-step displacement is the backbone displacement, but `v_new` is the next state and drives
future backbone calls; a one-step probe changes the next displacement by ~62% from a ~21%
velocity correction. The local `q_{t+1}-q_t = h v_new` equality does not hold, but that is a
kinematic discretization issue, not proof of no closed-loop dynamics. D is 22–25% low; the AR(10)
is a 0–50 fs local fit and a candidate contributor, not a proven sole cause. The paper should
present it as a closed-loop readout with explicit limitations, not a post-hoc correction or a
SOTA dynamical model.

**Q5. What is ready for the ICML main text, and what is not?**
Ready: the pointwise-vs-ensemble diagnostic; the published TrajCast paradox and the 250k
counterexample; NVE energy stability is a red herring; deterministic GLR mean partition/RDF/VDOS
at 2.5 ns (two starts); backbone-agnostic v0 + GLR 100 ps; quartz 5 fs phonon-band stochastic
rescale (two seeds); component ablations (rescale necessary, FD noise not necessary at λ=0.1,
memory needed for water but not quartz 5 fs); the new reference-ACF/D block-uncertainty and
calibration-window diagnostic; the position-velocity audit and the independent 100/200 ps
calibration reproducibility.
Not ready: exact D recovery or a bare "25% low" claim without full-reference uncertainty and a
recalibrated readout; full KE variance for deterministic GLR; the "stochastic GLR breaks GK/MSD"
claim as a systematic 2.5 ns statement (the 8-block ratio is 1.06); a local
`q_{t+1}-q_t = h v_new` equality (the loop is closed through next-state feedback, but the
current displacement is the backbone displacement); a 0–50 fs AR fit that predicts the
autonomous D; a one-rule cross-material claim; the official TrajCast 2.5 ns
(pending); a converged native TrajCast (ep145/1500 is not a fair baseline); 2.5 ns quartz;
external action-map/FlashMD comparisons; a clean held-out reference test; and the new GLE
(offline gate failed).

## 7. Claim boundary

- Do **not** claim FDT alone guarantees Gibbs invariance when composed with frozen R3-Opt. The
  composed process was not validated; the GLE gate failed before coupling.
- Do **not** describe deterministic GLR mean-KE recovery as full equilibrium statistical
  recovery. Its KE variance is suppressed by ~150× in translation.
- Do **not** use the incomplete native TrajCast ep145/1500 as a fair external baseline.
- Do **not** claim mode decomposition, AR, or FDT as novel by themselves.
- Do **not** claim the GLE improves autonomous dynamics; it was not run.
- Do **not** claim the current-step displacement equals `h v_new`: the current displacement is
  the backbone displacement, but `v_new` is the next state and feeds the next backbone call. The
  loop is closed; the local `q–v` equality is a kinematic discretization issue, not proof of no
  closed-loop dynamics.
- Do **not** claim stochastic GLR systematically breaks GK/MSD over 2.5 ns: the 8-block ratio is
  1.06, within the reference block variability. Do report its larger kinematic residual (3.8%).
- Do **not** claim the AR(10) offline ACF predicts autonomous D: it uses 0–50 fs, overpredicts the
  integral by 2.3–2.5×, while the coupled rollout underpredicts D.
- The official TrajCast 2.5 ns result is pending; the 100 ps control is context only.

## 8. Deliberately not run

- No 100 ps GLE rollout (offline gate failed).
- No new rescale, kernel family, mode count, or backbone variant after the gate failure.
- No 2.5 ns GLE rollout.
- No native TrajCast retraining; ep145/1500 is not a fair baseline.
- No 2.5 ns quartz; no external action-map/FlashMD comparison.

## 9. Provenance

`MANIFEST.json` records every copied file with its original path, size, and SHA256.
`CHECKSUMS.sha256` is a `sha256sum -c`-compatible list. The snapshot is a local git repository
on branch `research/glr-gle-20261127`, tag `research-snapshot-20261127`. Large trajectories are
referenced, not copied.
