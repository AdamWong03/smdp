# Frozen dynamics attribution: reproduction and scope

Branch: `diagnosis/dynamics-mechanisms-20261109`, based on `1b81434`.
The exact inspected checkpoint paths, SHA256, selected epochs/updates,
configurations and existing physical-trajectory summaries are in `assets.json`.
Source weights/trajectories are read-only and stay on the server.

This study does not train any model or select a new checkpoint. All 12 required
checkpoints already existed: original R3-Opt, matched 25k-update R3-Opt two-step,
SMDP-R K3 stage1/stage2, and K1 stage1/stage2, each for Quartz and Water.
Both K1 primary trainings completed 250k updates and their fine tuning 25k;
the preexisting K1 Water production pipeline was still running at launch and
was not interrupted. K1 has 2.647M vs K3 2.572M parameters, not exactly equal
FLOPs. Historical R3-Opt stage1 stopped early (Quartz 231,250 updates); this
remaining training-history difference limits architectural causal claims.

## Environment and commands

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export CUDA_VISIBLE_DEVICES=GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900
export PYTHONPATH=$PWD:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/data1/wyd/envs/mlip/bin/python

# All formal training starts for label validation; all six frozen models for
# real-data PBC, graph-rebuild, endpoint identity and gradient finite differences.
$python_bin -m dynamics_diagnosis.audit --system quartz
$python_bin -m dynamics_diagnosis.audit --system water

# Fail before evaluation if the original reference continuation cannot match
# actual saved NVE frames, potential energies and forces.
bash scripts/run_dynamics_diagnosis.sh quartz
bash scripts/run_dynamics_diagnosis.sh water

# Consolidate actual completed models; partial status remains explicit.
$python_bin -m dynamics_diagnosis.summarize
$python_bin -m dynamics_diagnosis.calibration
$python_bin -m dynamics_diagnosis.prediction_arrays
$python_bin -m dynamics_diagnosis.existing
$python_bin -c 'from dynamics_diagnosis.common import inventory; inventory()'
# Replot/refresh tables without recomputing the completed frozen-state spectra:
$python_bin -m dynamics_diagnosis.summarize --reuse-spectra
```

Logs, per-model 50 ps diagnostic trajectories and evaluation-only fresh MD
labels live under
`/data1/wyd/research/foundation_mlip_adaptation/experiments/dynamics_diagnosis_20261109/`.
Each system has `run.log`, `run.pid` and an exclusive execution lock. These
new validation trajectories answer a different question from the previous
test NVE/NVT trajectories; existing production trajectories are not regenerated.

Hardware: shared L40S GPU 2; CPU LAMMPS single rank, original validated MPICH
runtime. Other research jobs were not killed or modified. FP32/TF32-disabled
frozen model inference; CPU float64 physical evaluation. Diagnostic batching
is not a production ns/day benchmark.
Independent Water models were also run concurrently using `--models` on the
same GPU after the shared K3-stage2 state bank completed. This changes wall time,
not data/weights/MD settings. Full vectors behind the CSVs are retained as
`predictions_*.h5` alongside the evaluation-only banks on the server.

## Fixed data/initial states and sample counts

Common initial states, identical across all six predictors:

- Quartz: `quartz/raw_nve/val_07/trajectory.h5` frame 0 and `val_08` frame 0.
- Water: `water/raw_nve/val_04/trajectory.h5` frames 0 and 10,000 (5 ps).
  These are different starts from **one** validation trajectory, not two
  independently generated MD sources.

All autonomous diagnostic trajectories are NVE with no reference reset, no
thermostat and no force call. They cover 50 ps so local-error windows are
0–1, 1–5, 5–20 and 20–50 ps. Summaries explicitly include 1/5/10/20 ps.
Macro steps remain 30 fs (Quartz) and 5 fs (Water). Exact Quartz window ends
can differ by up to one macro step and actual times are reported.

Each local-error window samples **32 states per start, 64 total**, without
replacement, seed 20261109. Quartz's first 1 ps contains only 34 macro states
per start; two starts are therefore necessary to obtain 64 genuine autonomous
states. No interpolated or internally queried state substitutes for an actually
visited macro state.

The shared real-reference bank contains 64 states/window from the same two
validation sources (256 total). Each model's own-state bank also has 256.
All six models additionally evaluate the **identical** K3-stage2 autonomous
bank. These paired inputs distinguish state-distribution effects from predictor
differences. The common bank is not selected based on energy outcome.

Each bank's true successor is freshly integrated from that state's own q/v,
with original LAMMPS BKS (1 fs x 30) or SPC/Fw (0.5 fs x 10), never another
source trajectory's future. These HDF5 labels are marked evaluation-only and
are not used by any trainer.

## Reference verification and units

`{system}_reference_verification.json` compares force/energy and full-macro
continuations at real frames 0, 1,000 and 20,000. Raw dump rounding gives a
nonzero verification floor; it is retained rather than calling this an exact
reproduction. Positions are Angstrom, velocities Angstrom/fs, masses u,
energies raw LAMMPS eV, row-vector periodic cell. No energy recentering.

Quartz has a preexisting converter mass discrepancy: original LAMMPS Si mass
28.085 u versus HDF5 28.0855 u. The integrator retains the original value;
reported official kinetic metrics retain HDF5 masses for compatibility. The
measured KE difference (about 4e-5 eV) is explicitly quantified, not used to
silently change either historical protocol. Water masses agree.

Real-data implementation checks compare the original cached initial graph
with Vesin rebuilding, integer cell reimaging, and two-step increment loss with
the equivalent absolute-endpoint loss. Gradients of the actual second endpoint
reach first q/v and proposal-head weights; a q-direction central finite
difference at epsilon=1e-3 Angstrom verifies their numerical magnitude.
No optimizer update is performed.

## Statistics and boundaries of inference

`energy_transition.csv` records signed KE/PE/total increments, their same-state
MD differences, signed q/v components/projections, and geometric/temperature
covariates. Cross/quadratic kinetic-error decomposition is retained. Water
adds molecular translation/rotation/internal KE changes and bond/angle values.

Temperature bins are fixed: <150, 150–225, 225–275, 275–325, 325–400, >=400 K.
Geometry bins use real-reference quartiles, independent of prediction errors.
Report strata counts and lack of covariate overlap. Block standard deviations
use four contiguous time blocks per start; correlated frames/atoms are not
claimed independent repetitions or a large-N confidence interval.
Blank CSV fields are explicitly inapplicable element/molecular columns in the
other system, or unavailable saved-reference coverage beyond its true end.
They are not filled with invented data. The reference bank, each own-state
bank and the common-state bank all contribute exactly 64 samples per window.
Unequal-duration windows are sampled equally for local comparisons; use
`local_time_weighted_energy.csv` for their time-weighted drift estimate.

NVE-window VACF/VDOS are nonstationary diagnostics. Quartz spectra are capped
at 16.6667 THz; no dense-spectrum restoration is claimed. Time reversal uses
`T F T F` with velocity sign reversal; its error is auxiliary and does not
establish or refute exact energy conservation on its own.

There is one training seed per configuration, approximately matched capacity
and unequal historical R3-Opt stage1 duration. Mechanism effects can be
supported by matched inputs/controls, not proven universally by a single run.
No new architecture, loss, correction force or test-set tuning is introduced.
