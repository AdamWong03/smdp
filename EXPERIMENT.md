# First formal quartz experiment

## Frozen implementation

This integrates the repository's `smdp_neural.py`, not the older server-side
two-stream model. Four per-atom motion tokens share recurrent and temporal
operators through mode-conditioned routing, with cross-mode attention between
two evolution stages. The encoder runs once for all requested times. Both
operators are evaluated under soft routing; routing is not compute skipping.

Necessary integration changes: graph-specific COM corrections in disconnected
batches; Vesin linked-cell periodic edges including multiple images and skew
cells; smooth cutoff on complete messages and weighted-degree normalization
(cutoff on RBF inputs alone leaves a message/degree discontinuity);
source-indexed true intermediate targets; full segment and autonomous
training; explicit failure handling and streamed trajectory output. No experts,
physics modes, gate regularizers, forces or implicit solvers were introduced.

## Data and units

Server dataset: `/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1`.
Use quartz `transitions/16fs/{train,val,test}.h5` source trajectory/frame indices.
Counts: 5000 / 1250 / 1000 from 6 / 2 / 2 independent NVE trajectories. Raw
cadence is 1 fs. Coordinates are unwrapped Angstrom, velocities Angstrom/fs,
masses u, and cells have lattice vectors in rows. No test coordinates are
loaded during training. The old action hypothesis in dataset metadata is
historical text, not the model architecture used here. Original data is unchanged.

Cache contains exact frames 0..16, 32,48 relative to each existing sample, not
interpolation. Endpoints are checked against the original transition files.
Starts too close to a trajectory's end remain in single-segment training but
are excluded from 3-step rollout training. Loss scales come from train only.

## Training and fixed evaluation

Default model dimensions remain 128 scalar / 32 vector channels, four encoder
layers and four motion tokens. Seed 20261022, float32, batch 8, AdamW, initial
LR 2e-4. Up to 100 segment epochs followed by up to 24 rollout epochs at 5e-5;
validation plateau scheduling and patience 15 (minimum 20 epochs per stage).
Each training query set is 4,8,16 fs plus two random distinct internal integer
times. Validation uses fixed 1,4,8,12,16 fs. Endpoint loss multiplier is two.
Rollout training averages segment loss with t+32/t+48 autonomous loss, without
detaching recurrent states. Both stages traverse all eligible formal samples.
Validation chooses `best_segment.pt`, then `best_rollout.pt`; test is evaluated
only after both stages complete. Test evaluation queries every integer fs.

NVE: two independent test starts, up to 100 ps each (6250 steps).
NVT: independent long-reference initial state, 1.5 ns thermalization followed
by a target 15 ns production, matching reference durations. CSVR target 300 K,
tau 100 fs, 3N-3 internal DOF, conserved initial COM, seed 2026102201. CSVR is
applied at 16 fs endpoints; this differs numerically from reference 1 fs updates.
Both pre-thermostat temperature and post-thermostat states are recorded.
Stop on nonfinite states, pre-thermostat T outside 20..2000 K, or minimum distance
below 0.8 A. These are failure stops, not trajectory repairs. Save the failed
state. Do not restart with a different thermostat or architecture.

Each call queries 1..16 fs. Store macro positions/velocities and dense velocities
in chunked HDF5. Dense endpoint velocities include CSVR kicks; document their
boundary effect. BKS is used **offline only** with the same Buckingham/Wolf
parameters and kcal/mol conversion as the formal reference. Stored float32
coordinates need not reproduce the original double-precision thermo energy
bit-for-bit (observed sub-meV whole-cell offsets); comparisons recompute BKS on
both stored reference and candidate configurations using the same evaluator.

Evaluate matched 16 fs spectra (Nyquist 31.25 THz), and separately queried 1 fs
spectra. Use the retained TrajCast-compatible uncentered VACF spectrum convention
with a 2 ps correlation window and 2 ps padding, and matched 16.384 ps time
blocks. No VDOS result is reported with less than 4.096 ps available.
Energy and RDF comparisons include quantitative differences and time-block
variation. Failed thermalization diagnostics are not production statistics.

## Commands

Server checkout: `/data1/wyd/research/foundation_mlip_adaptation/smdp`.
Current run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1`.
The preceding `quartz16_20261022` run was stopped after 19 segment epochs when
the cutoff discontinuity was found. It is preserved with `STOP_REASON.json`,
is not a completed experiment, and supplied no test or autonomous results.
The corrected run restarts model/optimizer from the same fixed seed and reuses
only the exact reference-segment cache. It does not warm-start aborted weights.

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib
/data1/wyd/envs/mlip/bin/python -m pytest -q
# Complete one experiment in a new output directory (do not overwrite the current run):
/data1/wyd/envs/mlip/bin/python -u run_experiment.py \
  --data-root /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1 \
  --output /data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_reproduction
```

For the current already-running training, `run_experiment.py --after-training`
is queued with `--wait-training-pid` and the existing output path. It exits if
training fails; it never evaluates an unfinished checkpoint. After training it
runs the fixed test/NVE/NVT/analysis sequence once. No automatic model search.
The queued runner also has `--publish-summary`: after evaluation it commits only
a size-limited whitelist of metrics, configuration, plots and the generated
report under `reports/quartz16_first`. No raw logs or binary trajectories are
published. Push conflicts or unrelated local changes stop publication rather
than being overwritten. `publish_summary.log` records publication failures.

Artifacts: `config.json`, `history.jsonl`, `best_segment.pt`, `best_rollout.pt`,
`last.pt` (optimizer/scheduler/RNG included), `test/query_errors.csv`,
`test/test_metrics.json`, `nve_test_09.h5`, `nve_test_10.h5`, `nvt.h5`, corresponding
`.summary.json`, analysis directories, plots and `RESULTS.md`. Large artifacts
and raw logs stay outside Git. This document specifies the experiment; it is
not a claim that training or any requested trajectory has completed.
# Targeted additive-mode / long-rollout iteration (2026-10-22)

Branch: `improvement/additive-modes-long-rollout`. Previous artifacts remain at
`/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1`.
New artifacts use sibling `quartz16_20261022_r2_additive`.

The architecture retains four tokens and the same recurrent/temporal operators.
Only final reconstruction and cross-mode exchange change: shared equivariant
readout is applied independently per token and contributions are summed; diagonal
cross-attention is masked, with separate sigmoid-gated scalar/vector residuals.
No equal-contribution regularizer or extra propagation expert is used. Old
checkpoints load with the two new config flags absent/false.

Training from scratch: seed 20261022; batch 8; at most 100 segment epochs,
then 12 epochs each at 4/8/16 autonomous steps. Pushforward retains gradients
through the final two segments while all earlier states remain self-generated.
All 5000 endpoint samples are used; only source windows with sufficient 256 fs
continuation participate in rollout. Validation in every rollout stage uses
16 steps; `best_rollout.pt` is selected within the final stage, never on test.
Actual 1 fs reference samples supervise neighboring-query kinematics and junction
defects. Species velocity, kinetic-energy moments and VACF target reference NVE
windows, not a prescribed temperature. Weights are frozen in `OBJECTIVE_WEIGHTS`.

```bash
CUDA_VISIBLE_DEVICES=2 /data1/wyd/envs/mlip/bin/python run_experiment.py \
  --data-root /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1 \
  --output /data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive \
  --previous /data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1 \
  --publish-summary --report-name quartz16_additive_long_rollout
```

The pipeline runs one training, full test evaluation, both 100 ps NVE trajectories,
and the unchanged CSVR protocol (100 fs coupling time; 1.5 ns thermalization and
15 ns production), then publishes bounded results on the same branch. It never
automatically changes architecture or retrains. BKS is evaluated offline only.
`COMPARISON.md`/`comparison.json` report previous/current/BKS values. Trajectories,
weights, optimizer states, raw logs and caches stay on the server. A non-triggered
termination threshold is NOT a claim of physical stability.
