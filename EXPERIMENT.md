# First formal quartz experiment

## Frozen implementation

This integrates the repository's `smdp_neural.py`, not the older server-side
two-stream model. Four per-atom motion tokens share recurrent and temporal
operators through mode-conditioned routing, with cross-mode attention between
two evolution stages. The encoder runs once for all requested times. Both
operators are evaluated under soft routing; routing is not compute skipping.

Necessary integration changes: graph-specific COM corrections in disconnected
batches; Vesin linked-cell periodic edges including multiple images and skew
cells; source-indexed true intermediate targets; full segment and autonomous
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
Current run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022`.

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

Artifacts: `config.json`, `history.jsonl`, `best_segment.pt`, `best_rollout.pt`,
`last.pt` (optimizer/scheduler/RNG included), `test/query_errors.csv`,
`test/test_metrics.json`, `nve_test_09.h5`, `nve_test_10.h5`, `nvt.h5`, corresponding
`.summary.json`, analysis directories, plots and `RESULTS.md`. Large artifacts
and raw logs stay outside Git. This document specifies the experiment; it is
not a claim that training or any requested trajectory has completed.
