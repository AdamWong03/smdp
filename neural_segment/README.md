# Neural segment SMDP — frozen experiment 2026-10-26

Scope: R2/R3-B scalar/vector encoder, independent learned tokens, shared
mode-conditioned recurrent evolution, mid-evolution coupling, one temporal
operator predicting shifted-Legendre acceleration coefficients. Exact
polynomial integration gives q and v from the same coefficients. No BKS,
MLIP, learned potential gradient, harmonic modes, or physical integrator is
called in network inference. SACMP is preserved but is not used here.

## Architecture and controls

`model.py` reuses the 4-layer 128-scalar/32-vector encoder, cutoff 4.5 A.
Four token-specific scalar/vector projections and embeddings have no
competitive softmax allocation. Recurrent weights are shared and conditioned
on each token's evolving state. Two updates precede coupling and two follow
it. Off-diagonal atom-local cross-mode exchange uses the existing gated
equivariant implementation. Temporal readout produces one set of coefficients
per atom, mode and basis. Every query reuses it. Per-graph mass projection is
applied to the coefficients, not independently to q and v.

Controls share the same encoder, decoder, data and supervision:

- Full: four tokens with coupling.
- Independent: four tokens, token-local residual update in place of exchange.
- Direct: no mode decomposition; one wider (192 scalar / 48 vector) latent
  state with recurrent updates and a local residual. It still predicts a
  segment directly, not a numerical integration trajectory.

Parameter/FLOP budgets are approximate, not equal. Measured initial counts:
Direct 1,003,680 / Independent 864,448 / Full 914,498. Profiled operator FLOPs:
2.70 / 3.02 / 3.08 billion on a 162-atom input. Report both costs and accuracy.

## Frozen pretraining checks

All 5,000 TRAIN segments are used for normalized q/v least-squares fits.
Thresholds fixed before computing fit: q RMSE <= 0.00025 A and v RMSE <=
0.0000343 A/fs (~one tenth of R3-B endpoint error). Six basis functions give
6.09e-5 A / 6.00e-6 A/fs; eight give 6.03e-5 A / 5.96e-6 A/fs. Choose **six**.
These are representation floors, not neural prediction results.

Tests check integrated polynomial identities, exact initial states, d(q)/dt=v,
per-system momentum, E(3) including reflections, unwrapped PBC invariance,
batch separation, gradient reachability, endpoint/dense equality and one GNN
encoding for all query times. Initial fixed-graph GPU costs are not production
ns/day; `benchmark_r3.py` later includes graph rebuilding on the same GPU.

`labels.py` verifies the established BKS/LAMMPS implementation against stored
forces, raw energy gradients, velocity-Verlet consistency and 16 fs replay.
Original dumps have ~6 significant digits and float32 coordinate storage.
The first strict test based on position second differences failed because it
amplifies rounding. Its report is preserved. Corrected checks use stored
velocity/force Verlet consistency and float64 energy finite differences.
Allow 0.1% RMS force difference for coordinate rounding; never change BKS.
Actual errors: 0.06665% recomputed force RMS, 1.44e-5 relative velocity Verlet,
8.65e-8 eV/A energy derivative, 1.48e-5 A / 1.04e-6 A/fs short replay.

## Training and on-policy labels

Only existing independent Quartz train/val/test splits are used. Original
training uses every one of 5,000 starts and all actual 1..16 fs states.
LAMMPS force labels at 0/4/8/12/16 fs are evaluated at original stored
coordinates; units are converted by `physics.units.FORCE_TO_ACCEL`.
Loss = normalized q MSE + normalized v MSE + 0.1 normalized a MSE, scales
computed only on original training. No other regularization objective.

Stage 0: at most 80 epochs, batch 4, AdamW 3e-4. Stages 1/2: at most 20
epochs each, 1e-4, 70:30 weighted original/new samples, 5,000 draws per epoch.
Same seeds, schedules and budgets for all variants. Plateau LR reduction;
early stop after 12 stale epochs, at least 20 epochs. Selection uses all
1,250 validation segments plus 0.25 * normalized 256 fs error on first 128
valid validation starts. No test selection.

Each on-policy round: fixed random 500 TRAIN starts; model visits up to
256 fs. Candidate states at 64/128/192/256 fs, at most 2,000 labels. Check
every visited state for nonfinite values, T outside 20..2000 K or distance
below 0.8 A. Rejected states and their source/checkpoint are retained.
Accepted states are independently integrated for 16 fs with ASE's established
velocity-Verlet, 1 fs, using the existing LAMMPS BKS calculator; they are
never assigned a displaced original trajectory's future. Original and new
q/v/a are trained with the identical fixed scales and loss. No OOD label
generation is permitted from validation/test starts.

## Validation and production gate (fixed before training)

Report all 1..16 fs and 32..256 fs errors, mode contributions, same-GPU
endpoint/dense costs. Two 10 ps validation starts `val_07/val_08` must both
complete and satisfy mean T error <=30 K, BKS energy-change RMS <=1 eV,
minimum distance >=1.2 A. These are screening thresholds, not proof of
physical validity. Failed screens remain results; do not tune gates afterward.

Only passing models freeze stage2/best.pt SHA256, then use the 1,000 tests,
two 100 ps test NVE trajectories and 1.5 ns heat-up + 15 ns CSVR NVT.
Existing thermostat and offline physics definitions are unchanged. Failed
test rollouts stop further long production for that model, not model selection.

## Run

From repository root, using the existing environment:

```bash
export CUDA_VISIBLE_DEVICES=GPU-e021d5a1-42fd-70cc-8dd1-08de8f1d8987
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
/data1/wyd/envs/mlip/bin/python -u -m neural_segment.run \
  --data-root /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1 \
  --output /data1/wyd/research/foundation_mlip_adaptation/experiments/neural_segment/quartz16_20261026_v1
```

Raw labels, checkpoints and trajectories stay on the server. Completed stages
are reused; partial training/rollouts are not overwritten automatically.
No new route, expert, physical correction, or adaptive loss is authorized.
