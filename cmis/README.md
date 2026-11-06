# CMIS — Coupled Modal Impulse Segments (research prototype)

This is an **independently implemented, working PyTorch core**, not a trained or validated molecular simulator. It was created to test a particular research hypothesis: state-conditioned, cross-coupled latent *impulse modes* can generate accurate finite-stride phase-space trajectories across materially different systems.

## Model and mathematical contract

Input: unwrapped q `[N,3]` in Å; v `[N,3]` in Å/fs; atomic numbers `[N]`, masses `[N]` in amu; periodic cell with row lattice vectors; time queries τ ∈ [0,h] in fs. Each forward makes a new periodic full-directed graph at **q0** and at **predicted q(h/2)**. The graph adapter returns `(edge_index[2,E],edge_shift[E,3])`, with `edge_index[0]` sender, `edge_index[1]` receiver and `dr=q[src]+shift-q[dst]`.

Structure:

1. Invariant mass/velocity- and neighborhood-conditioned `ScalarEncoder` (three GNN layers by default).
2. `ModeCoupling`: `K=4` trainable mode seeds plus atom-conditioned mode states. Cross-mode attention exchanges information **before** and **after** geometry propagation. There is *no competing softmax allocation* of the state into modes.
3. `PairModeStep`: each undirected image-pair produces equal and opposite vector impulse amplitudes `[E,K,3 temporal bases,3 vector coords]`. The vector bases are the edge direction and relative-velocity tangential component. Each mode learns a local angular frequency, initialized to a broad range but *not fixed*.
4. Generate a candidate midpoint by analytically integrating the first impulse field. Rebuild the periodic graph at that *predicted* midpoint and update/couple modes again. The second step emits a small learned correction impulse field on the new graph.
5. The final phase-space trajectory is the exact analytic integration of the sum of both impulse fields, queried at arbitrary times within the segment **without rerunning graph/message-passing computations**.

For each pair e=(i,j), mode k, and local time s=τ/h:

`dp_i/ds = +Σ_{e incident to i} Σ_k J_{e,k}(s)` and `dp_j/ds = -Σ_k J_{e,k}(s)`,

`J_{e,k}(s) = A^0_{e,k} + A^c_{e,k} cos(ω_{e,k} h s) + A^s_{e,k} sin(ω_{e,k} h s)`.

Here each A is a learned **vector-valued impulse amplitude** with units amu·Å/fs. The sign convention for each pair is purely an implementation choice; only antisymmetry is physical. The model computes `p(τ)` and `q(τ)` by analytic integration from `p0=m*v0`, so `∂q(τ)/∂τ = v(τ)` at every query time. Momentums exchange pairwise; total linear momentum is conserved and COM travels at constant velocity per disconnected system.

**Not guaranteed**: energy conservation, angular momentum conservation, reversibility, semigroup composition across different macrosteps, symplecticity, any particular mode semantics, or thermodynamic fidelity. PBC position reimaging and arbitrary graph-size scale have separate conditions.

Mode frequencies are state dependent. The initial mode logit biases correspond roughly to angular frequencies between 0.03 and 0.7 rad/fs to avoid identical mode initialization. These are *trainable initializations*, not fixed species-specific vibrations or claims of learned phonon modes. The current allowed angular frequencies are in `(0.005,1.05)` rad/fs; this is a fixed operational range and must be assessed for a dataset's fastest modes.

## Quick verification

```bash
python -m py_compile cmis_model.py vesin_adapter.py train_cmis.py rollout_nve.py
OMP_NUM_THREADS=1 python test_cmis.py
```

`test_cmis.py` contains 13 checks callable with `pytest -q test_cmis.py`. They cover exact t=0 identity, COM/momentum conservation, translation/rotation/Galilean covariance, image reimaging, analytic trajectory derivative, mode and frequency gradient paths, no-edge free flight, atom permutation, query independence and a frozen-mode intervention. These are mathematical/software checks **only**.

A batch-training smoke test can be generated without `vesin`:

```bash
python make_toy_data.py
python train_cmis.py --train toy_data/toy_train.npz --val toy_data/toy_val.npz \
  --out toy_data/smoke --horizon 5 --device cpu --width 24 --epochs 2 \
  --batch-size 4 --debug-small-cell
```

The toy sim is a manually constructed exact two-particle harmonic oscillation; it is **not** evidence of long-range or physically realistic MD performance.

## Formal dataset integration

The *existing* independent NVE splits in the user's `smdp_reference_md_v1` repository should be retained. Do NOT guess raw HDF5 schema. Inspect the existing `train_quartz.py` and `train_relational.py` readers; convert only the selected contiguous TRAIN and VAL windows to NPZ files with these keys:

- `q0: float32[S,N,3]` (unwrapped Å)
- `v0: float32[S,N,3]` (Å/fs)
- `q_target, v_target: float32[S,T,N,3]` at every real `times_fs` query
- `times_fs: float32[T]`, **uniformly spaced and final time = horizon** (required for the provided time-reversal augmentation)
- `z: int64[N]`, `mass: float32[N]`, `cell: float32[3,3]` or `[S,3,3]`

The conversion must verify atom ordering, units, independent trajectories, exact raw-frame indices and PBC shifts. Separate systems (Water and Quartz) should have separate training/validation files and **separately trained weights**; this is a shared architecture, not a claim of zero-shot cross-element transfer.

Recommended initial protocols, conditioned on actual stored cadence:
- **Water**: 5fs macrostep, times `[1,2,3,4,5]`fs (or use `[0.5,1,1.5,...,5]`fs if exact 0.5fs snapshots are available and budget permits).
- **Quartz**: 30fs macrostep, times `[3,6,9,...,30]`fs (ten exact supervision points); if raw data only supports a subset, use that explicitly and retain true uniform spacing.

Run:

```bash
python train_cmis.py --train water_train.npz --val water_val.npz \
  --out outputs/cmis_water5 --horizon 5 --epochs 350 --device cuda --batch-size 8
python train_cmis.py --train quartz_train.npz --val quartz_val.npz \
  --out outputs/cmis_quartz30 --horizon 30 --epochs 350 --device cuda --batch-size 8
```

`train_cmis.py` does true B-sample disjoint-graph batching, normalized **mass-weighted state supervision** with `train`-split-only scales, time-reversed NVE examples, gradient clipping, reproducible seed and validation-selected best checkpoint. The learning rate is 2e-4 and the endpoint has weight 2. It is a baseline training scaffold, **not** a fully tuned official training protocol. Adapt LR schedule, validation stage, larger runs and mixed precision only after checking the existing server environment.

The formal comparison must report all data usage, including intermediate-frame supervision: endpoints-only baselines and segment-supervised CMIS are **not** identical target budgets. Train a target-budget-matched state predictor or an endpoint-only CMIS control to quantify that advantage.

## Autonomous evaluation

`rollout_nve.py` accepts a checkpoint and initial-state NPZ, streams state trajectories to HDF5 and stops on nonfinite states or unsafe temperatures. It does not compute potential energy. Use the original project's verified BKS / SPC-Fw offline force-field evaluator to assess total energy, RDF, VDOS, phase statistics and water COM MSD. For NVT, integrate with the **existing audited CSVR implementation** rather than duplicating thermostat logic from scratch.

The provided rollout code is for a **single system**, and must be profiled at the target N and GPU: the stage-2 midpoint graph and `E × K × width` memory are not free. `small_cell_periodic_graph()` is O(N²) and MUST NOT be used for production; `vesin_adapter.py` is required there.

## Research comparison and honest novelty boundary

- **TrajCast**: direct equivariant next-state prediction, excellent reference-MD agreement on water and quartz.
- **FlashMD**: PET spatial network, time-strided position/momentum prediction, explicit physical constraints and extensive physical diagnostics; do not claim pairwise impulses or COM preservation as our novelty.
- **DYNAMI-CAL GraphNet (Nature Communications 2026)**: pairwise antisymmetric linear/angular impulses and spatiotemporal graph updates **already published**. Pairwise impulse conservation alone is neither new nor sufficient evidence.
- **MDtrajNet**: differentiable time-conditioned coordinate map and velocity as a derivative.
- **EGNO / GF-NODE / DoMiNO**: graph-time or spectral evolution, mostly evaluated using future-coordinate errors rather than long-duration equilibrium MD.

CMIS's **testable** contribution is the joint operation of: learned local temporal spectra of impulse modes, geometry-conditioned inter-mode exchange, and an analytically integrated, dense-query phase-space segment. This is a hypothesis, not a verified novelty claim or a physics result.

## Main experimental stopping conditions

1. Unit/graph tests and toy smoke must pass before formal MD.
2. Fix the dataset splits, reference evaluator and checkpoint protocol before training.
3. First evaluate validation short NVE, H/O and Si/O kinetic distributions, segment derivative consistency, per-mode effective-rank and Fourier/VDOS behavior. Failed runs are recorded; do not adjust test conditions.
4. Water must overcome prior near-zero molecular COM diffusion, pathological RDF peaks and translational/rotational/internal KE partition; Quartz must retain previous R3-Opt long-time VDOS/RDF and NVE diagnostics. One is not sufficient.
5. External strong comparison: official TrajCast under matched reference protocol, and FlashMD/MDtrajNet only where their model/benchmark objectives and force-field references are genuinely comparable.
6. If a model performs well only under a much stronger thermostat, that does not establish force-free NVE fidelity. If mode interventions do not reveal distinct dynamics or causal utility, do not claim learned mode specialisation.