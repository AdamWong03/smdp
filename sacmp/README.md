# SACMP: Spectrally Anchored Canonical Mode Propagation

**Research status: implementation candidate; no Quartz training/rollout results are
claimed.** This implementation is separate from the previous all-neural
SMDP (`smdp_neural.py`) and does not change its checkpoints, metrics or
code. Use independent experiment directories.

## Physical question

Can a neural *segment propagator*, rather than a 1 fs force surrogate,
exploit learned near-harmonic fast motion while learning slow and mixed
dynamics in one structure-preserving phase-space map? The required
evidence is lower held-out endpoint error **and** faithful 100 ps NVE and
15 ns NVT physical statistics at a measured wall-clock advantage.

## Algorithm, mathematical specification

Use the *fixed* train-derived crystal reference x_ref, positive atomic masses
m, and the mass-weighted coordinates Q=M^(1/2)(x-x_ref-cm_shift),
P=M^(1/2)(v-v_cm). The COM drifts independently.

1. Build a 3N-by-K trainable modal frame W, initialized by PCA of
   finite-difference **accelerations on the train split alone**.
   Project out the 3 translation vectors and orthogonalize W by
   differentiable QR. W is fixed with respect to the current state
   during each call and all rollout steps.
2. Canonical decomposition:
   q_f=W^T Q, p_f=W^T P, q_s=(I-WW^T)Q, p_s=(I-WW^T)P.
3. **Exact learned-frequency harmonic phase.** Each mode has trainable
   frequency omega_k and Hamiltonian
   H_f=sum_k (p_f,k^2 + omega_k^2 q_f,k^2)/2.
   Evolve with the exact linear angle theta_k=h*omega_k:
   q'=cos(theta)q+sin(theta)p/omega,
   p'=cos(theta)p-omega*sin(theta)q.
4. **Single learned scalar potential** U_theta(q_f,q_s; reference graph),
   made from O(3)-invariant reference-bond projection/magnitude/cross
   features. It has a positive slow-on-site and slow
   bond term, a learnable fast anharmonic bond term, and a genuinely
   **mixed** term proportional to fast-bond x slow-bond invariants. A Hamiltonian kick is
   (p_f,p_s) <- (p_f,p_s)-h*grad_(q_f,q_s) U_theta.
   The same scalar energy supplies reciprocal fast/slow momentum
   transfer. Slow gradients are projected to the canonical complement.
   **Implementation invariant:** compute partial derivatives wrt independent
   clones of q_f and q_s; using the original qs=Q-Wq_f tensor lets
   autograd include a spurious q_f -> q_s ancestry term and destroys the
   reversible/symplectic property even though U is scalar.
5. The slow kinetic drift q_s <- q_s+h*p_s is an exact Hamiltonian
   subflow. A substep of duration h is
   fast_phase(h/2), kick(h/2), slow_drift(h), kick(h/2),
   fast_phase(h/2).
   At query time t, use n=ceil(|t|/4 fs) palindromic substeps.
   This is reversible and symplectic in exact arithmetic for fixed
   modal frame; it is **not** an exact flow of the external BKS system,
   and 16 fs query is not a free 16 fs integration with zero substeps.
6. Decode fast+slow coordinates simultaneously to the same (x,v)
   state at *any requested time*. No independent position or velocity
   head, projection-after-decoding, energy rescaling or original BKS
   force call occurs in the propagation kernel.

The model is **crystal-size-specific** at this stage: the trained 3N-by-K
frame and periodic reference graph depend on the exact atomic ordering
and cell. Scalar neural bond potentials are equivariant to rigid rotations
when the reference and mode frame are rotated consistently. A transfer
claim to new structures/supercells is NOT established by these Quartz
experiments. Per-step smoothness/symplecticity is established only while
the fixed reference-topology chart is valid (before bond changes or
crystal diffusion).

This is an implementation of one integrated algorithm, not a collection
of independent post-hoc stability patches. The modal basis is initialized
by **velocity-difference covariance, not by a physical Hessian**.

## Files

- `model.py`: full canonical flow, learnable basis, fast/slow and mixed neural energy.
- `prepare.py`: train-only PCA calibration, fixed periodic reference.
- `train.py`: segment training and autonomous 32/48 fs continuation training.
- `evaluate.py`: held-out 0..16 fs q/v errors; reversibility, semigroup, COM,
  kinematic and learned-H diagnostics.
- `rollout.py`: autonomously stream NVE/NVT trajectories; use the
  identical CSVR thermostat implementation and early instability guards
  from the previous experiment. HDF5 with `--dense` follows the established
  `analyze_quartz.py` interface.
- `run.py`: isolated short screens; optional full evaluation.
- `tests/test_model.py`: exact t=0, COM, reversibility, kinematic tangent,
  canonical symplectic Jacobian, optimizer-gradient reachability, and
  no-grad inference tests.

## Reproduce on existing server

Use the same project virtualenv as the previous Quartz runs. Run from the
repo root. Do **not** reuse/overwrite old experiment directories:

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib
PY=/data1/wyd/envs/mlip/bin/python
DATA=/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1
OUT=/data1/wyd/research/foundation_mlip_adaptation/experiments/sacmp/quartz_20261024_v1

$PY -m pytest sacmp/tests/test_model.py -q
$PY -m sacmp.train --data-root "$DATA" --output "$OUT" \
  --device cuda --rank 192 --batch-size 4 \
  --segment-epochs 80 --rollout-epochs 20
$PY -m sacmp.evaluate --checkpoint "$OUT/best_rollout.pt" \
  --data-root "$DATA" --output "$OUT/test" --device cuda

# Pre-registered independent test trajectories, 10ps initial screening.
$PY -m sacmp.run --data-root "$DATA" --output "$OUT" \
  --after-training --device cuda --dense

# Only after examining screens and test: full 100ps NVE + 15ns NVT.
$PY -m sacmp.run --data-root "$DATA" --output "$OUT" \
  --after-training --full --dense --device cuda
```

The orchestrator reuses completed screens and held-out evaluations in the
same output directory, but never overwrites partial or existing rollouts.
Long NVT with 16 dense queries per 16 fs step may be expensive; measure
short-screen wall time before the complete 15 ns evaluation.

An endpoint-only timing run omits `--dense` but cannot be passed as-is
to the previous `analyze_quartz.py` dense-VDOS interface. Always compare
wall time on the same hardware, same sample count, and explicitly say
whether dense 1 fs trajectories were queried.

## Frozen evaluation / failure criteria

Report against old R3B (~0.0025 Angstrom position RMSE and ~0.000343
Angstrom/fs velocity RMSE at 16 fs; ~+10.71/+8.57 eV BKS energy change
on two NVE starts; Si/O VDOS overlaps ~0.731/~0.599; 29.9 ns/day).
These are **historical reference numbers**, not SACMP results.

1. **Before a GPU experiment:** test the exact q/v and canonical Jacobian
   invariants; verify train-only anchor, distinct split source trajectories,
   and the fraction of acceleration variance captured by K=192.
   If fast acceleration capture < 0.8, the proposed fast/slow
   decomposition is inadequately resolved: record this as a structural
   negative finding, do not silently inflate K after looking at test.
2. **Held-out at 16 fs:** q and v RMSE, all 0..16 fs query errors,
   reversal and semigroup error; prevent selecting on test.
3. **Autonomous NVE:** two independent 10 ps safety screens, then 100 ps
   with identical BKS *offline* energy and no corrections;
   report drift per atom per ps and energy change, stability and elapsed time.
4. **NVT:** 300 K, CSVR tau=100 fs on macro boundaries as in previous run;
   15 ns production plus 1.5 ns thermalization (16 fs endpoint CSVR
   differs from the 1 fs reference integrator), O/Si element-resolved
   VDOS matched at 16 fs and queried 1 fs, RDF, temperature, diffusion
   or structural failures. No fabricated numbers or replacement reference.
5. **Ablations, only after primary run:** zero mixed energy coefficients,
   frozen pure-fast anharmonic term, random-orthonormal vs PCA-initialized W,
   no end-to-end W learning, and exact same time-query substep budget.
   Hold train/val/test trajectories and compute fixed; do not pick
   favorable seeds after seeing test.

**Success requires simultaneous held-out, long-run physical and speed
evidence.** Symplecticity only guarantees local geometric structure;
it cannot guarantee the BKS reference energy is conserved by an
inaccurately learned potential. Slow/fast substep resonance, force
gradient expense and crystal-specific basis remain primary risks.

## Related work and boundaries

- TrajCast (Nature Machine Intelligence 2026): direct learned segment
  dynamics/physical projections, the task baseline.
- FlashMD (NeurIPS 2025): neural finite-time MD stepping; its optional
  symplectic stepper uses implicit midpoint, unlike our explicit mode phase
  and canonical split.
- Hamiltonian Flow Maps (ICML 2026): learned finite-time mean velocity/
  force flow map, generally not an exactly symplectic map.
- Learning the Action (PRL 2026): symplectic/reversible learned action
  propagation; prior art to cite explicitly rather than claiming
  generic structure preservation as new.
- Integrable Building Blocks for Symplectic Neural Networks (2026):
  explicit action-dependent twists. We deliberately do NOT use a
  nonlinear action twist: it would make instantaneous q_dot unequal to
  physical p and invalidate the v=dx/dt condition for MD. The paper
  still delineates the existing symplectic-layer design space.

The research claim to validate is a **learned, data-anchored canonical
fast/slow modal chart with reciprocal neural inter-mode energy exchange
that produces competitive *physical* crystal dynamics at fewer neural
gradient evaluations than atomistic force-stepping**. The final claim
must be shaped by held-out results, not by the architecture diagram.
