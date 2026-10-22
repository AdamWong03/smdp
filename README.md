# SMDP – Neural Motion-Mode Propagation

Formal quartz integration is now available. See [EXPERIMENT.md](EXPERIMENT.md)
for the frozen configuration, independent data splits, commands and server
artifact paths. `train_quartz.py`, `evaluate_quartz.py`, `rollout_quartz.py` and
`analyze_quartz.py` implement training and physical evaluation. The first formal
run is complete (see `reports/quartz16_first`): it ran 15 ns NVT but failed physical
fidelity, including NVE heating and mode concentration. The targeted additive-mode
iteration is described at the top of `EXPERIMENT.md`; implementation and passing
unit tests are not experimental evidence of long-time stability.

This package contains an **implemented and unit-checked neural model core** for the frozen research route:

`Equivariant Encoder → Learned Motion Modes → (Recurrent/Temporal operators) → Cross-Mode Exchange → Coupled Segment Decoder`.

The two operators execute per mode and are combined by a mode-state- and horizon-conditioned router; propagation proceeds in two halves with **cross-mode exchange between them**. One graph encoding yields multiple time-queried future position and velocity states. No Hessian, Jacobi charts, learned residual action, force evaluation or implicit symplectic solver is used.

## Files

- `smdp_neural.py`: independent PyTorch model with scalar/vector O(3)-equivariant messages, continuous time queries, motion tokens, two trainable propagation operators, coupling and phase-space decoder.
- `training_objectives.py`: normalized endpoint/intermediate position-velocity loss and short autoregressive rollout utilities.
- `test_neural_smdp.py`: numerical checks of nontrivial outputs under rigid translation, rotation, PBC reimaging, zero-time identity, COM conservation, differentiability and time query consistency.

## Verified locally

`python test_neural_smdp.py`

This verifies **mathematical/software invariants only**. It is not a physical MD experiment, not an external dataset training run, and not proof of long-time stability.

## Example usage

```python
import torch
from smdp_neural import NeuralSMDP, SMDPConfig, build_periodic_edges

model = NeuralSMDP(SMDPConfig(horizon_fs=16.0, additive_modes=True, exclude_self_exchange=True))
# q: [N,3] Angstrom, velocity: [N,3] Angstrom/fs
# atomic_numbers: [N] int64, masses: [N] amu, cell: [3,3] row lattice vectors
edge_index, edge_shift = build_periodic_edges(q, cell, cutoff=4.5)
out = model(q, velocity, atomic_numbers, masses, edge_index, edge_shift, [4.,8.,12.,16.])
pred_q16, pred_v16 = out['positions'][-1], out['velocities'][-1]
```

The `build_periodic_edges` reference helper is `O(N²)` and uses minimum image: **DO NOT use it for 4,374-atom production**. Integrate an efficient periodic neighbor list such as `vesin` with the project already using it. Edge `shift` is a Cartesian vector **added to the sender** so that `dr = q[src] + shift - q[dst]`; verify signs once against neighbor-list ground truth. Recompute neighbor lists during autoregressive steps.

## Project integration in `/data1/wyd/research/foundation_mlip_adaptation`

1. Keep the 212 GB `data/smdp_reference_md_v1` untouched. Read `dataset_metadata.json`, `physical_summary.json`, `long_reference_summary.json`, transition manifests and raw NVE storage schema **before** implementing the reader. No assumed field names, array layout, velocity units or PBC conventions.
2. Determine velocity units from the original MD engine metadata. Convert to **Angstrom/fs**, not Angstrom/ps, and use **unwrapped** positions for trajectory targets. Masses must be the same atom ordering as coordinates. Verify this for each training sample.
3. Build indices to existing **independent Quartz NVE train trajectories**: obtain contiguous frames for `t`, `t+4fs`, `t+8fs`, `t+12fs`, `t+16fs`. Use reference steps of 1fs; if the actual storage cadence differs, use its metadata instead of guessing. Keep original train/val/test grouping by complete independent trajectory; no overlapping windows from one trajectory across sets.
4. On the training trajectories *only*, estimate RMS scales of the **16 fs free-drift position residual** `q(t+h)-q(t)-h*v(t)` and velocity increment `v(t+h)-v(t)` and save these with the model checkpoint. For every segment apply `supervised_segment_loss`. Use actual positions/velocities at 4/8/12/16fs; never interpolate a target from endpoints.
5. Train the full model on all 5,000 Quartz 16fs train windows, validate on 1,250, report 1,000 test. Train normally to convergence using endpoint plus intermediate supervision; then fine-tune with truncated differentiable autoregressive rollout losses drawn from contiguous **training NVE**. Select checkpoints using held-out validation, not test.
6. For force-free NVE rollout, run `h=16 fs` repeatedly with **fresh** neighbor lists from the predicted positions at each new step. For NVT, port the reference-equivalent CSVR thermostat (taking care of correct h/tau and degrees of freedom; validate against the dataset metadata/TrajCast Methods). No BKS force evaluations during prediction. Offline BKS potential evaluations are allowed for physical evaluation.
7. Evaluate on test NVE (trajectory stability, temperatures, energy drift, min distances), then full 15ns 300K NVT, comparing offline BKS potential energy distribution, Si/O VDOS and Si–O/O–O RDF against the reference NVT. **There are two different VDOS claims:** (a) stride-sampled 16fs outputs, compared to MD downsampled to 16fs (Nyquist ≈1042 cm⁻¹); (b) additionally queried 1fs internal trajectory outputs, separately compared to 1fs MD, without claiming these are independent integration steps. Report both without conflating their information content.
8. Report wall-clock throughput/ns/day for real autoregressive trajectory generation (with neighbor updates, GPU synchronization, thermostat and output I/O accounted for). Record 162-atom and larger-supercell inference where practicable. The model accepts variable N and supports external scalable neighbor lists.
9. Measure mode routing distributions, per-mode branch occupancy, cross-mode attention and their variation across trajectory/time. These logs establish network operation, but are **not proof that modes correspond to named physical vibrations**.
10. Keep legacy Structured Action code and logs in place. Do not mix old PET/FlashMD/Hessian implementation into this architecture.

## Requirements and honest limits

- Package currently requires PyTorch, with optional `vesin` supplied by the project for large-scale PBC neighbor handling.
- The core has been integrated with the server's formal dataset, a batched
  periodic graph adapter, training, CSVR rollout and offline BKS evaluation.
  Completed training and physical results must be read from actual run artifacts;
  the requested 15 ns duration is not a claim of completion.
- Unit tests verify the final decoder for nonzero output weights, so equivariance checks are not trivially passing due to zero-initialized initial predictions.
- The model is a useful faithful realization of the fixed algorithm idea, but final parameter choices should be measured through full official runs rather than assumed optimal.

## Research implementation references

- Thiemann et al., *Force-free molecular dynamics through autoregressive equivariant networks*, NMI 2026: https://doi.org/10.1038/s42256-026-01227-7 ; https://github.com/IBM/trajcast
- Schütt et al., *Equivariant message passing for the prediction of tensorial properties and molecular spectra*, ICML 2021: https://proceedings.mlr.press/v139/schutt21a.html
- Xu et al., *Equivariant Graph Neural Operator for Modeling 3D Dynamics*, ICML 2024: https://proceedings.mlr.press/v235/xu24j.html
- Thompson et al., *ATOM*, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8adbebb54e63b5f82b306847861b9168-Abstract-Conference.html
- DoMiNO, *ACM TKDD*, 2026: https://doi.org/10.1145/3822365
