# TrajCast-based foundation: scope and implementation review

This is a new route, not a continuation of `neural_segment`. All previous code,
checkpoints, data and negative results remain unchanged. No Motion-Mode Adapter
or segment network is trained by this directory.

## Immutable upstream and licensing

- IBM repository: https://github.com/IBM/trajcast
- Commit: `c52f107a02176671b21a420b2d4c67e0327b04f7` (2025-09-23).
- Models: `ibm-research/trajcast.models-arxiv2025`, revision
  `621f5933a1020c68ab7f9bc5071b1362499bbfa7`.
- Data: `ibm-research/trajcast.datasets-arxiv2025`, revision
  `71cdd22807033010bd0d7a852b84520763822e44`.
- Source LICENSE and both Hugging Face cards specify Apache-2.0. Model/data
  repositories have a card declaration, not a separate LICENSE file. Preserve
  cards, upstream authors, LICENSE, and provenance. Do not imply IBM endorsement.
- `acquire.py` validates every Quartz file against upstream LFS SHA-256 or Git
  blob SHA-1; historical local bytes are accepted only after this verification.
  The entire official source is cloned outside this Git repository, untouched.
- Citation: Fabian L. Thiemann et al., *Force-free molecular dynamics through
  autoregressive equivariant networks*, Nature Machine Intelligence 8,
  764–776 (2026), https://doi.org/10.1038/s42256-026-01227-7.
- Upstream also acknowledges MACE checkpoint and NequIP-derived utilities;
  these acknowledgements remain in the original source. We do not relabel those
  components as SMDP contributions.

## Source-level interfaces

All paths below are relative to the pinned upstream repository.

| Interface | Observed behavior | Integration decision |
|---|---|---|
| `data/dataset.py:AtomicGraphDataset` | ASE → dictionary → graph; optional time reversal; PyG collated cache | Use unchanged official cache contract; independently verify all converted 16 fs samples |
| `data/atomic_graph.py:AtomicGraph.from_atoms_dict` | ASE periodic neighbor construction with explicit image shifts | Preserve official PBC implementation |
| `AtomicGraph.update_edge_index` | `torch_nl` neighbor rebuild using current predicted coordinates | Keep original Forecast calls, no cached true geometry |
| `model/models.py:EfficientTrajCastModel` | velocity norm/SH embedding; four velocity-conditioned equivariant MP blocks; compression and readout | Use official Quartz YAML and strict checkpoint loading |
| `_encoding.Normalization` | In-place normalization of target fields, not input velocities | Clone references before forward; never normalize twice |
| `_int_out.mp_layer_3` → `LinearCompression` | Last node-feature representation before endpoint readout | Future adapter insertion point, exact irreps exported by interface check |
| `nn/modules.py:ConservationLayer` | Mass-weighted momentum correction; separate Batch handling | Preserve, test single/batched agreement and momentum numerically |
| `model/training.py:Trainer` | Training-only RMS scaling, original losses/optimizer/scheduler/checkpoint loop | Invoke upstream Trainer, not a rewritten SMDP training loop |
| `model/forecast.py:Forecast._make_timestep` | rescale displacement, wrap positions, replace velocity, CSVR, neighbor refresh | Call upstream Forecast, no substitute propagator |
| `forecast_tools/_thermostat.py:CSVRThermostat` | stochastic velocity rescaling using DOF and damping | Keep official implementation and parameters |

The network outputs **displacement and final velocity**, not velocity increment.
Raw model outputs are normalized; `rms_targets` must be applied once. Length is
Å, time fs, velocity Å/fs, mass u (g/mol numerically), and cell vectors are rows.
ASE native `get_velocities()` uses different time units: this integration reads
the explicitly named `velocities` array, as upstream does.

## Physical protocol and differences that must not be hidden

Official checkpoint: 162 atoms, Si54O108, 30 fs, float64, four layers, lmax=2,
64 channels, 4.5 Å cutoff; O→0, Si→1. Momentum conservation fixes linear
momentum; angular momentum conservation is disabled for the periodic crystal.
Forecast uses 300 K, 483 DOF, official Gaussian initialization, and CSVR damping
75 fs. Initial angular momentum is zeroed as described in Methods, but is not
constrained during periodic propagation. Production target in the paper is
500,000 steps / 15 ns, with 50,000 preceding thermalization steps.

The initial 100.02 ps validation run has 10.02 ps warmup and is explicitly
**shorter than the published protocol**. It can validate numerical behavior and
preliminary physical statistics, not establish 15 ns reproduction. The CLI
supports the full unchanged 50,000 + 500,000 step protocol separately.

Methods identifies BKS with truncated Coulomb via Carré et al.; the official
repository does not supply the original Quartz LAMMPS input deck. Our preserved
reference uses shifted Buckingham (5.5 Å) plus `coul/dsf 0.0 10.17`, Si/O charges
2.4/-1.2. Full parameters remain in `physics/bks.py` and the data metadata.
These are not silently changed. The DSF self-energy constant must be
distinguished from the pair-potential energy convention when comparing Fig. 3.
Energy-distribution agreement is evidence to check, not assumed from the name
"BKS". Missing original input-deck evidence remains a reproducibility limitation.

Official input cell starts at
`[[14.7402,0,0],[-4.9134,8.510258,0],[0,0,16.2156]]` Å.
Our independent train cell starts at approximately
`[[15.11329155,0,0],[-5.03776385,8.72566294,0],[0,0,16.62603563]]` Å.
Our MD Si mass is 28.0855 u; default ASE Si mass in official extxyz is 28.085 u.
These differences are recorded and preserved. Official 30 fs verification uses
official data; the 16 fs baseline uses our existing independent split and masses.
The latter is **not** a reproduction of the paper's exact training dataset.

## Independent dataset adapter

`adapt_data.py` reads all 5000/1250/1000 endpoint pairs and original NVE source
indices. It verifies each endpoint against the corresponding raw trajectory,
cell, atom identities/order, masses, 16 fs separation, and momentum units.
Displacement is calculated from continuous unwrapped endpoints; only input
positions are wrapped. There is no minimum-image alteration of labels.
It writes official `(Data, slices)` caches directly, avoiding extxyz decimal
rounding and loss of the MD masses. No source data is modified.

The official Trainer consumes those caches with its original normalization,
batch handling and loss. Train-only normalization; trajectory-disjoint validation
and test. Adapter access to test data checks representation integrity only, not
model selection. No new BKS simulation is required.

## Original 16 fs training configuration

Same published Quartz architecture; float64; batch size 2; Adam/AMSGrad at 0.01;
ReduceLROnPlateau factor 0.8/patience 20; maximum 1500 epochs; equal normalized
displacement and final-velocity MSE. The released `Trainer` lacks the paper's
100-epoch early stop; our logging callback adds only that stopping condition.
No optimizer, model or Forecast source changes. No pretrained 30 fs output head
is added to another model; original16 starts from the upstream initialization.
Training is blocked until official30 validation is reviewed and recorded.

## Future modifications (not implemented in this phase)

Three separate versions will be Original / +Modes / +Modes+Segment. Adapter
output must be zero-initialized and preserve last-layer irreps; endpoint equality
at initialization needs a numerical test before training. Keep original readout
and ConservationLayer. Segment corrections must vanish with their first
derivatives at both endpoints; query times share a single graph encoding. These
requirements do not license replacing the training loop, trunk or Forecast.

## Environment and result status

New isolated venv: `/data1/wyd/envs/trajcast-c52f107`, no system-site packages.
PyTorch 2.5.1+cu121; e3nn 0.5.9 (0.5.4 is unavailable on the package index; this
meets upstream >=0.5.4). Exact resolved dependencies saved after installation.
Previous PyTorch/e3nn environments are unchanged. BKS offline analysis runs in
the preserved MPICH environment, separate from neural inference.

Only generated JSON and trajectory artifacts constitute results. Loading a
checkpoint, passing interface checks or preparing a YAML is not a successful
physical reproduction. `reports/trajcast_foundation_20261027` records observed
completion and outstanding work; no adapter training is automatically launched.
