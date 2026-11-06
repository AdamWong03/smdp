# First formal CMIS Quartz / Water experiments

Upstream: `experiment/cmis-coupled-impulse-segments`, commit `d7e20ed`.
The user explicitly requested this new implementation and full experiments;
this supersedes the older DSRT-only freeze for this experiment branch. DSRT
weights, data and physical results are retained unchanged.

## Fixed first-run method and training

- Architecture: the upstream scalar encoder, four learned impulse modes,
  mode coupling, predicted-midpoint graph and analytic phase-space decoder.
- Width 96, three encoder layers, cutoff 4.5 Å, 20 radial channels.
- 431,827 trainable parameters; this is not parameter-matched to 2.358M DSRT.
- Random initialization, seed 20261106; no pretrained model.
- Full formal 5,000 / 1,250 / 1,000 transition split per system.
- Quartz 30 fs labels at 3,6,...,30 fs; Water 5 fs labels at 1,2,...,5 fs.
  Every query is a real raw NVE frame, never endpoint interpolation.
- Batch 8, FP32, AdamW at constant 2e-4, weight decay 1e-6, gradient clip 1.
- 350 epochs, no early stopping, 625 optimizer steps per epoch.
- Train-only mass-weighted residual scales. Uniform query weighting except
  endpoint weight 2; 50% random time-reversal augmentation of training segments.
- Checkpoint selected by validation normalized mass-weighted segment loss.
  Test data are not used for training, normalization or model selection.

These settings follow the new upstream integration document. They differ
from DSRT's endpoint-only targets, normalization, parameter count and optimizer
schedule. Raw endpoint/physical metrics can be compared, but the comparison
does not isolate a single architecture mechanism or match supervision budgets.

## Data and software integration

`cmis/formal_data.py` reads source HDF5 chunks sequentially to produce compact
selected-state caches, then opens those caches lazily in training workers.
No raw data were copied wholesale or modified. Every cached endpoint matches
the corresponding original formal sample exactly. Source trajectory sets are
disjoint across train/val/test. Interior labels are explicit source-frame offsets.

Train-only scales (position residual in Å, velocity increment in Å/fs):

| System | scale_q | scale_v |
|---|---:|---:|
| Quartz | 0.0607969654 | 0.00448408311 |
| Water | 0.0122827774 | 0.00542227888 |

Necessary software repairs preserve the model definition: package imports;
stable `(x-sin(x))/omega²` evaluation chosen by small phase `x` rather than
frequency alone; diagnostic edge-index outputs; safe checkpoint resume/logging.
The 13 original mathematical tests and three added graph/integral/reversal
tests pass. Together with the retained DSRT network regression checks, 52
checks pass. No toy fit, reduced-data training or smoke experiment was run.

## Queued evaluation and physical experiments

After all 350 training epochs, each service automatically runs:

1. Full validation segment/endpoint and 1/2/4/8-step autonomous error; real 1 fs
   query errors; element errors; mode contribution covariance/effective rank,
   frequency distributions and causal deletion of each mode.
2. A 50 ps validation NVE diagnostic using a held-out validation source.
3. Locked-checkpoint independent test evaluation with the same metrics.
4. Quartz: two 100 ps NVE starts and one 1.5 ns + 15 ns NVT run.
   Water: one 100 ps NVE and one 250 ps + 2.5 ns NVT run.
5. Original BKS/SPC-Fw offline energy and structure/spectral evaluation. Water
   includes molecular kinetic partition, thermostat exchange, COM MSD and
   diffusion; macro-state spectra and 1 fs queried spectra are reported separately.

This is the first formal comparison, not completion of the full three-replica
matrix or all ablations. Existing test sources provide two independent Quartz
NVE trajectories and one Water NVE trajectory. No new independent source is
claimed by reusing a frame from the same trajectory.

The common production harness records failure state/duration and never
continues from a rejected step. NVT uses the original CSVR protocol. Dense
velocities use neural interior queries and committed post-thermostat endpoint
velocities at segment boundaries. Dense queries share the two graph encodings
within each neural call. Energy/forces are never used during inference.

Both 10–100 ps diffusion fits and exact references remain as previously
declared; the additional 1–10 ps sensitivity fit from the CMIS specification
is reported separately rather than replacing the existing interval.

## Paths and reproduction

- Quartz run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/cmis/quartz30_20261106_r1`
- Water run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/cmis/water5_20261106_r1`
- Main logs: `train_and_experiment.log` in each run.
- Checkpoints: `best.pt` and resumable `last.pt`; full history: `history.jsonl`.
- Services: `cmis-quartz-formal-20261106.service`, `cmis-water-formal-20261106.service`.
- Physical GPUs: Quartz L40S 5, Water L40S 4, selected by UUID. GPUs are shared;
  end-to-end measurements must not be described as exclusive-GPU speedups.

From the repository root in the existing environment:

```bash
OMP_NUM_THREADS=1 /data1/wyd/envs/mlip/bin/python -m pytest -q cmis/test_cmis.py cmis/test_formal.py
bash scripts/run_cmis_formal.sh quartz
bash scripts/run_cmis_formal.sh water
```

The two last commands require choosing the intended GPU through
`CUDA_VISIBLE_DEVICES`; deployed systemd services already pin their GPU UUID.
Configs are versioned under `configs/cmis_*_20261106.json`.
