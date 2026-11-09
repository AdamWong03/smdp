# Conditional successor learning: official execution

2026-11-09. Branch: `research/conditional-transition-learning`.
Protocol: [accepted design](../../docs/CONDITIONAL_TRANSITION_LEARNING_PROPOSAL.md).
No new architecture, inference calculator, or physical loss.

## Fixed comparison

Two systems, four arms each: `original`, `two_step`, `extra_reference`,
`autonomous`. All start from the same system-specific original R3-Opt `best.pt`;
the optimizer is reset. These are incremental learning comparisons, not final
from-scratch paper models. The historical two-step experiments are retained;
the new two-step arm is rerun on the same GPU as the other arms of its system.

Each arm uses 25,000 updates and 400,000 supervised macro-transition targets.
Two-step uses eight starts and two differentiable predictions. The other arms
use two microbatches of eight with averaged gradients before one optimizer step.
FP32, TF32 disabled; AdamW/AMSGrad, zero weight decay, clip 10, original endpoint
objective and proposal coefficient 0.02. Learning rate is cosine 3e-5 to 1e-5.
Seeds are 20261302 (Quartz) and 20261303 (Water), inherited from the existing
two-step protocol. Normalization remains the original D0 training scale.

Each new-data arm has 2,000 labels per round, two rounds. The autonomous pool
comes only from 125 source-stratified training starts, 256 neural macrosteps
maximum per start, and four states per prescribed depth bin. Actual visited
states receive their own true reference-MD successors. No thermostat, truth
replacement, validation-bank labels, or test samples enter training.

The extra-reference arm samples 4,000 distinct additional real training frames,
excluding exact D0 source/frame identities, and runs the same reference labeler.
Its round allocation is randomized before sorting the selected frame indices;
it does not deliberately assign early/late trajectory halves to different rounds.
Both new-data arms draw exactly 120,000 additional targets and 280,000 D0 targets
over training using a deterministic 4/5-new-samples-per-update pattern.

Original-only and the new-data arms retain all 5,000 D0 samples. As in the prior
two-step implementation, two end-of-source starts lack a second true endpoint
in each system and are excluded only from the two-step arm: Quartz indices
833/3333, Water 1665/1666. Thus that arm has 4,998 valid starts; all exclusions
are recorded. The 400,000 target budget is unchanged. These are not new split
assignments or data reductions for convenience.

Checkpoint comparison uses the actual final update 25,000, with step 12,500
retained for round-two collection. Validation is monitored, not used to select
another collector/final checkpoint by physical outcomes. Tests remain deferred.

## Numerical checks completed before launch

`conditional_transition/checks.py` used real formal data without a fitting run.
Both systems loaded 5,000 train and 1,250 validation graphs. Their original
reference calculators were independently verified on **train_01**, including
potential, force, units, and true 30 fs/5 fs continuation. Original integration
masses, potentials, and time steps were preserved. The previously known Quartz
Si metadata/integrator mass discrepancy (0.0005 u) is explicitly recorded.

Cached versus production periodic graph maximum prediction differences:

| System | Position (Å) | Velocity (Å/fs) | Accumulated/full-batch gradient relative difference |
|---|---:|---:|---:|
| Quartz | 9.54e-7 | 1.07e-8 | 2.57e-5 |
| Water | 9.54e-7 | 1.83e-8 | 2.88e-5 |

Two-step forward/backward were finite on formal samples. Full-batch and
accumulated objectives agreed within 1e-6. Oracle verification JSON is retained
on the server and exported alongside this report. This checks implementation,
not convergence or physical success.

## Running layout and logs

Server root:

```
/data1/wyd/research/foundation_mlip_adaptation/experiments/conditional_transition_20261109
```

Each `SYSTEM/ARM/` contains immutable `config.json`, `pipeline_status.json`,
`train.log`, `progress.json`, `history.jsonl`, `last.pt`, and retained phase
checkpoints. Data arms additionally contain `collect_round1.log`,
`collect_round2.log`, and `collection_roundR/{visited_states,labels}.h5`, oracle
verification and failure records. Files appear when their stage actually runs.

GPU 2, UUID `GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900`: Quartz, L40S.
GPU 3, UUID `GPU-9e62da9a-1b28-25c8-e1a0-e96004a11a95`: Water, Blackwell.
Both GPUs are shared with existing jobs; no other job was stopped. Initial free
memory fluctuated substantially (GPU 2 approximately 10–28 GiB). These are
throughput/cost observations under shared load, not exclusive GPU benchmarks.

There are two persistent queues per system. `method` runs autonomous, then
extra-reference; `control` runs two-step, then original-only. Each arm completes
its fixed training and validation analysis before that queue advances. Therefore
eight arms are scheduled, with at most four simultaneous worker processes; not
all eight begin optimizing at the same time. PID/queue logs live in `SYSTEM/`.
`flock` prevents duplicate queue workers. Reinvoking a queue resumes from saved
training/collection state; a partial trajectory or label pool is retained and
requires explicit examination, never silent overwrite.

Monitor, for example:

```bash
root=/data1/wyd/research/foundation_mlip_adaptation/experiments/conditional_transition_20261109
tail -f "$root/water/two_step/train.log"
cat "$root/quartz/autonomous/collection_round1/collection_progress.json"
cat "$root/water/two_step/progress.json"
```

## Post-training evaluation (scheduled, not yet results)

For every fixed-final arm: all 1,250 validation endpoints; 1/2/4/8-step prediction;
the exact same two validation NVE initial states for 20 ps; uncentered BKS/SPC-Fw
KE/PE/total-energy curves; temperature; Water molecular translation, rotation,
and internal kinetic partition. Evaluate frozen **validation-only** reference
and R3-Opt autonomous state banks for local signed energy errors. Their labels
are evaluation-only and not copied into training pools. No new full long NVT
is launched automatically and no physics result selects another checkpoint.

`evaluation_status.json` and `pipeline_status.json` separate training completion
from physical analysis. Numerical failure trajectories are retained; a completed
analysis does not imply stable/correct dynamics. No benefit is claimed yet.

## Reproduce / resume

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export PYTHONPATH=.:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=/data1/wyd/envs/mlip/bin/python
$python -m conditional_transition.setup
# Once per machine/protocol before training, using real data:
$python -m conditional_transition.checks
# Run only if the matching queue is not already active (flock also protects it).
CUDA_VISIBLE_DEVICES=GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900 bash scripts/run_conditional_transition.sh quartz method
CUDA_VISIBLE_DEVICES=GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900 bash scripts/run_conditional_transition.sh quartz control
CUDA_VISIBLE_DEVICES=GPU-9e62da9a-1b28-25c8-e1a0-e96004a11a95 bash scripts/run_conditional_transition.sh water method
CUDA_VISIBLE_DEVICES=GPU-9e62da9a-1b28-25c8-e1a0-e96004a11a95 bash scripts/run_conditional_transition.sh water control
```

Use separate terminal/background queues for concurrent commands. The setup
refuses changed configs. Optimizer/RNG/sampler state and actual update number
are restored together; round-two refresh does not restart the learning rate.
Checkpoints, HDF5 and full logs remain on the server. Small configs, numerical
checks, and progress snapshots are versioned here; future results will be added
without overwriting historical negative experiments.
