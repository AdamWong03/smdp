# Phase-space formal training: live status

Started 2026-11-20 02:12 CST. Two independent 250,000-update runs of the
3,036,224-parameter `phase_space_operator_v0` from random initialization. This
is the first formal training of the design prototype. No test split, checkpoint
selection or physical evaluation has been completed yet.

## Runs

| System | Run directory | Service | GPU | Epoch budget |
|---|---|---|---|---:|
| Water 5 fs | `experiments/phase_space_20261120/water5` | `dsrt-phase-water5-20261120` | L40S GPU 2 | 400 x 625 = 250k |
| Quartz 30 fs | `experiments/phase_space_20261120/quartz30` | `dsrt-phase-quartz30-20261120` | L40S GPU 4 | 400 x 625 = 250k |

Frozen config SHA-256:

- Water: `f7a2293d50e06f70f4b1942569f6d0f311d5abd411c9a6f92b5da7f919c77103`
- Quartz: `d5880604176b4de4a38e6a6cde57eb8da11383ea7f791cbbdcb8f53937838021`

Both runs use the complete 5,000-graph TRAIN split, FP32, batch 8,
AdamW/AMSGrad, warmup 5,000 to 3e-4 then cosine to 1e-5, proposal weight 0.02,
gradient clip 10, and validation-only model selection (`selection_loss`). The
TRAIN-only normalization is stored in `metadata.json` and in every checkpoint
config. Test split is read only after validation selection ends.

## First recorded epochs

Snapshot 2026-11-20 02:15 CST; these are early optimization values, not results.

| Run | Epoch | Train seconds | Samples/s | Peak CUDA | Val position RMSE | Val velocity RMSE | Selection loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Water | 1 | 133.4 | 37.5 | 4.18 GB | 0.026258 Å | 0.006482 Å/fs | 0.2364 |
| Quartz | 3 | 48.8 | 102.5 | 1.51 GB | 0.007761 Å | 0.000669 Å/fs | 0.0231 |

## Monitoring

```bash
systemctl --user status dsrt-phase-water5-20261120
systemctl --user status dsrt-phase-quartz30-20261120

tail -f experiments/phase_space_20261120/water5/history.jsonl
tail -f experiments/phase_space_20261120/quartz30/history.jsonl
```

The trainer writes `status.json` after every epoch, `best.pt` on validation
improvement and `last.pt` atomically. On `SIGTERM` it pauses after the current
epoch; resume with `--resume`.

## Post-training pipeline

Two user services wait for their training run and then execute the frozen
evaluation sequence automatically:

- `dsrt-phase-water5-pipeline-20261120`
- `dsrt-phase-quartz30-pipeline-20261120`

Stages: validation and test multistep (1/2/4/8 horizons), system NVE protocol
(Water: 20 ps from `val_04`; Quartz: 100 ps from `test_09` and `test_10`),
offline physical analysis, and a machine-readable/markdown comparison against
the frozen baselines. The pipeline refuses to overwrite partial trajectories and
records analysis failures without hiding completed tasks. Water `energy`
remains gated by the known reference verification failure; `full_energy`,
motion, RDF, VDOS and diffusion still run.

Outputs per run:

```text
evaluation/{val,test}_multistep.json
physics/<protocol>/{trajectory.h5,trajectory.summary.json,analysis/}
pipeline_stages.txt, pipeline_stage.txt, pipeline.log
summary.json, SUMMARY.md
```

Exit codes: 0 complete, 1 fatal (training/checkpoint/lock), 2 partial evidence
retained. `--summary-only` regenerates the report without rerunning stages;
`--include-nvt` adds the long NVT protocol. The pipeline is launched already and
is currently waiting for training to finish.

```bash
systemctl --user status dsrt-phase-water5-pipeline-20261120
journalctl --user -u dsrt-phase-quartz30-pipeline-20261120 -f
tail -f experiments/phase_space_20261120/water5/pipeline.log
```

## Not yet done

- No completed 250k budget, so no best checkpoint has been selected.
- No test metrics, multistep error, autonomous rollout, energy, RDF, VDOS or
  diffusion analysis.
- GPUs are shared with other users; absolute speed is not a dedicated-node
  benchmark.
