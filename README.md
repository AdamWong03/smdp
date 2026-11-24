# DSRT: future-conditioned relational refinement for learned MD

> **Status 2026-11-24 — research reset; no submission claim.** The 1 fs backbone trained
> (test position RMSE \(3.16\times10^{-5}\) Å, velocity RMSE \(4.58\times10^{-5}\) Å/fs).
> The frozen-backbone + post-hoc GLR/Langevin integrator family failed the water hard gate at
> both 5 fs and 1 fs: no configuration simultaneously matched the mode partition, RDF/VDOS,
> and transport. The paper is **not submittable** as a method paper. The remaining decision is
> end-to-end training with the integrator in the loop vs stopping/pivoting.
>
> Project state and paper snapshot: `research/RESEARCH_CONTRACT.md`, `research/RESEARCH.md`,
> `research/paper/CLAIM_BOUNDARY.md`, `research/paper/latex/main_clean.pdf`.
> This branch contains the GLR/1 fs experiment code and the research records.

## Current work: executable operator design (2026-11-20)

A new design prototype keeps wide `0e + 1o + 2e` node states, transports narrow
edge messages, and couples the aggregated environment with node motion through
node-local equivariant products. It preserves the two-stage endpoint contract.

- [Full operator specification and limitations](paper/PHASE_SPACE_OPERATOR_DESIGN_20261120.md)
- [Independent reference implementation](phase_space_transition.py)
- [Correctness checks and reproduction](reports/phase_space_design_20261120/IMPLEMENTATION.md)

The 3,036,224-parameter prototype has passed algebra/gradient tests and full-width
FP32 TRAIN-batch checks for Quartz and Water. It is wired into the production
trainer, checkpoint loader and diagnostics path behind the explicit
`phase_space_operator_v0` architecture switch, with TRAIN-only normalization
frozen into checkpoint configs. Two formal 250k-update runs (Water 5 fs and
Quartz 30 fs, random initialization) **completed on 2026-11-22**. Test metrics:
water position MAE \(4.9\times10^{-4}\) Å, quartz \(1.9\times10^{-3}\) Å. The raw
water v0 NVT still fails long-time dynamics (VDOS 0.356/0.236, \(D=2.8\times10^{-13}\)
m²/s); the paper's method is the frozen-backbone + GLR readout, not the raw network.
[Training status](reports/phase_space_training_20261120/STATUS.md).

Same-device FP32 cost (one L40S, batch 8, shared load): the training step is
+15% (Water) and +65% (Quartz) versus frozen DSRT; peak training memory falls
21-37%; model-only one-graph inference is 1.9x/2.6x slower. Integration, raw
records and limits: [integration report](reports/phase_space_integration_20261120/INTEGRATION.md).

## Completed basis: source-level network and training audit (2026-11-20)

Architecture decisions now follow the implemented operators, effective configs,
training definitions and computational cost, rather than historical experiment
rankings. The audit covers our propagator and pinned TrajCast, MDtrajNet, PaiNN,
GotenNet, GNS, FlashMD, E2Former-V2 and EquiformerV3 source paths.

- [Implementation and training audit](reports/implementation_audit_20261119/AUDIT.md)
- [Frontier operators and execution cost](reports/implementation_audit_20261119/FRONTIER_OPERATORS.md)
- [Source provenance and reproduction](reports/implementation_audit_20261119/REPRODUCE.md)

CPU checks verify tensor/target contracts only. No new architecture, training,
MD simulation or external-method reproduction is claimed by this audit.

## Prior status: completed control and reference-training stop (2026-11-19)

The Full / Current-Only training, endpoint tests and paired 20 ps NVE runs are
complete. Their comparison does not demonstrate an overall benefit of predicted
future geometry. Both retain a large internal-motion kinetic-energy error.
[Results and limitations](reports/future_relation_control_20261117/RESULTS.md).

Native TrajCast reference training was stopped on November 19, preserving its
epoch-149 resumable state and epoch-145 validation-best checkpoint. It is not a
completed native-protocol baseline. Method development will not wait for more
reference training. [Research focus and stopping record](paper/RESEARCH_FOCUS_20261119.md).

## Completed experiment: predicted future relations vs current-only (2026-11-17)

This branch tests DSRT's central mechanism. Reuse the completed Full Water 6 Å
model and train a Current-Only control from scratch for the same 250,000 updates.
The only network change is the geometric condition of refinement; both stages,
heads, parameter count, proposal supervision and endpoint residual remain intact.

- [Frozen protocol and reproduction](reports/future_relation_control_20261117/PROTOCOL.md)
- [Measured status](reports/future_relation_control_20261117/STATUS.md)
- [Machine-readable snapshot](reports/future_relation_control_20261117/snapshot.json)
- [Control configuration](configs/water6_current_only_20261117.json)

Current-Only trained on shared RTX PRO 6000 Blackwell and Full on shared L40S:
optimization budgets match, hardware/wall time do not. Both completed 250,000
updates. The pipeline also completed validation multistep prediction, paired
20 ps trajectories and endpoint timing. PE/total-energy analysis remains
unverified; the service's failure status refers to that postprocessing gate.

Server run: `/data1/wyd/research/foundation_mlip_adaptation/experiments/future_relation_control_20261117/current_only`.
Logs: `train.log`, `history.jsonl`, `pipeline.log`, `pipeline_stages.txt`.
Service: `systemctl --user status dsrt-current-only-water6-20261117`.

## Historical SMDP-R and successor-supervision experiments

The motion-mode and two-step controls are preserved, but are not the active
method development. Their completed diagnosis did not establish robust benefits
of multiple modes. Conditional-successor supervision also failed its fixed
physical improvement criterion. Read the actual results rather than treating
the older progress snapshots as current status:

- [Frozen-model dynamics diagnosis](reports/dynamics_diagnosis_20261109/DIAGNOSIS.md)
- [Conditional-successor results](reports/conditional_transition_20261109/RESULTS.md)
- [Historical SMDP-R protocol](docs/SMDPR_PROTOCOL.md)

## Historical DSRT method and results

DSRT learns a fixed-horizon phase-space transition without evaluating forces
during inference. It first proposes the future atomic state from current
periodic relations, converts that proposal into predicted future pair geometry,
and uses the future-conditioned relations to refine the endpoint.

Historical evaluations keep R3-Opt frozen. The research question is not whether
a neural network can directly propagate MD—prior work already establishes
that—but whether predicted future geometry is a useful conditioning signal for
finite-time relational propagation and whether any gain survives long-time
physical evaluation.

## Active implementation

- [`relational_transition.py`](relational_transition.py): DSRT model and losses.
- [`train_relational.py`](train_relational.py): fixed-horizon endpoint training.
- [`relational_inference.py`](relational_inference.py): frozen-checkpoint loader.
- [`rollout_relational.py`](rollout_relational.py): autonomous NVE/NVT rollout.
- [`evaluate_relational_rollout.py`](evaluate_relational_rollout.py): short
  autonomous prediction evaluation.
- [`analysis_tasks.py`](analysis_tasks.py), [`physics/`](physics): offline BKS,
  RDF, VDOS and thermodynamic analysis.
- [`analyze_water.py`](analyze_water.py): original SPC/Fw energy decomposition,
  molecular kinetic partition, RDF, VDOS/VACF and NVT diffusion.
- [`diagnose_water_bias.py`](diagnose_water_bias.py): signed prediction errors
  on real and autonomous states, with evaluation-only true continuations.
- [`periodic.py`](periodic.py): scalable periodic graph construction.

The frozen configuration is in
[`configs/quartz30_r3_opt.json`](configs/quartz30_r3_opt.json). The manuscript
draft and fixed evidence plan live in [`paper/`](paper/README.md).

## Frozen Quartz result

R3-Opt has 2,357,616 parameters and was trained on 5,000 independent-split
Quartz 30 fs transitions. Its independent-test endpoint RMSE is 0.001800 Å for
position and 0.0001221 Å/fs for velocity. Two 100 ps NVE trajectories remained
structurally stable but showed measurable BKS energy drift. A complete 1.5 ns
thermalization plus 15 ns NVT production trajectory achieved O/Si VDOS overlap
of 0.946/0.950 within the 30 fs-resolvable band and 106.1 ns/day end-to-end on
one L40S. Full numbers and limitations are in
[`R3_OPT_PHYSICS_RESULTS.md`](reports/stable_geometry_optimization/R3_OPT_PHYSICS_RESULTS.md).

These results do **not** establish that future geometry is necessary. The Water
Current-Only control now supplies negative/mixed evidence for that hypothesis;
further ablations are not scheduled automatically to preserve the old narrative.

## Water 5 fs result and diagnosis

The same architecture trained from scratch completes 250,000 updates and
reaches independent-test position/velocity RMSE 0.000654 Å / 0.0002376 Å/fs.
Autonomous Water dynamics are substantially worse than endpoint error suggests:
the first 50 ps gains 6.30 eV total energy, internally overheats, and loses
molecular translation/rotation energy. H/O VDOS overlap is 0.391/0.614.
The original-protocol 2.5 ns CSVR evaluation is also complete: internal motion
dominates, molecular translation/rotation are suppressed, and H/O VDOS overlap
is only 0.215/0.233. See [the measured diagnosis](reports/water5_20261103/PHYSICS.md)
and [complete NVT results](reports/water5_20261103/NVT_RESULTS.md).

## Reproduction paths

Server assets are intentionally not committed:

- reference data: `/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1`
- frozen checkpoint: `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261103_r3_opt/best.pt`
- derived graph cache: `data/smdp_reference_md_v1/quartz/derived_graphs/trajcast_p64_30fs`

```bash
export PYTHONPATH=/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
/data1/wyd/envs/mlip/bin/python train_relational.py --run /path/to/run

/data1/wyd/envs/mlip/bin/python evaluate_relational_rollout.py \
  --checkpoint /path/to/best.pt --transitions /path/to/test.h5 \
  --output /path/to/evaluation.json
```

The upstream TrajCast code is used for mature graph/data infrastructure and is
licensed separately under Apache-2.0. DSRT does not load TrajCast weights.

## Scope

The repository no longer treats Structured Action, Motion Modes, SACMP or the
continuous-segment prototype as active methods. Their committed history remains
recoverable from Git; large server artifacts were removed after their negative
results had been recorded.
