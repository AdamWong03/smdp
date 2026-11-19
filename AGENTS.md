# Latest instruction: next operator design (2026-11-20)

The user authorized design of the next code version. Work is on
`research/phase-space-operator-20261120`, with a separately implemented
`PhaseSpaceTransition` prototype and an explicit operator specification in
`paper/PHASE_SPACE_OPERATOR_DESIGN_20261120.md`. This supersedes the older
architecture freeze for this design work. Do not rank historical outcomes to
choose components. Preserve the old model, trainer, results and checkpoints.

This phase implemented the calculation graph and then wired it into the
production trainer/loader behind the explicit `phase_space_operator_v0`
architecture switch. TRAIN-only normalization is frozen into checkpoint configs.
It still does **not** launch formal training or reference reproduction.
Configs under `configs/design/` remain specifications; the flattened launch
configs carry `"formal_training_started": false`. Measured same-device cost and
remaining limits are in `reports/phase_space_integration_20261120/INTEGRATION.md`.
Any later integration must preserve target units, graph orientation, train-only
normalization and validation-only model selection.

# Implementation-first design (2026-11-19)

The user explicitly said to stop making design judgments from historical
experimental outcomes and instead inspect actual implementations together with
frontier work. Preserve historical artifacts, but do not use their ranking to
select or reject the next architecture. Audit effective config dispatch, tensor
operations, training targets and execution cost. Reference code is not assumed
correct or equivalent to a paper configuration merely because it is official.
See `reports/implementation_audit_20261119/`. No new model/training is launched
by this audit. This instruction supersedes outcome-driven rationale below.

# Active priority: develop our method (2026-11-19)

The user explicitly redirected work away from continued TrajCast reproduction.
The native reference trainer has been stopped with best/last checkpoints kept.
Do not restart it or schedule further reference runs automatically. The Full /
Current-Only formal comparison is complete except for the disclosed energy
verification. Its result does not establish an overall future-feedback benefit.

The next primary deliverable is an implementable optimization of our neural
state propagator, grounded in the shared molecular-motion error and prior
negative results. Do not substitute another broad diagnostic campaign or a
baseline completion milestone for that deliverable. No new architecture or
training is represented as already implemented by this change of priority.
See `paper/RESEARCH_FOCUS_20261119.md`. Preserve all existing artifacts.

## Completed scope: future-relation mechanism control (2026-11-17)

The user accepted the research decision and authorized autonomous execution.
Follow `reports/future_relation_control_20261117/PROTOCOL.md`. Reuse the completed
Water 6 Å Full DSRT run; train one Current-Only control from random initialization
with exactly its data, parameters, optimizer and 250k-update budget. Change only
the relation condition in refinement. Keep proposal/correction heads, all four
blocks, proposal supervision and the final proposal residual. This supersedes
the Stage A restriction on adding this controlled experiment.

Use this isolated worktree; do not edit the original worktree serving the running
TrajCast training. Preserve historical checkpoints and results. Do not add
architecture, loss terms, data, physics correction or additional experiments
to respond to intermediate metrics. Validation alone selects checkpoints.

## Historical Water reference comparison (2026-11-13)

The research lead accepted `paper/NMI_RESEARCH_PLAN_20261113.md` and authorized
Stage A. Train unchanged R3-Opt with Water radius 6 Å and native pinned TrajCast
on the same existing Water split, from random initialization. Rebuild graphs
at the new cutoff; do not reuse the 4.5 Å cache under a different setting.
Use the native baseline's documented optimization and report budget differences.
Preserve all historical artifacts. Validation alone selects checkpoints.
Do not introduce a new network, motion modes, physical losses or on-policy data.
This authorization supersedes the older experiment scope below.

## Historical conditional-transition scope

The research lead authorized the protocol in
`docs/CONDITIONAL_TRANSITION_LEARNING_PROPOSAL.md` on 2026-11-09 and requested
execution. This supersedes the older SMDP-R-only scope below. Freeze the R3-Opt
geometry_gated network and endpoint objective. Execute the four matched 25k
update arms for Quartz/Water. Only TRAIN sources may generate additional labels;
diagnostic validation banks must never enter training. Do not add architecture,
physics losses, inference force calls, or unplanned training branches. Preserve
all historical results. Use fixed-final checkpoints for the new comparison.

## Historical SMDP-R scope (retained)

The research lead explicitly authorized SMDP-R on 2026-11-07. This is the
applicable exception to the older parent DSRT-only architecture freeze.

Follow `docs/SMDPR_PROTOCOL.md`. Preserve the R3-Opt backbone and all historical
experiments/checkpoints. Primary SMDP-R models train from scratch. Do not add
new physics, modes, losses or experts during these official runs. Checkpoints
are selected on validation only. Record failures without replacing trajectories
or modifying thermostat settings. Query supervision must remain detached from
the endpoint backbone. Never label pending physical evaluation as completed.

Active run paths, service names and logs are in
`reports/smdpr/IMPLEMENTATION_STATUS.md`. Do not change their configs or training
code while they run, except to fix and explicitly record a demonstrated bug.
