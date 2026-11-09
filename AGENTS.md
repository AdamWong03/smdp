# Conditional-transition experiment scope

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
