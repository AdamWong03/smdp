# DSRT: future-conditioned relational refinement for learned MD

DSRT learns a fixed-horizon phase-space transition without evaluating forces
during inference. It first proposes the future atomic state from current
periodic relations, converts that proposal into predicted future pair geometry,
and uses the future-conditioned relations to refine the endpoint.

The paper-stage method is frozen at R3-Opt. The active question is not whether
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

These results do **not** yet establish that future geometry is necessary. That
claim requires matched Current-Only and Ballistic-Geometry controls.

## Reproduction paths

Server assets are intentionally not committed:

- reference data: `/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1`
- frozen checkpoint: `/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261103_r3_opt/best.pt`
- derived graph cache: `data/smdp_reference_md_v1/quartz/derived_graphs/trajcast_p64_30fs`

```bash
/data1/wyd/envs/trajcast/bin/python train_relational.py --run /path/to/run

/data1/wyd/envs/trajcast/bin/python evaluate_relational_rollout.py \
  --checkpoint /path/to/best.pt --output /path/to/evaluation
```

The upstream TrajCast code is used for mature graph/data infrastructure and is
licensed separately under Apache-2.0. DSRT does not load TrajCast weights.

## Scope

The repository no longer treats Structured Action, Motion Modes, SACMP or the
continuous-segment prototype as active methods. Their committed history remains
recoverable from Git; large server artifacts were removed after their negative
results had been recorded.
