# DSRT: future-conditioned relational refinement for learned MD

Latest experiment: [CMIS formal Quartz / Water protocol](reports/cmis_20261106/PROTOCOL.md)
on `experiment/cmis-coupled-impulse-segments`. Full 350-epoch training runs are
in progress. [CMIS](cmis/README.md) is the newly supplied model; the DSRT
implementation and completed results below remain historical comparisons.

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

These results do **not** yet establish that future geometry is necessary. That
claim requires matched Current-Only and Ballistic-Geometry controls.

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
