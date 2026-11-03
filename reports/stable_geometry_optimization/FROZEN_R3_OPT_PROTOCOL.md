# Frozen R3-Opt research stage

The architecture is frozen after the validation-selected R3-Opt experiment.
New physical conditions and systems train new weights when needed, but do not
change the method definition below.

## Method

- `geometry_gated` relational blocks
- 192 scalar, 48 vector and 96 edge channels
- two proposal blocks and two future-conditioned refinement blocks
- current relations -> proposed future -> future-conditioned refinement
- 30 fs endpoint position and velocity supervision for Quartz
- proposal weight 0.02
- AdamW/AMSGrad, FP32, batch 8
- 5,000-update warmup to 3e-4 followed by cosine decay to 1e-5

The frozen 5K-Quartz checkpoint is validation-selected epoch 364 at
`/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition/quartz30_20261103_r3_opt/best.pt`.
Its SHA256 is
`d59c8110762e328c8c59debf60820090b6ab912fc896e675db619af02e395508`.
The checkpoint is not changed or reselected using subsequent physical tests.

## Fixed research order

1. Freeze and organize the existing Quartz evidence, manuscript protocol and
   retained R1 lightweight comparison.
2. Train the same architecture from scratch on the existing flexible-water
   reference data at 5 fs, then evaluate RDF, MSD, diffusion, VACF/VDOS,
   temperature and energy distributions.
3. Train matched Current-Only and Ballistic-Geometry controls on Quartz and
   Water. These preserve the two-stage capacity and change only the relation
   supplied to the refinement blocks.
4. Complete Quartz independent repeats, 250/350 K transfer and independently
   thermalized 1,296/4,374-atom size transfer.
5. Build a diverse 25K Quartz training set from independent trajectory states;
   retrain the frozen architecture at a matched optimizer-update budget.

No new attention module, force/energy loss, propagator branch, dense-time loss
or architecture change is introduced during these stages. BKS is used only to
generate reference data and to evaluate energies, never during neural rollout.
External published results are contextual references, not same-protocol local
baselines. Test and physical trajectories never select checkpoints. If Full
DSRT does not reproducibly beat Current-Only, future geometry is not retained
as a supported mechanism claim.
