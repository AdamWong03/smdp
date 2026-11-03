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

1. Complete Quartz physical validation: 15 ns NVT, longer/multiple NVE,
   250/350 K generalization, independently thermalized 1,296/4,374-atom cells,
   and end-to-end efficiency.
2. Build a 25K Quartz training set emphasizing independent trajectories and
   non-overlapping state coverage; retrain exactly the same architecture and
   compare at a matched update budget.
3. Train the same architecture from scratch on the existing flexible-water
   reference data at a water-appropriate horizon, then evaluate RDF, MSD,
   diffusion, VACF/VDOS, temperature and energy distributions.

No new attention module, force/energy loss, propagator branch, dense-time loss
or architecture change is introduced during these stages. BKS is used only to
generate reference data and to evaluate energies, never during neural rollout.
External published results are contextual references, not same-protocol local
baselines. Any later no-future-feedback ablation is kept separate from this
physical-validation sequence.
