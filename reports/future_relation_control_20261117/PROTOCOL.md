# Does predicted future geometry help DSRT?

Authorized 2026-11-17. One formal Current-Only training run, reusing the completed
Full Water 6 Å model. This is the primary mechanism comparison, not a new model
family or a search over additional losses, representations or thermostats.

## Hypothesis and exact intervention

At the same depth, parameter count and supervised optimization budget, explicitly
conditioning refinement on predicted future relative position/velocity improves
finite-time prediction and autonomous molecular motion relative to current-only
geometry. Improvement is a hypothesis, not a requirement for reporting results.

Both arms keep geometry_gated 192/48/96, 2 proposal + 2 refinement blocks,
persistent node/edge features, both endpoint heads, the proposal output residual,
COM/momentum treatment and the same periodic graph. No parameters are removed.

Full: `relations(d0, u0, d1_proposal, u1_proposal)`.
Current-Only: `relations(d0, u0, d0, u0)`.

The candidate still contributes to the final q/v through the existing additive
readout. The main endpoint loss still trains the proposal head even with zero
auxiliary weight. Only its geometric feedback to refinement is removed. Both
versions recompute relation encoding in refinement. Forward/backward FLOPs are
not claimed identical: the control removes the corresponding gradient path.

`relation_conditioning` defaults to `predicted_future`, adding no parameters or
buffers. Historical checkpoints load strictly. The old default forward is
compared against source commit b3b2f76 on real TRAIN graphs, including the frozen
Full checkpoint. Seeded parameters and RNG consumption must match exactly.

## Formal training

- Flexible SPC/Fw Water, 64 molecules / 192 atoms, 5 fs macrostep, 6 Å radius.
- Existing 5,000 / 1,250 / 1,000 source-independent train/val/test split and the
  verified 6 Å graph cache. No reference data generation or relabeling.
- Same seed 20261103; FP32, TF32 disabled; batch 8; AdamW AMSGrad, weight decay 0.
- Normalized endpoint objective `.5*(q_MSE + v_MSE) + .02*proposal_loss`.
- Same training-only RMS, clip 10, 250,000 updates / 400 epochs.
- Linear warmup 1e-5 to 3e-4 over 5,000 updates, cosine to 1e-5 thereafter.
- No ordinary early stopping. Abort on nonfinite gradients; preserve failure.
- Select best using all validation endpoint samples. The unchanged trainer
  evaluates the test set once after training/model selection finishes. Test
  metrics do not select checkpoints, determine stopping or change this design.
- Current-Only starts from random initialization, never from Full weights.
  Loading Full weights in a separate correctness process is not training.

Full completed 250,000 updates, validation-best epoch 398 / update 248,750,
2,357,616 parameters. Reuse its metrics, weights and trajectory rather than
rerunning successful work. Current-Only has exactly the same parameter count.

Full trained on a shared L40S. The new run uses the available memory on GPU 0,
RTX PRO 6000 Blackwell, with the same PyTorch 2.13/CUDA13 environment and FP32
settings. Both GPUs are shared. This matches data/optimization/parameter budgets,
not wall time or bitwise GPU arithmetic. No training-speed advantage will be
inferred from different hardware. Final paired forward timing uses the same GPU
and input, reports shared load, and is distinct from production throughput.

## Evaluation sequence

1. Complete the entire training budget, retaining best/last, optimizer/scheduler,
   RNG state and full per-epoch history. No interim physics-based reselection.
2. Reuse Full endpoint and validation 1/2/4/8-step results; evaluate Current-Only
   on the identical 1,249 valid validation continuations (the full 1,250 endpoint
   samples remain unchanged).
3. Run Current-Only 20 ps NVE, val_04 frame 0, 4,000 macrosteps. Compare with the
   existing Full trajectory at exactly the same initial state. Preserve failures.
4. Use existing motion/RDF/VACF/VDOS analysis at the same 2,000-frame sampling,
   with translation/rotation/internal KE and the full temperature evolution.
   Report 0–1 / 1–5 / 5–20 ps windows; these nonstationary values do not establish
   equilibrium transport or diffusion. Do not claim independent replicas from
   correlated frames. This first comparison has one training seed and one start.
5. Keep the strict eight-frame reference energy verification. It currently fails
   on the retained val_04 coordinate serialization at one frame. Do not relax the
   tolerance, change the force field or use the legacy two-frame full-energy pass
   to bypass that failure. Attempt energy independently and report it incomplete
   if verification still fails; do not represent missing energy as zero drift.
6. Compare final endpoints on identical prebuilt graphs on the same GPU, without
   optimizer updates; record wall time, peak tensor memory and shared-device caveat.

No new long NVT is launched automatically before assessing this comparison.
Physical validation cannot substitute another checkpoint. A favorable single-seed
result is a reason for matched replication, not proof of a general mechanism.
Endpoint-only gains, wrong motion partition, or no useful gain over Current-Only
do not support the intended long-time-dynamics claim. No negative result triggers
an automatic new architecture, loss, cutoff, fine-tuning or extended budget.

## Assets and execution

Root: `/data1/wyd/research/foundation_mlip_adaptation`.

- Isolated code: `smdp-future-control`, branch
  `experiment/future-relation-control-20261117`, based on b3b2f76.
- Reused Full: `experiments/water6_reference_20261113/dsrt_radius6`.
- New control: `experiments/future_relation_control_20261117/current_only`.
- Config: `configs/water6_current_only_20261117.json`.
- Full logs, HDF5 and weights remain on the server. Export only reports, complete
  training curves, metrics and figures. Original worktree/TrajCast run stay intact.

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp-future-control
CUDA_VISIBLE_DEVICES=GPU-f48b2c84-f983-64e3-000b-d669b9e64454 \
  bash scripts/run_future_relation_control.sh
# After an interruption, explicitly pass --resume. Existing trajectories are
# never overwritten and completed postprocessing is reused.
```

This is a controlled development comparison. Broad claims will require matched
replication, additional prediction spans/conditions and independent final
physical evaluation, chosen after this central question has an answer.
