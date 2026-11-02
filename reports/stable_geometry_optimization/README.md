# Stable Geometry-DSRT controlled optimization experiment

This experiment freezes the Quartz 30 fs data, 192/48/96 width, 2+2 depth,
endpoint loss, proposal weight and random seed. It trains exactly two models:

1. **R3-Opt:** existing `geometry_gated` block with the new fixed schedule.
2. **Stable-DSRT:** `geometry_stable`, replacing Post-Norm residuals with
   Pre-Norm branches and learned per-channel edge/scalar/vector residual scales
   initialized to 0.1.

Both use FP32, batch 8, AdamW/AMSGrad, zero weight decay, 5,000-step linear
warmup from 1e-5 to 3e-4, cosine decay to 1e-5 and exactly 250,000 optimizer
steps (400 full data epochs). Validation selects `best.pt`; ordinary validation
stagnation does not stop training. Test data is first read after the complete
fixed schedule and model selection.

## Historical evidence before training

| Run | Best epoch | Best updates | Train final loss | Val final loss | Val q RMSE (A) | Val v RMSE (A/fs) |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 348 | 217,500 | 0.0007675 | 0.0007365 | 0.0016344 | 0.00011107 |
| R2 | 27 | 8,451 | 0.0023731 | 0.0021800 | 0.0027791 | 0.00019233 |
| R3 | 35 | 21,875 | 0.0010891 | 0.0010363 | 0.0019598 | 0.00013095 |

After their selected points, both R2 and R3 show increasing training and
validation final losses, as well as increasing proposal losses. The recorded
history therefore does not support describing those runs as ordinary validation
overfitting while training loss kept improving. R1/R2/R3 did not record separate
normalized training position and velocity terms; this missing information is
explicit in `history_comparison.json`.

## New logging

Every epoch records final/proposal position and velocity loss terms, LR and
cumulative optimizer steps. Every 10 epochs (and epoch 1) a fixed training batch
records per-block feature RMS, residual/main-path ratios, residual scale means,
and gradient norms under the final loss and the weighted proposal loss. The
diagnostic backward passes never update parameters.

The model architecture, scheduler endpoints and diagnostic path were checked on
real cached Quartz graphs before formal training. `geometry_gated` has 2,357,616
parameters; `geometry_stable` has 2,358,960.
