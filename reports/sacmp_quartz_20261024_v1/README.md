# SACMP Quartz launch — 2026-10-24

This is the candidate uploaded on `origin/main` through `877517b`, not a
continuation of the R3-B network. Branch: `experiment/sacmp-quartz-v1`.
No trained accuracy or long-time physical result is claimed here.

- Existing independent Quartz 16 fs dataset, 5,000 train / 1,250 validation /
  1,000 held-out test samples; no regenerated MD.
- Train-only 192-dimensional acceleration-PCA anchor: capture 0.8166641078;
  mass-weighted velocity capture 0.4071433604; registration RMS 0.1332020108 A.
- Fixed reference graph, learnable modal frame and frequencies, shared scalar
  fast/slow/mixed potential; the upstream method and objective are unchanged.
- Batch 4, seed 20261024; up to 80 segment epochs then 20 rollout epochs,
  upstream learning-rate scheduling and validation-only checkpoint selection.
- Validation/code checks: all 9 tests passed, including time reversal,
  canonical Jacobian, gradient reachability and cached-query time limits.
- GPU UUID: `GPU-f48b2c84-f983-64e3-000b-d669b9e64454` (RTX PRO 6000
  Blackwell Server Edition), shared server. Timing is not a matched L40S
  comparison to R3-B.

Server output:
`/data1/wyd/research/foundation_mlip_adaptation/experiments/sacmp/quartz_20261024_v1`

The output already contains the independently prepared train-only anchor
and source-validated segment cache. Launch from the project root:

```bash
CUDA_VISIBLE_DEVICES=GPU-f48b2c84-f983-64e3-000b-d669b9e64454 \
  bash scripts/run_sacmp_quartz_v1.sh
```

The launcher refuses existing training logs. It performs formal training,
held-out evaluation, and two 10 ps autonomous NVE screens with offline BKS
analysis. It does **not** automatically start 100 ps / 15 ns production or
ablations; the uploaded README requires examining the first results.
MPICH libraries are reused from the validated runtime.

Execution fixes: preserve external anchor metrics, enforce the documented
0.8 capture condition, validate all query horizons before prefix caching,
record configuration/code/hardware/progress, prevent trajectory overwrite,
and label endpoint-only segment differences in velocity units. No new loss,
mode selector, thermostat change, or physical correction was introduced.

Large checkpoints, trajectories, caches and raw logs stay on the server.
