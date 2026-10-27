# Original TrajCast foundation for the next SMDP route

This directory integrates, but does not modify, the official TrajCast model,
Trainer, ConservationLayer, periodic graphs and Forecast. Read [AUDIT.md](AUDIT.md)
for licenses, exact input/output semantics, verified differences and remaining
physical-protocol uncertainties. See [results](../reports/trajcast_foundation_20261027/RESULTS.md)
for observed results, not anticipated performance.

## Environment and immutable assets

Run from `/data1/wyd/research/foundation_mlip_adaptation/smdp`:

```bash
git clone https://github.com/IBM/trajcast.git ../upstream/trajcast-c52f107
git -C ../upstream/trajcast-c52f107 checkout --detach c52f107a02176671b21a420b2d4c67e0327b04f7
bash scripts/setup_trajcast_foundation.sh
/usr/bin/python3 -m trajcast_based.acquire
```

The checkout step is for a new clone, never a dirty historical checkout.
`acquire.py` locks Hugging Face model/data revisions separately, downloads cards,
verifies all Quartz assets, and records hashes. The new environment has
`include-system-site-packages = false`; nothing is installed in the old environments.
The fully resolved dependency lock is exported with the results.

## Actual reproduction and data adaptation

```bash
PY=/data1/wyd/envs/trajcast-c52f107/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
$PY -m trajcast_based.check_interfaces
$PY -m trajcast_based.adapt_data
$PY -m trajcast_based.verify_cache
$PY -m trajcast_based.check_scheduler
$PY -m trajcast_based.configure
CUDA_VISIBLE_DEVICES=1 $PY -m trajcast_based.official evaluate
CUDA_VISIBLE_DEVICES=1 bash scripts/run_trajcast_foundation.sh
```

The first run uses 334 warmup + 3334 production steps, all 30 fs: 10.02 ps +
100.02 ps. It compares real autonomous spectra/temperature/offline BKS energy,
not only checkpoint loading. For the full paper duration, use:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_trajcast_foundation.sh --full-paper-duration
```

That adds an independent original Forecast run with 50,000 warmup + 500,000
production steps (1.5 + 15 ns), unchanged 75 fs CSVR damping. Existing trajectories
are never overwritten. If a run fails halfway, preserve it and diagnose; do not
rerun into the same file. Git push failure should be retried independently of
completed physics. Long trajectories and all weights stay on the server.

For unattended execution, use a persistent user service rather than an
interactive terminal job. On 2026-10-27 the first long job disappeared during
thermalization (last logged step 2336, cause undetermined). Its partial output is
retained under `official30_15ns`, with `interruption.json`. The fresh replacement
uses the same original seed/initial condition, not a concatenation or restart
from a state lacking the thermostat RNG state:

```bash
systemd-run --user --unit=smdp-trajcast-official30-20261027 \
  --working-directory=/data1/wyd/research/foundation_mlip_adaptation/smdp \
  --setenv=CUDA_VISIBLE_DEVICES=GPU-18ea7690-4b23-bbfd-b455-f6efe865ab85 \
  --setenv=SMDP_SERVICE_UNIT=smdp-trajcast-official30-20261027.service \
  /bin/bash scripts/run_trajcast_official_long.sh official30_15ns_service1
systemctl --user status smdp-trajcast-official30-20261027.service
```

This command documents the actual launch. A repeat needs a new unit and output
tag; existing output is deliberately rejected. GPU UUID is host-specific and
must be selected from available resources on another host. The service runs
offline analysis and publishes small results after successful forecasting;
it never starts training. Logs live at `quartz_20261027/<tag>.log`. If forecasting
finishes but analysis/publication fails, rerun those modules separately, not the
trajectory generator.

`analyze_official.py` invokes the preserved `src/evaluate_quartz_trajectory.py` in
the validated LAMMPS/MPICH environment. It additionally records uncentered energy
differences so a centered distribution cannot conceal a potential discrepancy.
The paper VDOS source grid is used exactly (30 fs Nyquist ≈16.67 THz); it cannot
establish recovery of higher-frequency modes.

The dataset adapter verifies every endpoint against its raw continuous source,
and exports an original PyG cache with exact masses. All train/val/test source
sets must be disjoint. A generic ASE extxyz roundtrip is not used to silently
replace 28.0855 u with 28.085 u or to round labels to eight decimal digits.

## Original 16 fs baseline (not adapter training)

`configure.py` creates `original16.yaml`. After actual official-model physics
review, record `official30_validation_accepted: true` in a JSON decision with the
evidence and remaining limitations, then launch:

```bash
$PY -m trajcast_based.train_original --approval /path/to/recorded_validation_decision.json
```

The Trainer is upstream. An early-stopping logging hook implements the paper's
100-epoch patience. A local scheduler counter adapter reconciles its once-per-
epoch call with the released wrapper's minibatch counter (see AUDIT.md and
`training_scheduler_check.json`). Neither changes training targets or loss.
Do not manufacture an acceptance decision merely because interfaces pass.
Original16 uses our fixed independent Quartz split, not the official 30 fs pairs.

After a reliable original16 baseline exists, separate Original / +Modes /
+Modes+Segment versions can be compared. This phase contains no adapter, new
expert, segment loss, or new trajectory decoder.

## Modification record

2026-10-27: added independent integration scripts, pinned acquisition,
exact cache adapter, interface checks, original-model evaluation/Forecast
launchers, baseline configuration and explicit result publication. No changes
to upstream source or historical SMDP models. Setup selects e3nn 0.5.9 within
upstream's allowed range because the index does not provide 0.5.4.

Upstream Apache-2.0 license is retained in [LICENSE.upstream](LICENSE.upstream).
Original author and paper attribution are preserved in [AUDIT.md](AUDIT.md).
