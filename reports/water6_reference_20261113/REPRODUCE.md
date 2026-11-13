# Commands and assets

Repository branch: `experiment/water6-native-reference-20261113`.
The exact code commit at launch and full package lists are exported under each
arm directory here. Model weights and trajectories stay on the server.

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export PYTHONPATH=.:/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Existing separate environments:

- DSRT: `/data1/wyd/envs/mlip/bin/python` (FP32, torch 2.13.0+cu130).
- TrajCast: `/data1/wyd/envs/trajcast-c52f107/bin/python`
  (FP64, torch 2.5.1+cu121, e3nn 0.5.9). Only the graph-builder dependency
  `vesin==0.6.2` was added; its torch/e3nn were not replaced.

Rebuild in a **new, empty** directory (the existing formal cache is complete;
do not run this over it):

```bash
/data1/wyd/envs/trajcast-c52f107/bin/python -m water_reference.prepare \
  --data /data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1 \
  --output /path/to/new/empty/cache
```

Formal launch (already running; do not duplicate):

```bash
CUDA_VISIBLE_DEVICES=GPU-83ddba27-1ec6-70a3-41f5-4b8a1d4e7900 \
  bash scripts/run_water6_reference.sh dsrt
CUDA_VISIBLE_DEVICES=GPU-e021d5a1-42fd-70cc-8dd1-08de8f1d8987 \
  bash scripts/run_water6_reference.sh trajcast
```

The launcher locks each run, rejects existing checkpoints, records the device,
checks real-data equations/gradients without optimizer updates, then starts the
full formal training. Correctness checks are not fitting runs or substitutes
for training. Native architecture was additionally matched against the public
Water checkpoint's entire state key/shape inventory; training never loads it.

Logs:

```bash
tail -f /data1/wyd/research/foundation_mlip_adaptation/experiments/water6_reference_20261113/dsrt_radius6/train.log
tail -f /data1/wyd/research/foundation_mlip_adaptation/experiments/water6_reference_20261113/trajcast_native/train.log
```

`history.jsonl` records full epoch metrics; native `progress.json` records
updates within an epoch. Both arms save `best.pt` and `last.pt` after an epoch.
Native SIGTERM requests a stop **after the current epoch**. Resume only after
the original process exits, using the same environment/device:

```bash
/data1/wyd/envs/mlip/bin/python -u train_relational.py \
  --run /data1/wyd/research/foundation_mlip_adaptation/experiments/water6_reference_20261113/dsrt_radius6 --resume
/data1/wyd/envs/trajcast-c52f107/bin/python -u -m water_reference.train_native \
  --run /data1/wyd/research/foundation_mlip_adaptation/experiments/water6_reference_20261113/trajcast_native --resume
```

Export actual progress and the full available history for GitHub:

```bash
/data1/wyd/envs/mlip/bin/python scripts/export_water6_reference.py
```

The initial rejected `TrajCastModel` run is retained separately under
`experiments/water6_reference_20261113/rejected_wrong_native_class`; it has no
scientific comparison result. Its discarded updates and elapsed time are
included in the exported snapshot. No historical model was interrupted.
