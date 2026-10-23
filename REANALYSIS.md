# R2 frozen-trajectory reanalysis

Scope: analyze existing R1/R2 trajectories and frozen validation checkpoints.
No training, new MD integration, changed physics, thermostat tuning or R3 run.
The published report is
[`reports/quartz16_additive_long_rollout/PHYSICS_REPORT.md`](reports/quartz16_additive_long_rollout/PHYSICS_REPORT.md).

## Runtime and source locations

```bash
cd /data1/wyd/research/foundation_mlip_adaptation/smdp
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib:/data1/wyd/envs/mlip/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PYTHON=/data1/wyd/envs/mlip/bin/python
R1=/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r1
R2=/data1/wyd/research/foundation_mlip_adaptation/experiments/motion_modes/quartz16_20261022_r2_additive
DATA=/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1
ANALYSIS=$R2/reanalysis_v1
```

The existing LAMMPS shared library requires `libmpi.so.12`. The compatible
installed library is MPICH in `lammps-runtime/lib`; do not create an OpenMPI
symlink with that name. `readelf -d` and `ldd` on
`/data1/wyd/envs/mlip/lib/python3.12/site-packages/lammps/liblammps.so`
identify/resolve this dependency. R2's original traceback is retained in
`$R2/analyze_nvt.log`. This fix changes the process library search path only.

First verify the original R1/BKS energy convention, before R2 energy analysis:

```bash
$PYTHON verify_bks_runtime.py --r1 "$R1" --data-root "$DATA" --output "$ANALYSIS"
```

## Independent physical tasks

```bash
for NAME in nve_test_09 nve_test_10; do
  $PYTHON analyze_quartz.py --candidate "$R2/$NAME.h5" \
    --reference "$DATA/quartz/raw_nve/${NAME#nve_}/trajectory.h5" \
    --output "$ANALYSIS/${NAME}_analysis"
done
$PYTHON analyze_quartz.py --candidate "$R2/nvt.h5" \
  --reference "$DATA/quartz/physical_reference/nvt_15ns/production.h5" \
  --output "$ANALYSIS/nvt_analysis" --spectral-workers 6
```

`--tasks thermo rdf energy thermostat vdos_macro vdos_dense` selects tasks.
Energy initialization is isolated: a missing LAMMPS/MPI library records an
energy failure but does not prevent the other tasks. Each task saves a JSON
status/result, failure traceback if applicable, and time. Successful tasks
are skipped on restart; failed tasks are retried. Input paths, size and mtime
are recorded and checked; changed input files require a new output directory.
Long spectra additionally checkpoint each complete physical-time block under
`spectral_cache/`. Interrupted spectral tasks resume these exact block spectra
without changing windowing, normalization or the selected time range. This
cache remains on the server and is not published.
Optional spectral workers use separate read-only HDF5 handles and independent
time blocks; results are merged in original time order. Software tests verify
bitwise agreement with the serial definition. This changes offline runtime,
not the model, physical trajectories or evaluation protocol.
Use a new versioned output directory when deliberately changing the analysis
definition; a successful old task is not silently overwritten.

Trajectory completion and physical-analysis completion are separate fields.
The top-level JSON consolidates task status after every finished task and at
exit. Missing physical analysis must not be interpreted as missing production.
All HDF5 files are opened read-only. The evaluation definitions match R1.

## Frozen checkpoint validation and thermostat replay

```bash
CUDA_VISIBLE_DEVICES=2 $PYTHON analyze_training_stages.py \
  --r1 "$R1" --r2 "$R2" --data-root "$DATA" --output "$ANALYSIS/stages"
$PYTHON verify_csvr_replay.py --trajectory "$R2/nvt.h5" \
  --output "$ANALYSIS/csvr_replay.json"
$PYTHON report_r2_reanalysis.py --r1 "$R1" --r2 "$R2" --output "$ANALYSIS" \
  --publish reports/quartz16_additive_long_rollout
$PYTHON -m pytest -q
```

Stage comparison uses all 1250 validation starts (1245 valid 256 fs windows),
not test data, with frozen `best_segment.pt`, `best_rollout4.pt`,
`best_rollout8.pt`, `best_rollout16.pt` and the R1 final checkpoint.
This is bounded validation inference, not generation of another MD trajectory.
The decoder-deletion diagnostic subtracts one mode's final contribution while
holding upstream computation fixed; it is not a full causal mode ablation.

Large raw files, checkpoints, per-sample validation CSVs and original logs
remain on the server. The repository contains summary JSON, reproducible
analysis code, selected plots and the report. The original incomplete published
summary is retained under `historical_incomplete/`, clearly labeled obsolete.

## Interpretation of uncertainty

Energy/thermodynamic comparison uses the same 2000 uniform frames as R1;
energy-window endpoints are interpolated between those samples. RDF uses the
same 512 frames. VDOS compares all complete 16.384 ps windows of production.
Reported `block_mean_std` is the standard deviation of eight block means,
not a standard error based on independent trajectories. Whole-average-spectrum
overlap and average per-block overlap differ and are both retained. CSVR
diagnostics additionally use every production step and are labeled separately
from the 2000-frame summary. No confidence interval is inferred from millions
of correlated frames as if they were independent replicates.
