#!/usr/bin/env bash
# Wait for the official published-TrajCast 2.5 ns run to finish, then run the
# exact same water analysis protocol used for the GLR 2.5 ns trajectory.
set -euo pipefail

ROOT=/data2/wyd/research/foundation_mlip_adaptation
DATA=/data1/wyd/research/foundation_mlip_adaptation
RUN=$ROOT/experiments/published_trajcast_2p5ns_20261126
REF=$DATA/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5
TOP=$DATA/data/smdp_reference_md_v1/water/equilibration/snapshots/water_5.data

# 720 * 5 min = 60 h; the run is GPU-contended and may take longer than 24 h.
for _ in $(seq 1 720); do
  if [ -f "$RUN/trajectory.summary.json" ]; then
    break
  fi
  if ! pgrep -f "native_nvt_screen.py.*published_trajcast_2p5ns_20261126" >/dev/null; then
    echo "TrajCast process exited without trajectory.summary.json" >&2
    exit 1
  fi
  sleep 300
done
if [ ! -f "$RUN/trajectory.summary.json" ]; then
  echo "Timed out waiting for TrajCast 2.5 ns" >&2
  exit 1
fi

# Give the writer a moment to close the HDF5 file.
sleep 20

cd "$ROOT/code/h1h2"
export PYTHONPATH=$DATA/upstream/trajcast-c52f107:$PWD
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

/data1/wyd/envs/mlip/bin/python -u analyze_water.py \
  --candidate "$RUN/trajectory.h5" \
  --reference "$REF" \
  --topology "$TOP" \
  --output "$RUN/analysis" \
  --tasks motion rdf vdos diffusion \
  --frames 4000 \
  --model-label "Official IBM TrajCast 2.5 ns, 5 fs"

/data1/wyd/envs/mlip/bin/python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
out = run / "analysis"
report = {"summary": json.loads((run / "trajectory.summary.json").read_text())}
motion = json.loads((out / "motion.json").read_text())
report["temperature_K"] = {name: value["mean"]
                           for name, value in motion["metrics"]["temperature_K"].items()
                           if isinstance(value, dict) and "mean" in value}
report["partition_eV"] = {name: {"TrajCast": value["SMDP"]["mean"],
                                  "reference": value["reference"]["mean"]}
                          for name, value in motion["metrics"]["kinetic_partition"].items()
                          if isinstance(value, dict) and "SMDP" in value}
report["partition_detail_eV"] = {name: value
                                 for name, value in motion["metrics"]["kinetic_partition"].items()
                                 if isinstance(value, dict) and "SMDP" in value}
rdf = json.loads((out / "rdf.json").read_text())
report["rdf_rmse"] = {k: v["RMSE"] for k, v in rdf["metrics"].items()}
vdos = json.loads((out / "vdos.json").read_text())
report["vdos_overlap"] = {k: v["overlap"] for k, v in vdos["metrics"]["elements"].items()}
diffusion = json.loads((out / "diffusion.json").read_text())
report["diffusion"] = diffusion["metrics"]
(out / "trajcast_2p5ns_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
