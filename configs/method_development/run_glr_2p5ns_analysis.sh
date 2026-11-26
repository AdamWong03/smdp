#!/usr/bin/env bash
# Analyze the GLR 2.5 ns production run after the writer has finished.
# The rollout stores ensemble='nve' because no external CSVR is applied; the GLR
# readout is the thermostat, so tag the trajectory 'nvt' for the equilibrium tasks.
set -euo pipefail

ROOT=/data2/wyd/research/foundation_mlip_adaptation
DATA=/data1/wyd/research/foundation_mlip_adaptation
RUN=${1:-$ROOT/experiments/glr_2p5ns_20261122}
LABEL=${2:-"GLR AR(10) + rescale, 2.5 ns"}

test -f "$RUN/trajectory.summary.json" || { echo "run not finished: $RUN"; exit 1; }
test -f "$RUN/trajectory.h5" || { echo "missing trajectory: $RUN"; exit 1; }

/data1/wyd/envs/mlip/bin/python - "$RUN/trajectory.h5" <<'PY'
import sys
import h5py
path = sys.argv[1]
with h5py.File(path, 'r+') as handle:
    handle.attrs['ensemble'] = 'nvt'
    handle.attrs['thermostat'] = 'GLR'
    handle.flush()
print('tagged', path)
PY

cd "$ROOT/code/h1h2"
export PYTHONPATH=$DATA/upstream/trajcast-c52f107:$PWD
export LD_LIBRARY_PATH=/data1/wyd/envs/lammps-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

/data1/wyd/envs/mlip/bin/python -u analyze_water.py \
  --candidate "$RUN/trajectory.h5" \
  --reference "$DATA/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5" \
  --topology "$DATA/data/smdp_reference_md_v1/water/equilibration/snapshots/water_5.data" \
  --output "$RUN/analysis" --tasks motion rdf vdos diffusion --frames 4000 \
  --model-label "$LABEL"

/data1/wyd/envs/mlip/bin/python - "$RUN/analysis" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
report = {}
motion = json.loads((out / 'motion.json').read_text())
report['temperature_K'] = {name: value['mean']
                           for name, value in motion['metrics']['temperature_K'].items()
                           if isinstance(value, dict) and 'mean' in value}
report['partition_eV'] = {name: {'GLR': value['SMDP']['mean'],
                                  'reference': value['reference']['mean']}
                          for name, value in motion['metrics']['kinetic_partition'].items()
                          if isinstance(value, dict) and 'SMDP' in value}
report['partition_detail_eV'] = {name: value
                                 for name, value in motion['metrics']['kinetic_partition'].items()
                                 if isinstance(value, dict) and 'SMDP' in value}
rdf = json.loads((out / 'rdf.json').read_text())
report['rdf_rmse'] = {k: v['RMSE'] for k, v in rdf['metrics'].items()}
vdos = json.loads((out / 'vdos.json').read_text())
report['vdos_overlap'] = {k: v['overlap'] for k, v in vdos['metrics']['elements'].items()}
diffusion = json.loads((out / 'diffusion.json').read_text())
report['diffusion'] = diffusion['metrics']
(out / 'glr_2p5ns_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
