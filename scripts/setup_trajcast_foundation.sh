#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/wyd/research/foundation_mlip_adaptation
ENV=/data1/wyd/envs/trajcast-c52f107
UPSTREAM=$ROOT/upstream/trajcast-c52f107
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = c52f107a02176671b21a420b2d4c67e0327b04f7
if ! test -f "$ENV/pyvenv.cfg"; then /usr/bin/python3.12 -m venv "$ENV"; fi
grep -q 'include-system-site-packages = false' "$ENV/pyvenv.cfg"
"$ENV/bin/python" -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
"$ENV/bin/python" -m pip install 'torch_scatter==2.1.2+pt25cu121' 'torch_cluster==1.6.3+pt25cu121' --no-deps --only-binary=:all: -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
"$ENV/bin/python" -m pip install 'numpy==1.26.4' 'scipy==1.14.1' 'e3nn==0.5.9' 'torch-geometric==2.6.1' 'torch-nl==0.3' 'ase==3.24.0' 'h5py==3.12.1' 'pandas==2.2.3' 'matplotlib==3.9.2' 'tensorboard==2.18.0' 'pyyaml==6.0.2' 'typer==0.15.1' 'flit_core==3.10.1' 'pytest==8.3.4'
"$ENV/bin/python" -m pip install --no-build-isolation --no-deps -e "$UPSTREAM"
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > "$ROOT/experiments/trajcast_based/quartz_20261027/environment.lock.txt"
