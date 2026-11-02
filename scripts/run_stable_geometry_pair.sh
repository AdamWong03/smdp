#!/usr/bin/env bash
set -euo pipefail

project_root="${SMDP_PROJECT_ROOT:-/data1/wyd/research/foundation_mlip_adaptation/smdp}"
experiment_root="${SMDP_EXPERIMENT_ROOT:-/data1/wyd/research/foundation_mlip_adaptation/experiments/relational_transition}"
python_bin="${SMDP_PYTHON:-/data1/wyd/envs/trajcast-c52f107/bin/python}"
trajcast_root="${TRAJCAST_ROOT:-/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107}"
export PYTHONPATH="${project_root}:${trajcast_root}${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
    local run_name="$1"
    local report_name="$2"
    local run_dir="${experiment_root}/${run_name}"

    mkdir -p "${run_dir}"
    if [[ ! -f "${run_dir}/config.json" ]]; then
        cp "${project_root}/reports/${report_name}/config.json" "${run_dir}/config.json"
    fi
    "${python_bin}" "${project_root}/train_relational.py" --run "${run_dir}" \
        >> "${run_dir}/train.log" 2>&1
}

run_one quartz30_20261103_r3_opt relational_opt_quartz30_20261103
run_one quartz30_20261103_stable relational_stable_quartz30_20261103
