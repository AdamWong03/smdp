#!/usr/bin/env bash
set -uo pipefail

project_root="/data1/wyd/research/foundation_mlip_adaptation"
data_root="$project_root/data/smdp_reference_md_v1"
python_bin="/data1/wyd/envs/mlip/bin/python"
lammps_bin="/data1/wyd/envs/mlip/bin/lmp"
runtime_lib="/data1/wyd/envs/lammps-runtime/lib"
export PYTHONPATH="$project_root/smdp${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$runtime_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_one() {
    local run_dir="$1"
    if [[ -f "$run_dir/.complete" ]]; then
        return 0
    fi
    (
        cd "$run_dir" || exit 1
        # Hold an advisory lock for the lifetime of LAMMPS.  This prevents a
        # resumed launcher from opening the same dump and log files twice.
        exec 9> .run.lock
        if ! flock -n 9; then
            echo "Run already active: $run_dir" >&2
            exit 0
        fi
        if [[ -f .complete ]]; then
            exit 0
        fi
        "$lammps_bin" -in in.lammps > run.log 2>&1
        status=$?
        if [[ $status -eq 0 ]]; then
            date --iso-8601=seconds > .complete
        else
            echo "$status" > .failed
        fi
        exit "$status"
    )
}
export -f run_one
export lammps_bin

mapfile -t run_dirs < <(
    find "$data_root/quartz/physical_reference" "$data_root/water/physical_reference" \
         "$data_root/quartz/raw_nve" "$data_root/water/raw_nve" \
         -mindepth 1 -type f -name in.lammps -printf '%h\n' | sort
)
printf '%s\n' "${run_dirs[@]}" > "$data_root/production_run_directories.txt"
printf '%s\0' "${run_dirs[@]}" | xargs -0 -n1 -P12 bash -c 'run_one "$1"' _
simulation_status=$?
if [[ $simulation_status -ne 0 ]]; then
    echo "One or more LAMMPS production runs failed" >&2
    exit "$simulation_status"
fi

convert_nve() {
    local run_dir="$1"
    local system="$2"
    local dt="$3"
    [[ -f "$run_dir/trajectory.h5" ]] && return 0
    "$python_bin" "$project_root/smdp/reference_md/convert_lammps_h5.py" \
        "$run_dir" trajectory.lammpstrj.gz "$run_dir/trajectory.h5" \
        --system "$system" --dt-fs "$dt" > "$run_dir/conversion.json"
}

while IFS= read -r run_dir; do convert_nve "$run_dir" quartz 1.0; done \
    < <(find "$data_root/quartz/raw_nve" -mindepth 1 -maxdepth 1 -type d | sort)
while IFS= read -r run_dir; do convert_nve "$run_dir" water 0.5; done \
    < <(find "$data_root/water/raw_nve" -mindepth 1 -maxdepth 1 -type d | sort)

for run_dir in "$data_root"/quartz/physical_reference/short_nve/replica_*; do
    convert_nve "$run_dir" quartz 1.0
done

convert_reference() {
    local run_dir="$1" dump_name="$2" output_name="$3" system="$4" dt="$5" log_name="$6"
    [[ -f "$run_dir/$output_name" ]] && return 0
    "$python_bin" "$project_root/smdp/reference_md/convert_lammps_h5.py" \
        "$run_dir" "$dump_name" "$run_dir/$output_name" --system "$system" --dt-fs "$dt" \
        --omit-derived-momenta --omit-redundant-coordinates \
        > "$run_dir/$log_name"
}

convert_reference "$data_root/quartz/physical_reference/nvt_15ns" production.lammpstrj.gz production.h5 quartz 1.0 conversion.json
convert_reference "$data_root/quartz/physical_reference/nvt_15ns" thermalization.lammpstrj.gz thermalization.h5 quartz 1.0 thermalization_conversion.json
convert_reference "$data_root/water/physical_reference/nvt_2p5ns" production.lammpstrj.gz production.h5 water 0.5 conversion.json
convert_reference "$data_root/water/physical_reference/nvt_2p5ns" thermalization.lammpstrj.gz thermalization.h5 water 0.5 thermalization_conversion.json

mkdir -p "$data_root/quartz/transitions" "$data_root/water/transitions"
if [[ ! -f "$data_root/quartz/transitions/30fs/test.h5" ]]; then
    "$python_bin" "$project_root/smdp/reference_md/build_transitions.py" \
        "$data_root/quartz" --horizons-fs 8 16 30 --seed 2026102001 \
        > "$data_root/quartz/transitions/build.json"
fi
if [[ ! -f "$data_root/water/transitions/10fs/test.h5" ]]; then
    "$python_bin" "$project_root/smdp/reference_md/build_transitions.py" \
        "$data_root/water" --horizons-fs 2 5 10 --seed 2026102001 \
        > "$data_root/water/transitions/build.json"
fi

date --iso-8601=seconds > "$data_root/.complete"
