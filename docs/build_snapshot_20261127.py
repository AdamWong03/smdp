"""Build the 2026-11-27 reproducible snapshot manifest and checksums.

The snapshot copies only small code/config/parameter/result/log files. Large
trajectories are referenced by path in data_sources.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "reproducibility" / "snapshot_20261127"

CODE = [
    "code/h1h2/fit_glr.py",
    "code/h1h2/glr_readout.py",
    "code/h1h2/fit_velocity_gle.py",
    "code/h1h2/gle_kernel_study.py",
    "code/h1h2/gle_kernel_study_mean.py",
    "code/h1h2/fit_gle_discrete.py",
    "code/h1h2/fit_gle_discrete_exact.py",
    "code/h1h2/fit_gle_general.py",
    "code/h1h2/verify_gle_discrete.py",
    "code/h1h2/audit_glr_coefficients.py",
    "code/h1h2/reference_acf_blocks.py",
    "code/h1h2/position_velocity_audit.py",
    "code/h1h2/calibration_dependence.py",
    "code/h1h2/feedback_probe.py",
    "code/h1h2/paired_feedback.py",
    "code/h1h2/analyze_feedback.py",
    "code/h1h2/audit_glr_ablations.py",
    "code/h1h2/physics/spcfw.py",
    "code/h1h2/physics/units.py",
    "code/h1h2/scripts/native_nvt_screen.py",
    "code/h1h2/analyze_water.py",
    "code/h1h2/rollout_relational.py",
    "code/h1h2/fd_readout.py",
    "code/h1h2/motion_loss.py",
    "code/h1h2/driven_gle.py",
    "code/h1h2/position_consistent.py",
    "code/h1h2/fit_slow_mode.py",
    "code/h1h2/species_glr.py",
    "code/h1h2/fit_species_glr.py",
    "code/h1h2/phonon_glr.py",
]

CONFIGS = [
    "method_development/run_glr_postgate.sh",
    "method_development/run_glr_ablations.sh",
    "method_development/run_glr_ltrans_sweep.sh",
    "method_development/run_glr_2p5ns_analysis.sh",
    "method_development/run_glr_mixed_ltrans_sweep.sh",
    "experiments/published_trajcast_2p5ns_20261126/analyze_when_done.sh",
]

PARAMETERS = [
    ("experiments/glr_fit_20261122/glr_coefficients.json",
     "parameters/glr_coefficients_production.json"),
    ("experiments/glr_fit_heldout0_20261122/glr_coefficients.json",
     "parameters/glr_coefficients_heldout0.json"),
    ("experiments/glr_provenance_20261127.json",
     "parameters/glr_provenance_20261127.json"),
    ("experiments/quartz5_glr_fit_20261122/species_coefficients.json",
     "parameters/quartz5_species_coefficients.json"),
    ("experiments/fit_gle_general_overdamped_20261126.json",
     "parameters/fit_gle_general_overdamped_20261126.json"),
    ("experiments/fit_gle_general_6mode_only_20261126.json",
     "parameters/fit_gle_general_6mode_only_20261126.json"),
    ("experiments/fit_gle_discrete_exact_20261126.json",
     "parameters/fit_gle_discrete_exact_20261126.json"),
    ("experiments/reference_acf_blocks_20261126.json",
     "parameters/reference_acf_blocks_20261126.json"),
    ("experiments/reference_acf_blocks_20261126.npz",
     "parameters/reference_acf_blocks_20261126.npz"),
    ("experiments/gle_kernel_study_mean_20261126.json",
     "parameters/gle_kernel_study_mean_20261126.json"),
    ("experiments/fit_glr_long_20261126.json",
     "parameters/fit_glr_long_20261126.json"),
    ("experiments/glr_coefficients_audit_20261127.json",
     "parameters/glr_coefficients_audit_20261127.json"),
    ("experiments/verify_gle_discrete_5mode_20261126.json",
     "parameters/verify_gle_discrete_5mode_20261126.json"),
    ("experiments/gle_kernel_study_20261126.json",
     "parameters/gle_kernel_study_20261126.json"),
    ("experiments/glr_provenance_checksums_20261127.txt",
     "parameters/glr_provenance_checksums_20261127.txt"),
    ("experiments/position_velocity_audit_20261127.json",
     "parameters/position_velocity_audit_20261127.json"),
    ("experiments/calibration_dependence_20261127.json",
     "parameters/calibration_dependence_20261127.json"),
    ("experiments/calib_independent_100a_20261127/glr_coefficients.json",
     "parameters/calib_independent_100a_glr_coefficients.json"),
    ("experiments/calib_independent_100b_20261127/glr_coefficients.json",
     "parameters/calib_independent_100b_glr_coefficients.json"),
    ("experiments/calib_independent_200_20261127/glr_coefficients.json",
     "parameters/calib_independent_200_glr_coefficients.json"),
    ("experiments/calib_independent_100a_20261127/ar_audit.json",
     "parameters/calib_independent_100a_ar_audit.json"),
    ("experiments/calib_independent_100b_20261127/ar_audit.json",
     "parameters/calib_independent_100b_ar_audit.json"),
    ("experiments/calib_independent_200_20261127/ar_audit.json",
     "parameters/calib_independent_200_ar_audit.json"),
    ("experiments/feedback_probe_initial_20261127.json",
     "parameters/feedback_probe_initial_20261127.json"),
    ("experiments/feedback_probe_glr100ps_20261127.json",
     "parameters/feedback_probe_glr100ps_20261127.json"),
    ("experiments/glr_ablation_audit_20261127.json",
     "parameters/glr_ablation_audit_20261127.json"),
    ("experiments/feedback_intervention_analysis_20261127.json",
     "parameters/feedback_intervention_analysis_20261127.json"),
    ("experiments/paired_feedback_200steps_20261127.json",
     "parameters/paired_feedback_200steps_20261127.json"),
]

RESULTS = [
    ("experiments/glr_2p5ns_20261122/analysis/glr_2p5ns_report.json",
     "results/glr_2p5ns_report_deterministic.json"),
    ("experiments/glr_stoch_2p5ns_20261122/analysis/glr_2p5ns_report.json",
     "results/glr_2p5ns_report_stochastic.json"),
    ("experiments/nvt_screen_published_trajcast_20261121/analysis/"
     "published_trajcast_100ps_report.json",
     "results/published_trajcast_100ps_report.json"),
    ("experiments/nvt_screen_published_trajcast_20261121/trajectory.summary.json",
     "results/published_trajcast_100ps_summary.json"),
    ("experiments/e1_baseline_20k_20261120/physics/nve_val_20ps/analysis/"
     "reference_verification_fixed.json",
     "results/reference_verification_fixed_e1_baseline.json"),
    ("experiments/e1_motion_20k_20261120/physics/nve_val_20ps/analysis/"
     "reference_verification_fixed.json",
     "results/reference_verification_fixed_e1_motion.json"),
    ("experiments/feedback_full_5ps_20261127/trajectory.summary.json",
     "results/feedback_full_5ps_summary.json"),
    ("experiments/feedback_no_feedback_5ps_20261127/trajectory.summary.json",
     "results/feedback_no_feedback_5ps_summary.json"),
    ("experiments/feedback_all_no_feedback_5ps_20261127/trajectory.summary.json",
     "results/feedback_all_no_feedback_5ps_summary.json"),
    ("experiments/feedback_thermostat_only_5ps_20261127/trajectory.summary.json",
     "results/feedback_thermostat_only_5ps_summary.json"),
    ("experiments/feedback_full_100ps_20261127/partial_status.json",
     "results/feedback_full_100ps_partial_status.json"),
]

LOGS = [
    ("experiments/reference_acf_blocks_20261126.log",
     "logs/reference_acf_blocks_20261126.log"),
    ("experiments/gle_kernel_study_mean_20261126.log",
     "logs/gle_kernel_study_mean_20261126.log"),
    ("experiments/gle_kernel_study_20261126.log",
     "logs/gle_kernel_study_20261126.log"),
    ("experiments/fit_glr_long_20261126.log",
     "logs/fit_glr_long_20261126.log"),
    ("experiments/fit_gle_discrete_20261126.log",
     "logs/fit_gle_discrete_20261126.log"),
    ("experiments/fit_gle_discrete_exact_20261126.log",
     "logs/fit_gle_discrete_exact_20261126.log"),
    ("experiments/fit_gle_general_20261126.log",
     "logs/fit_gle_general_20261126.log"),
    ("experiments/fit_gle_general_overdamped_20261126.log",
     "logs/fit_gle_general_overdamped_20261126.log"),
    ("experiments/fit_gle_general_6mode_only_20261126.log",
     "logs/fit_gle_general_6mode_only_20261126.log"),
    ("experiments/verify_gle_discrete_5mode_20261126.log",
     "logs/verify_gle_discrete_5mode_20261126.log"),
    ("experiments/glr_coefficients_audit_20261127.log",
     "logs/glr_coefficients_audit_20261127.log"),
    ("experiments/nvt_screen_published_trajcast_20261121/analysis_diffusion.log",
     "logs/published_trajcast_100ps_diffusion.log"),
    ("experiments/published_trajcast_2p5ns_20261126/run.log",
     "logs/published_trajcast_2p5ns_partial_run.log"),
    ("experiments/position_velocity_audit_20261127.log",
     "logs/position_velocity_audit_20261127.log"),
    ("experiments/calibration_dependence_20261127.log",
     "logs/calibration_dependence_20261127.log"),
    ("experiments/calib_independent_100a_20261127/fit.log",
     "logs/calib_independent_100a_fit.log"),
    ("experiments/calib_independent_100b_20261127/fit.log",
     "logs/calib_independent_100b_fit.log"),
    ("experiments/calib_independent_200_20261127/fit.log",
     "logs/calib_independent_200_fit.log"),
    ("experiments/calib_independent_100a_20261127/ar_audit.log",
     "logs/calib_independent_100a_ar_audit.log"),
    ("experiments/calib_independent_100b_20261127/ar_audit.log",
     "logs/calib_independent_100b_ar_audit.log"),
    ("experiments/calib_independent_200_20261127/ar_audit.log",
     "logs/calib_independent_200_ar_audit.log"),
    ("experiments/feedback_probe_initial_20261127.log",
     "logs/feedback_probe_initial_20261127.log"),
    ("experiments/feedback_probe_glr100ps_20261127.log",
     "logs/feedback_probe_glr100ps_20261127.log"),
    ("experiments/glr_ablation_audit_20261127.log",
     "logs/glr_ablation_audit_20261127.log"),
    ("experiments/glr_ltrans_analysis_20261127.log",
     "logs/glr_ltrans_analysis_20261127.log"),
    ("experiments/feedback_full_5ps_20261127/rollout.log",
     "logs/feedback_full_5ps_rollout.log"),
    ("experiments/feedback_no_feedback_5ps_20261127/rollout.log",
     "logs/feedback_no_feedback_5ps_rollout.log"),
    ("experiments/feedback_all_no_feedback_5ps_20261127/rollout.log",
     "logs/feedback_all_no_feedback_5ps_rollout.log"),
    ("experiments/feedback_thermostat_only_5ps_20261127/rollout.log",
     "logs/feedback_thermostat_only_5ps_rollout.log"),
    ("experiments/feedback_intervention_analysis_20261127.log",
     "logs/feedback_intervention_analysis_20261127.log"),
    ("experiments/paired_feedback_500steps_20261127.log",
     "logs/paired_feedback_500steps_20261127.log"),
    ("experiments/paired_feedback_200steps_20261127.log",
     "logs/paired_feedback_200steps_20261127.log"),
]

DOCS = [
    ("SMDP_ICML2027_STATUS_AUDIT.md", "docs/SMDP_ICML2027_STATUS_AUDIT.md"),
    ("paper/CLAIM_BOUNDARY.md", "docs/CLAIM_BOUNDARY.md"),
    ("reproducibility/build_snapshot_20261127.py",
     "docs/build_snapshot_20261127.py"),
    ("method_development/GLR_STATE_TRANSITION_20261127.md",
     "docs/GLR_STATE_TRANSITION_20261127.md"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    SNAP.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []

    def copy_one(src_rel: str, dst_rel: str | None = None):
        src = ROOT / src_rel
        if not src.exists():
            missing.append(src_rel)
            return
        dst = SNAP / (dst_rel or src_rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dict(original_path=src_rel,
                           snapshot_path=str(dst.relative_to(SNAP)),
                           size_bytes=dst.stat().st_size,
                           sha256=sha256(dst)))

    for rel in CODE:
        copy_one(rel)
    for rel in CONFIGS:
        copy_one(rel, f"configs/{rel}")
    for src, dst in PARAMETERS:
        copy_one(src, dst)
    for src, dst in RESULTS:
        copy_one(src, dst)
    for src, dst in LOGS:
        copy_one(src, dst)
    for src, dst in DOCS:
        copy_one(src, dst)

    readme = SNAP / "README.md"
    if readme.exists():
        copied.append(dict(original_path="(snapshot README)",
                           snapshot_path="README.md",
                           size_bytes=readme.stat().st_size,
                           sha256=sha256(readme)))

    # Large read-only artifacts: reference them, do not copy.
    checksum_files = [
        ROOT / "experiments/reference_sha256_20261127.txt",
        ROOT / "experiments/data_sources_checksums_20261127.txt",
    ]
    checksums = {}
    for checksum_file in checksum_files:
        if not checksum_file.exists():
            continue
        for line in checksum_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                checksums[str(Path(parts[1]).resolve())] = parts[0]
    large = [
        ("reference_nvt_2p5ns",
         "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/"
         "water/physical_reference/nvt_2p5ns/production.h5",
         "calibration + evaluation; 0.5 fs storage; 5,000,001 frames"),
        ("independent_nvt_200ps",
         "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/"
         "water/physical_reference/train_nvt_200ps/production.h5",
         "independent NVT trajectory used for ACF validation"),
        ("independent_nvt_500ps_dump",
         "/data2/wyd/research/foundation_mlip_adaptation/experiments/"
         "independent_nvt_500ps_20261121/production.lammpstrj.gz",
         "LAMMPS NVT independent start; text dump"),
        ("water_topology",
         "/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/"
         "water/equilibration/snapshots/water_5.data",
         "SPC/Fw topology; 64 molecules, 192 atoms"),
        ("published_trajcast_checkpoint",
         "/data2/wyd/research/foundation_mlip_adaptation/reference/"
         "trajcast_published_water/published_checkpoint.pt",
         "IBM published water e3nn state; SHA256 known"),
    ]
    data_sources = []
    for role, path, note in large:
        p = Path(path)
        entry = dict(role=role, path=path,
                     exists=p.exists(),
                     size_bytes=p.stat().st_size if p.exists() else None,
                     mtime=datetime.fromtimestamp(
                         p.stat().st_mtime, timezone.utc).isoformat()
                     if p.exists() else None,
                     sha256=checksums.get(str(Path(path).resolve())),
                     note=note)
        if role == "published_trajcast_checkpoint":
            entry["sha256"] = ("a7f189542d655b7675ebc74bfce11b4e36772e68b448fae4ba5a3f1eb15a62ea")
        data_sources.append(entry)

    manifest = dict(
        created=datetime.now(timezone.utc).isoformat(),
        snapshot=str(SNAP),
        description=("Reproducible snapshot of the GLR/GLE audit, 2026-11-27. "
                     "Large trajectories are referenced in data_sources.json."),
        seeds=dict(
            glr_production_rollouts=[2026110401, 2026110402, 2026110403],
            gle_fit="20261126 + 1000*modes",
            monte_carlo=20261126,
            trajcast_2p5ns=2026110401,
        ),
        environment=dict(
            main="/data1/wyd/envs/mlip (Python 3.12.3, numpy 2.2.6, h5py 3.16.0, "
                 "torch 2.13.0+cu130, LAMMPS 20250722, ASE 3.29.0)",
            cpu_fit="/data2/wyd/tmp_opencode/env_mace (Python 3.11, numpy/scipy)",
            trajcast_upstream="/data1/wyd/research/foundation_mlip_adaptation/upstream/trajcast-c52f107",
        ),
        files=copied,
        missing=missing,
        data_sources=data_sources,
    )
    (SNAP / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (SNAP / "data_sources.json").write_text(
        json.dumps(dict(note="Large read-only artifacts; not copied.",
                        artifacts=data_sources), indent=2) + "\n")

    lines = [f"{item['sha256']}  {item['snapshot_path']}" for item in copied]
    lines.append(f"{sha256(SNAP / 'MANIFEST.json')}  MANIFEST.json")
    (SNAP / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    print(f"copied {len(copied)} files; missing {len(missing)}")
    if missing:
        print("missing:", missing)
    print("wrote", SNAP / "MANIFEST.json")


if __name__ == "__main__":
    main()
