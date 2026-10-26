"""Export bounded, observed SACMP results; never copy weights or trajectories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    args = p.parse_args()
    src, dst = args.output, args.report_dir
    dst.mkdir(parents=True, exist_ok=True)
    names = ["config.json", "anchor_report.json", "training_complete.json",
             "sacmp_pipeline_complete.json", "test/test_metrics.json", "test/query_errors.csv"]
    for run in ("nve_screen_test_09", "nve_screen_test_10"):
        names.append(run + ".summary.json")
        names.extend(run + "_analysis/" + name for name in (
            "metrics.json", "energy.png", "energy_partition.png", "rdf.png", "rdf.csv",
            "vdos_macro_16fs.png", "vdos_queried_1fs.png", "vdos_macro_16fs.csv",
            "vdos_queried_1fs.csv", "thermodynamics.csv", "kinetic.csv",
            "nve_trajectory_error.csv"))
    for name in names:
        source = src / name
        if source.stat().st_size > 4 * 1024**2:
            raise ValueError(f"Unexpectedly large report artifact: {name}")
        target = dst / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    history = [json.loads(line) for line in (src / "history.jsonl").read_text().splitlines()]
    rows = []
    for row in history:
        item = {key: row[key] for key in ("phase", "epoch", "lr", "elapsed_s", "peak_gpu_bytes")}
        item.update({split + "_" + key: value for split in ("training", "validation")
                     for key, value in row[split].items()})
        rows.append(item)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (dst / "training_curve.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, phase in zip(axes, ("segment", "rollout")):
        selected = [row for row in history if row["phase"] == phase]
        for split in ("training", "validation"):
            ax.plot([row["epoch"] + 1 for row in selected],
                    [row[split]["loss"] for row in selected], label=split)
        ax.set(title=phase + " (phase-specific objective)", xlabel="Epoch", ylabel="Loss", yscale="log")
        ax.legend()
    fig.tight_layout()
    fig.savefig(dst / "training_curve.png", dpi=160)
    plt.close(fig)
    ckpt_hash = hashlib.sha256((src / "best_rollout.pt").read_bytes()).hexdigest()
    summary = dict(checkpoint_sha256=ckpt_hash, server_output=str(src),
                   training_completed=True, independent_test_samples=1000,
                   nve_duration_ps=10, nve={}, nve_100ps_completed=False,
                   nvt_15ns_completed=False)
    for run in ("nve_screen_test_09", "nve_screen_test_10"):
        run_info = json.loads((src / (run + ".summary.json")).read_text())
        assert run_info["checkpoint_sha256"] == ckpt_hash
        data = np.genfromtxt(src / (run + "_analysis/thermodynamics.csv"), delimiter=",", names=True)
        record = dict(trajectory_completed=run_info["completed_steps"] == 625,
                      physical_analysis_completed=json.loads(
                          (src / (run + "_analysis/metrics.json")).read_text())["physical_analysis_completed"])
        for label in ("SMDP", "reference"):
            delta = data[label + "_total_eV"] - data[label + "_total_eV"][0]
            record[label] = dict(energy_change_RMS_eV=float(np.sqrt(np.mean(delta**2))),
                                 energy_change_min_eV=float(delta.min()),
                                 energy_change_max_eV=float(delta.max()))
        summary["nve"][run] = record
    (dst / "result_status.json").write_text(json.dumps(summary, indent=2))
    print(f"Exported {len(names)} bounded source artifacts to {dst}")


if __name__ == "__main__":
    main()
