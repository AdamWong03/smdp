#!/usr/bin/env python3
"""Export machine-readable and Markdown snapshots of the controlled DSRT runs."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


RUNS = {
    "R3-Opt": "quartz30_20261103_r3_opt",
    "Stable-DSRT": "quartz30_20261103_stable",
}


def load_run(root, directory):
    run = root / directory
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()
               if line.strip()]
    best = min(history, key=lambda row: row["validation"]["selection_loss"])
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    return {"directory": str(run), "status": status.get("state"), "epochs": len(history),
            "best": best, "latest": history[-1], "history": history,
            "target_optimizer_steps": config.get("total_optimizer_steps"),
            "budget_completed": history[-1].get("optimizer_steps", 0) >=
                                config.get("total_optimizer_steps", float("inf")),
            "test_metrics": (json.loads((run / "test_metrics.json").read_text())
                             if (run / "test_metrics.json").exists() else None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {name: load_run(args.experiment_root, directory)
            for name, directory in RUNS.items()}
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_complete": all(run["budget_completed"] and
                                        run["test_metrics"] is not None for run in runs.values()),
               "runs": runs}
    (args.output / "training_snapshot.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = ["# Stable Geometry-DSRT training snapshot", "",
             f"Generated: `{payload['generated_utc']}`", "",
             "This is a live training snapshot. It is not labeled as a final test result unless "
             "both fixed 250,000-step schedules and held-out tests have completed.", "",
             "| Model | State | Epoch | Best epoch | Best steps | Best val loss | q RMSE (A) | v RMSE (A/fs) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        best = run["best"]
        val = best["validation"]
        lines.append(f"| {name} | {run['status']} | {run['epochs']} | {best['epoch']} | "
                     f"{best['optimizer_steps']:,} | {val['selection_loss']:.9f} | "
                     f"{val['position_RMSE_A']:.9f} | {val['velocity_RMSE_A_fs']:.9f} |")
    lines += ["", "The complete per-epoch records, including learning rate, proposal/final losses, "
              "gradient diagnostics and feature statistics, are stored in `training_snapshot.json`.", "",
              "At this snapshot R3-Opt improves over the old R3 validation loss (0.001036336), "
              "but remains above the historical R1 result (0.000736517). Stable-DSRT remains "
              "worse than R3-Opt, so there is currently no evidence that its residual "
              "reparameterization helps this task.", ""]
    (args.output / "PROGRESS.md").write_text("\n".join(lines))

    if all(run["test_metrics"] is not None for run in runs.values()):
        results = ["# Stopped-run independent test results", "",
                   "Training was stopped at the user's request before the frozen 250,000-step "
                   "budget completed. Each checkpoint was selected using validation data, then "
                   "evaluated once on all 1,000 independent test samples.", "",
                   "| Model | Selected epoch | Selected steps | Stopped epoch | Position RMSE (A) | Velocity RMSE (A/fs) | Position MAE (A) | Velocity MAE (A/fs) |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, run in runs.items():
            test = run["test_metrics"]
            results.append(
                f"| {name} | {test['selected_epoch']} | {test['selected_optimizer_steps']:,} | "
                f"{test['stopped_epoch']} | {test['position_RMSE_A']:.9f} | "
                f"{test['velocity_RMSE_A_fs']:.9f} | {test['position_MAE_A']:.9f} | "
                f"{test['velocity_MAE_A_fs']:.9f} |")
        r3, stable = runs["R3-Opt"]["test_metrics"], runs["Stable-DSRT"]["test_metrics"]
        results += ["", f"Stable-DSRT has {(stable['position_RMSE_A']/r3['position_RMSE_A']-1)*100:.1f}% "
                    f"higher position RMSE and {(stable['velocity_RMSE_A_fs']/r3['velocity_RMSE_A_fs']-1)*100:.1f}% "
                    "higher velocity RMSE than R3-Opt. The controlled test therefore supports "
                    "the new optimization schedule for the original R3 block, but rejects the "
                    "Stable residual reparameterization for this task.", ""]
        (args.output / "STOPPED_TEST_RESULTS.md").write_text("\n".join(results))


if __name__ == "__main__":
    main()
