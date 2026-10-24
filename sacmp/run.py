"""Reproducible SACMP orchestration; intentionally never rewrites old SMDP runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--full", action="store_true",
                   help="Also run 100ps NVE and 15ns NVT after short screens")
    p.add_argument("--after-training", action="store_true",
                   help="Use an already completed SACMP training checkpoint")
    p.add_argument("--dense", action="store_true",
                   help="Store 1fs velocities for compatible VDOS analysis")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    def call(label, module, *flags):
        log = args.output/(label+".log")
        cmd = [sys.executable, "-m", module, *map(str, flags)]
        print("Running", " ".join(cmd), flush=True)
        with log.open("a") as handle:
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{label} returned {result.returncode}; inspect {log}")
    if not args.after_training:
        call("train", "sacmp.train", "--data-root", args.data_root,
             "--output", args.output, "--device", args.device)
    completed = args.output/"training_complete.json"
    if not completed.is_file():
        raise RuntimeError("Refusing held-out evaluation of unfinished training")
    checkpoint = args.output/"best_rollout.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (args.output/"test/test_metrics.json").exists():
        call("test", "sacmp.evaluate", "--data-root", args.data_root,
             "--checkpoint", checkpoint, "--output", args.output/"test",
             "--device", args.device)
    # Stage A: limited autonomous screening on independent test trajectories.
    for stem in ["test_09", "test_10"]:
        initial = args.data_root/f"quartz/raw_nve/{stem}/trajectory.h5"
        candidate = args.output/f"nve_screen_{stem}.h5"
        options = ["--dense"] if args.dense else []
        if not candidate.exists():
            call("nve_screen_"+stem, "sacmp.rollout",
                 "--checkpoint", checkpoint, "--initial", initial,
                 "--output", candidate, "--ensemble", "nve",
                 "--steps", 625, "--device", args.device, *options)
        report = json.loads(candidate.with_suffix(".summary.json").read_text())
        if not report["stable"] or report["completed_steps"] < 625:
            raise RuntimeError("NVE screening failed; do not run expensive NVT")
        if args.dense and not (args.output/f"nve_screen_{stem}_analysis/metrics.json").exists():
            if not report["dense_queries"]:
                raise RuntimeError("Existing screen has no 1fs data for VDOS")
            call("analyze_screen_"+stem, "analyze_quartz",
                 "--candidate", candidate, "--reference", initial,
                 "--output", args.output/f"nve_screen_{stem}_analysis")
    if args.full:
        for stem in ["test_09", "test_10"]:
            initial = args.data_root/f"quartz/raw_nve/{stem}/trajectory.h5"
            candidate = args.output/f"nve_{stem}.h5"
            options = ["--dense"] if args.dense else []
            if not candidate.exists():
                call("nve_full_"+stem, "sacmp.rollout",
                     "--checkpoint", checkpoint, "--initial", initial,
                     "--output", candidate, "--ensemble", "nve",
                     "--steps", 6250, "--device", args.device, *options)
            report = json.loads(candidate.with_suffix(".summary.json").read_text())
            if not report["stable"] or report["completed_steps"] < 6250:
                raise RuntimeError("Full NVE failed; halt before NVT")
            if args.dense and not (args.output/f"nve_{stem}_analysis/metrics.json").exists():
                if not report["dense_queries"]:
                    raise RuntimeError("Existing full NVE run has no dense data")
                call("analyze_nve_"+stem, "analyze_quartz",
                     "--candidate", candidate, "--reference", initial,
                     "--output", args.output/f"nve_{stem}_analysis")
        nvt = args.data_root/"quartz/physical_reference/nvt_15ns/production.h5"
        candidate = args.output/"nvt.h5"
        options = ["--dense"] if args.dense else []
        if not candidate.exists():
            call("nvt_full", "sacmp.rollout",
                 "--checkpoint", checkpoint, "--initial", nvt,
                 "--output", candidate, "--ensemble", "nvt",
                 "--steps", 937500, "--thermalization-steps", 93750,
                 "--thermostat-metadata", nvt.parent/"metadata.json",
                 "--device", args.device, *options)
        if args.dense and not (args.output/"nvt_analysis/metrics.json").exists():
            call("analyze_nvt", "analyze_quartz",
                 "--candidate", candidate, "--reference", nvt,
                 "--output", args.output/"nvt_analysis")
    (args.output/"sacmp_pipeline_complete.json").write_text(json.dumps(
        {"full":args.full,"dense":args.dense,"note":
         "No BKS force during propagation; run physics evaluation offline. "
         "Output files are never overwritten."}, indent=2))


if __name__ == "__main__":
    main()
