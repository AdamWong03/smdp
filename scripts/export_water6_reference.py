"""Export actual run state and curves, never turn missing physics into success."""
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data1/wyd/research/foundation_mlip_adaptation')
RUNS = ROOT/'experiments/water6_reference_20261113'
REPORT = ROOT/'smdp/reports/water6_reference_20261113'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    snapshot = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), experiments={})
    for name in ['dsrt_radius6', 'trajcast_native', 'rejected_wrong_native_class']:
        root = RUNS/name
        if not root.exists():
            continue
        records = [json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()
                   if line.strip()] if (root/'history.jsonl').exists() else []
        status = read(root/'status.json')
        progress = read(root/'progress.json')
        result = dict(run_directory=str(root), status=status, progress=progress,
            metadata=read(root/'metadata.json'), correctness=read(root/'correctness.json'),
            completed_epochs=len(records), history=records,
            test_metrics=read(root/'test_metrics.json'),
            trajectory_completed=False, physical_analysis_completed=False,
            physical_evaluation_status='not_started')
        snapshot['experiments'][name] = result
        if name != 'rejected_wrong_native_class':
            dest = REPORT/name
            dest.mkdir(exist_ok=True)
            for field in ['config.json', 'environment.txt', 'gpu_at_launch.csv', 'source_commit.txt']:
                if (root/field).exists():
                    shutil.copy2(root/field, dest/field)
        if records:
            with (REPORT/f'{name}_history.csv').open('w') as f:
                fields = ['epoch', 'optimizer_steps', 'train_final_loss', 'val_final_loss',
                    'position_RMSE_A', 'velocity_RMSE_A_fs', 'lr', 'epoch_seconds', 'peak_cuda_bytes']
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                for rec in records:
                    row = {key: rec.get(key) for key in fields}
                    row.update(val_final_loss=rec['validation']['selection_loss'],
                        position_RMSE_A=rec['validation']['position_RMSE_A'],
                        velocity_RMSE_A_fs=rec['validation']['velocity_RMSE_A_fs'])
                    writer.writerow(row)
    source = ROOT/'data/smdp_reference_md_v1/water/derived_graphs/water6_reference_20261113/preparation.json'
    shutil.copy2(source, REPORT/'data_preparation.json')
    (REPORT/'training_snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print(json.dumps({name: dict(completed_epochs=value['completed_epochs'],
          status=value['status'], progress=value['progress'])
          for name, value in snapshot['experiments'].items()}, indent=2))


if __name__ == '__main__':
    main()
