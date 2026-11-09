"""Export small progress/config artifacts; never export weights or state banks."""
import datetime
import json
import subprocess
from pathlib import Path

from conditional_transition.common import ROOT,ARMS
from smdpr.train import write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    report=Path(__file__).resolve().parents[1]/'reports/conditional_transition_20261109'
    report.mkdir(parents=True,exist_ok=True)
    records=[];configs=[]
    for system in ['quartz','water']:
        for arm in ARMS:
            run=ROOT/system/arm
            configs.append(read(run/'config.json'))
            row=dict(system=system,arm=arm,run=str(run),pipeline=read(run/'pipeline_status.json'),
                training=read(run/'status.json'),progress=read(run/'progress.json'),
                evaluation=read(run/'evaluation_status.json'),
                collection=[read(run/f'collection_round{i}/status.json') for i in [1,2]],
                collection_progress=[read(run/f'collection_round{i}/collection_progress.json') for i in [1,2]])
            hist=run/'history.jsonl'
            if hist.exists():
                row['validation_history']=[json.loads(line) for line in hist.read_text().splitlines()]
            records.append(row)
    write_json(report/'configs.json',configs)
    write_json(report/'numerical_checks.json',read(ROOT/'numerical_checks.json'))
    analysis_complete=all(r['evaluation'] and r['evaluation'].get('validation_physics_completed',False)
                          for r in records)
    write_json(report/'progress_snapshot.json',dict(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpu_snapshot=subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.free,utilization.gpu,uuid','--format=csv'],text=True),
        records=records,physical_conclusions_available=analysis_complete,
        note='Training status/progress describe training only; pipeline/evaluation status are authoritative for analysis. '
             'Completed analysis is not a guarantee of stable/correct NVE. No test selection or new long NVT.'))


if __name__=='__main__':main()
