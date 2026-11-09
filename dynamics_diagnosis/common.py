import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import h5py

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
DATA = PROJECT/'data/smdp_reference_md_v1'
RUNS = PROJECT/'experiments/smdpr'
ARTIFACTS = PROJECT/'experiments/dynamics_diagnosis_20261109'
REPORT = REPO/'reports/dynamics_diagnosis_20261109'
SEED = 20261109


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def checkpoints(system):
    tag = 'quartz30' if system=='quartz' else 'water5'
    base = PROJECT/'experiments/relational_transition'/f'{tag}_20261103_r3_opt'
    return dict(r3opt=base/'best.pt',
        r3opt_two_step=RUNS/f'{tag}_r3opt_matched_two_step/best.pt',
        k3_stage1=RUNS/f'{tag}_20261107_k3/best.pt',
        k3_stage2=RUNS/f'{tag}_20261107_k3/two_step/best.pt',
        k1_stage1=RUNS/f'{tag}_20261107_k1/best.pt',
        k1_stage2=RUNS/f'{tag}_20261107_k1/two_step/best.pt')


def source(system):
    return DATA/system/'raw_nve'/('val_07' if system=='quartz' else 'val_04')/'trajectory.h5'


def initial_sources(system):
    if system=='quartz':
        return [(source(system),0),(DATA/system/'raw_nve/val_08/trajectory.h5',0)]
    return [(source(system),0),(source(system),10000)]


def inventory():
    rows = []
    split_sources={}
    for system in ['quartz', 'water']:
        h=30 if system=='quartz' else 5
        split_sources[system]={}
        for split in ['train','val','test']:
            path=DATA/system/'transitions'/f'{h}fs'/f'{split}.h5'
            with h5py.File(path) as f:
                split_sources[system][split]=[p.decode() if isinstance(p,bytes) else str(p)
                                               for p in f['source_paths'][:]]
        for a,b in [('train','val'),('train','test'),('val','test')]:
            if set(split_sources[system][a])&set(split_sources[system][b]):
                raise ValueError(f'Source leakage in {system} {a}/{b}')
        for name, path in checkpoints(system).items():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            status = path.parent/'status.json'
            runs = [path.parent, path.parent.parent] if 'stage2' in name else [path.parent]
            rows.append(dict(system=system, model=name, checkpoint=str(path),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                selected_epoch=saved['epoch']+1, selected_updates=saved.get('optimizer_steps'),
                status=json.loads(status.read_text()) if status.exists() else None,
                discovered_prediction_results=[str(p) for p in sorted((path.parent/'evaluation').glob('*.json'))],
                source_selection=(json.loads((path.parent/'source_selection.json').read_text())
                    if (path.parent/'source_selection.json').exists() else None),
                config=saved['config'], parameters=sum(x.numel() for k,x in saved['model'].items()
                    if k not in {'rms_targets','relations.rbf.centers'}),
                existing_trajectory_summaries=[dict(path=str(p), **json.loads(p.read_text()))
                    for run in runs for p in run.glob('physics/*/trajectory.summary.json')]))
    save(REPORT/'assets.json', dict(git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
         seed=SEED, models=rows, all_checkpoints_frozen=True,
         split_sources=split_sources,
         evaluation_source_split='validation; no training or checkpoint selection'))
    return rows
