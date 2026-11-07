"""Publish measured training progress; never represent pending physics as results."""
import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(base,out):
    out.mkdir(parents=True,exist_ok=True);records={}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,ax in zip(['quartz30','water5'],axes):
        run=base/(name+'_20261106_r1');history=run/'history.jsonl'
        if not history.exists():continue
        rows=[]
        for line in history.read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue
        if not rows:continue
        best=min(rows,key=lambda r:r['validation']['total_loss'])
        meta=json.loads((run/'metadata.json').read_text())
        state=json.loads((run/'status.json').read_text())
        record=dict(run=str(run),metadata=meta,last=rows[-1],best=best,completed_epochs=len(rows),
                    planned_epochs=350,stage=state['state'],training_status=state)
        for split in ['val','test']:
            path=run/'evaluation'/(split+'.json')
            if path.exists():record[split+'_evaluation']=json.loads(path.read_text())
        for phase in ['validation','test']:
            path=run/'physics'/(phase+'_status.json')
            if path.exists():record[phase+'_physics_status']=json.loads(path.read_text())
        records[name]=record
        epochs=[r['epoch'] for r in rows]
        ax.semilogy(epochs,[r['train']['total_loss'] for r in rows],label='Train')
        ax.semilogy(epochs,[r['validation']['total_loss'] for r in rows],label='Validation')
        ax.set(title=name,xlabel='Epoch',ylabel='Mass-weighted segment loss');ax.legend()
    fig.tight_layout();fig.savefig(out/'training_progress.png',dpi=160);plt.close(fig)
    snapshot=dict(updated_at=datetime.datetime.now().astimezone().isoformat(),runs=records)
    (out/'progress.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    lines=['# CMIS formal experiment progress','',f"Snapshot: {snapshot['updated_at']}",'',
        'Training uses the full formal split. Training and final independent tests',
        'are complete; detailed physical failures are retained in RESULTS.md.','',
        '| System | Epoch / 350 | Optimizer steps | Best epoch | Val endpoint q RMSE (Å) | Val endpoint v RMSE (Å/fs) |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in records.items():
        v=r['best']['validation']
        lines.append(f"| {name} | {r['last']['epoch']} | {r['last']['optimizer_steps']} | {r['best']['epoch']} | {v['endpoint_position_RMSE_A']:.7f} | {v['endpoint_velocity_RMSE_A_fs']:.7f} |")
    lines+=['', 'The fixed training budget is 350 epochs (218,750 optimizer updates per',
        'system). Both models start from random initialization. The current config',
        'contains 431,827 trainable parameters. Independent short/multistep and',
        'modal assessments are complete. Quartz NVE/NVT and Water NVT fail their',
        'safety criteria; there are no complete NVT production results. See',
        'RESULTS.md and results.json for measured outcomes and missing analyses.']
    (out/'PROGRESS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.base,a.out)
