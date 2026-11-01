"""Freeze the segment-best checkpoint for a separate evaluation-only run."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import torch
import yaml
from trajcast_smdp.data import RUN

OUT=RUN.parent/'quartz30_segment_physics_20261101'

def main():
    source=RUN/'segment/best.pt'
    payload=torch.load(source,map_location='cpu',weights_only=False)
    assert payload['record']['stage']=='segment'
    cfg=yaml.safe_load((RUN/'config.yaml').read_text())
    assert cfg['model']==payload['config']['model']
    OUT.mkdir(exist_ok=False)
    shutil.copy2(source,OUT/'best.pt')
    digest=hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest()
    (OUT/'cache').symlink_to(RUN/'cache',target_is_directory=True)
    cfg['smdp'].update(run=str(OUT),nve_ps=100.,nvt_thermalization_ps=1500.,nvt_production_ps=1000.)
    (OUT/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    manifest=dict(source_checkpoint=str(source),checkpoint_sha256=digest,
        selected_validation_record=payload['record'],training_performed=False,
        service='smdp-segment-physics-20261101-gpu2.service',
        source_training_run=str(RUN),reference_root=cfg['smdp']['reference_root'],
        nve_replicas=2,nve_ps=100.,nvt_thermalization_ps=1500.,nvt_production_ps=1000.,
        duration_reason='Storage-limited first physical evaluation; not a 15 ns reproduction',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (OUT/'evaluation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
