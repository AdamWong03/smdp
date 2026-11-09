"""Materialize immutable configurations for the accepted full-data experiment."""
import json

import torch

from conditional_transition.common import ROOT,ARMS,checkpoint
from smdpr.train import write_json


def main():
    for system in ['quartz','water']:
        cp=checkpoint(system)
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        assert saved['config']['model']['block_type']=='geometry_gated'
        for arm in ARMS:
            cfg=dict(system=system,arm=arm,initial_checkpoint=str(cp),
                seed=saved['config']['seed']+200,total_updates=25000,
                updates_per_round=12500,microbatch=8,transition_targets_per_update=16,
                optimizer='AdamW',amsgrad=True,weight_decay=0.,precision='FP32',tf32=False,
                initial_lr=3e-5,minimum_lr=1e-5,gradient_clip=10,proposal_weight=.02,
                original_train_samples=5000,validation_samples=1250,
                rounds=2,additional_labels_per_round=2000 if arm in ['autonomous','extra_reference'] else 0,
                extra_fraction=.3 if arm in ['autonomous','extra_reference'] else 0.,
                collector_starts=125,collector_max_depth=256,
                collector_depth_bins=[[1,4],[5,16],[17,64],[65,256]],
                collector_samples_per_bin=4,
                selection='fixed final update 25000; validation monitoring only',
                inference_reference_calls=False,test_accessed=False)
            run=ROOT/system/arm;run.mkdir(parents=True,exist_ok=True)
            dest=run/'config.json'
            if dest.exists():
                assert json.loads(dest.read_text())==cfg,'Refuse to overwrite changed configuration'
            else:write_json(dest,cfg)
            print(run,flush=True)


if __name__=='__main__':main()
