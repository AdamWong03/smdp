"""Prepare the original 16 fs baseline config; never launches training."""
import json
import yaml
from trajcast_based.acquire import RUN


def main():
    model=yaml.safe_load((RUN/'official/models/quartz/config_e3nn.yaml').read_text())['model']
    root=RUN/'independent16'
    def data(split):
        return dict(root=str(root),name=f'quartz16_{split}',cutoff_radius=4.5,files=[],
                    atom_type_mapper={8:0,14:1},time_reversibility=False)
    cfg=dict(model=model,data=data('train'),training=dict(
        seed=20261027,device='cuda',model_type='EfficientTrajCastModel',restart_latest=False,
        target_field='target',reference_fields=['displacements','update_velocities'],batch_size=2,
        num_epochs=1500,criterion='mse',optimizer='adam',optimizer_settings=dict(lr=.01,amsgrad=True),
        scheduler=['ReduceLROnPlateau'],scheduler_settings={'ReduceLROnPlateau':dict(factor=.8,patience=20)},
        chained_scheduler_hp=dict(milestones=[10000000],per_epoch=True,monitor_lr_scheduler=False),
        checkpoint_settings=dict(root=str(RUN/'original16/checkpoints'),interval=10,keep_latest=False),
        tensorboard_settings=dict(log_dir=str(RUN/'original16/tensorboard'),loss=True,lr=True,
            loss_validation={'data':data('val'),'batch_size':2})))
    (RUN/'original16.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (RUN/'experiment_versions.json').write_text(json.dumps({
        'Original TrajCast':{'status':'configuration_prepared_not_trained','horizon_fs':16},
        'TrajCast + Modes':{'status':'not_implemented_pending_reliable_original16'},
        'TrajCast + Modes + Segment':{'status':'not_implemented_pending_reliable_original16'}},indent=2)+'\n')


if __name__=='__main__': main()
