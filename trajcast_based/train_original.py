"""Launch the unmodified upstream Trainer only after explicit baseline approval.

The optional stopping hook implements the paper's 100-epoch patience, absent
from the released Trainer. It does not alter batches, gradients or updates.
The epoch scheduler adapter corrects the upstream call-count mismatch without
changing the Trainer loop or its optimization objective.
"""
import argparse
import copy
import json
import time
import yaml
from trajcast_based.acquire import RUN


def epoch_scheduler_class(base):
    """Trainer calls step once/epoch; upstream counts it as one minibatch.

    Keep the original scheduler's minibatch-unit counter/milestones and advance
    it by exactly one epoch per call. The underlying plateau scheduler receives
    the actual validation loss once each epoch, as specified in Methods.
    """
    class EpochScheduler(base):
        def step(self, val_loss):
            if self.per_epoch:
                self.step_count += self.steps_per_epoch - 1
            return super().step(val_loss)
    return EpochScheduler


def main(config, approval):
    permit=json.loads(approval.read_text())
    if not permit.get('official30_validation_accepted'):
        raise RuntimeError('Official 30 fs validation has not been accepted; no training launched.')
    import trajcast.model.training as upstream_training
    # Local process only: leave the pinned checkout and inference unchanged.
    upstream_training.CustomChainedScheduler=epoch_scheduler_class(
        upstream_training.CustomChainedScheduler)
    Trainer=upstream_training.Trainer
    cfg=yaml.safe_load(config.read_text())
    if (RUN/'original16').exists():
        raise FileExistsError('Existing original16 experiment is never overwritten')
    assert json.loads((RUN/'independent16/adaptation.json').read_text())['split_disjoint']
    trainer=Trainer(copy.deepcopy(cfg))
    trainer.dump_config_to_yaml(str(RUN/'original16_resolved.yaml'))
    update=trainer.tensorboard.update
    best, best_epoch=float('inf'),-1
    class EarlyStop(Exception): pass
    def checked_update(*args, **kw):
        nonlocal best,best_epoch
        value=update(*args,**kw)
        score=float(value); epoch=kw['epoch']
        if score<best: best,best_epoch=score,epoch
        if epoch-best_epoch>=100: raise EarlyStop()
        return value
    trainer.tensorboard.update=checked_update
    started=time.perf_counter(); reason='maximum_epochs'
    try: trainer.train()
    except EarlyStop: reason='100_epochs_without_validation_improvement'
    (RUN/'original16/training_complete.json').write_text(json.dumps(dict(
        elapsed_s=time.perf_counter()-started,stop_reason=reason,best_validation=best,best_epoch=best_epoch),indent=2))


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=RUN/'original16.yaml')
    p.add_argument('--approval',type=Path,required=True)
    a=p.parse_args();main(a.config,a.approval)
