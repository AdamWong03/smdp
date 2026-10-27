"""Check epoch scheduling against PyTorch directly; no model training."""
import json
import torch
from trajcast.model.utils import CustomChainedScheduler
from trajcast_based.acquire import RUN
from trajcast_based.train_original import epoch_scheduler_class


def main():
    traces={}
    for name,kind in [('upstream',CustomChainedScheduler),
                      ('epoch_corrected',epoch_scheduler_class(CustomChainedScheduler)),
                      ('pytorch_reference',None)]:
        parameter=torch.nn.Parameter(torch.zeros(()))
        optimizer=torch.optim.Adam([parameter],lr=.01)
        plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.8,patience=20)
        scheduler=plateau if kind is None else kind(
            True,[plateau],[10000000],1500,2500,False,['ReduceLROnPlateau'])
        trace=[]
        for _ in range(50):
            scheduler.step(1.)
            trace.append(optimizer.param_groups[0]['lr'])
        traces[name]=trace
    assert traces['epoch_corrected']==traces['pytorch_reference']
    assert traces['upstream']==[.01]*50
    report=dict(passed=True,epochs=50,batches_per_epoch=2500,
        constant_validation_loss=1.,lr_traces=traces,
        reason='Upstream Trainer calls step once per epoch; upstream scheduler counts minibatches.',
        changes='Local training entrypoint counter adapter only; official model, loss and Forecast unchanged.')
    (RUN/'training_scheduler_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='lr_traces'},indent=2))


if __name__=='__main__': main()
