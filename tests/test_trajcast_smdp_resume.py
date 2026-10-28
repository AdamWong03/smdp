"""CPU checks for resuming the original optimizer and scheduler protocol."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from trajcast_smdp.train import resume_handler,upstream


class ResumeTest(unittest.TestCase):
    def test_batch_change_keeps_epoch_scheduler_and_adam(self):
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,amsgrad=True)
        def scheduler(opt,steps):
            return upstream.CustomChainedScheduler(True,
                [torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.8,patience=1)],
                [10000000],20,steps,False,['ReduceLROnPlateau'])
        original=scheduler(optimizer,2500);history=[]
        for epoch in range(4):
            optimizer.zero_grad();model(torch.ones(1,2)).square().mean().backward();optimizer.step()
            row=dict(stage='segment',epoch=epoch,selection_score=1.,lr=optimizer.param_groups[0]['lr'])
            history.append(row)
            payload=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                config={'training':{'optimizer_settings':{'lr':.001}}},record=row,
                torch_rng=torch.get_rng_state(),cuda_rng=[])
            original.step(1.)
        other=torch.nn.Linear(2,1);opt=torch.optim.Adam(other.parameters(),lr=.001,amsgrad=True)
        changed=scheduler(opt,625)
        with patch('torch.cuda.set_rng_state_all'):
            resume_handler(object,payload,history)().load_latest(SimpleNamespace(model=other,optimizer=opt,lr_scheduler=changed))
        self.assertEqual(changed.steps_per_epoch,625)
        self.assertEqual(changed.step_count,4*625)
        self.assertEqual(changed.schedulers[0].state_dict(),original.schedulers[0].state_dict())
        self.assertEqual(opt.param_groups[0]['lr'],optimizer.param_groups[0]['lr'])
        for p,q in zip(model.parameters(),other.parameters()):
            for key,value in optimizer.state[p].items():
                torch.testing.assert_close(value,opt.state[q][key],rtol=0,atol=0)

    def test_pre_scheduler_checkpoint_restores_next_epoch(self):
        torch.set_default_dtype(torch.float64)
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,amsgrad=True)
        def scheduler(opt):
            return upstream.CustomChainedScheduler(True,
                [torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.8,patience=1)],
                [10000000],20,2500,False,['ReduceLROnPlateau'])
        sched=scheduler(optimizer);history=[]
        for epoch,score in enumerate([1.,1.,1.,.9,.9,.9]):
            optimizer.zero_grad();model(torch.ones(1,2)).square().mean().backward();optimizer.step()
            row=dict(stage='segment',epoch=epoch,selection_score=score,lr=optimizer.param_groups[0]['lr'])
            history.append(row)
            payload=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(optimizer.state_dict()),
                config={'training':{'optimizer_settings':{'lr':.001}}},record=row,
                torch_rng=torch.get_rng_state(),cuda_rng=[])
            sched.step(score)
        expected_rng=torch.rand(8)
        other=torch.nn.Linear(2,1);opt=torch.optim.Adam(other.parameters(),lr=.001,amsgrad=True)
        other_sched=scheduler(opt)
        state=SimpleNamespace(model=other,optimizer=opt,lr_scheduler=other_sched)
        with patch('torch.cuda.set_rng_state_all'):
            epoch,best=resume_handler(object,payload,history)().load_latest(state)
        self.assertEqual(epoch,5);self.assertEqual(best,.9)
        self.assertEqual(other_sched.save(),sched.save())
        self.assertEqual(opt.param_groups[0]['lr'],optimizer.param_groups[0]['lr'])
        torch.testing.assert_close(torch.rand(8),expected_rng,rtol=0,atol=0)
        for p,q in zip(model.parameters(),other.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for key,value in optimizer.state[p].items():
                torch.testing.assert_close(value,opt.state[q][key],rtol=0,atol=0)


if __name__=='__main__': unittest.main()
