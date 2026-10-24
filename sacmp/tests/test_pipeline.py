"""CPU smoke tests for data-driven initialization, training and checkpoints."""
from types import SimpleNamespace

import numpy as np
import torch

from sacmp.model import SACMPConfig
from sacmp.prepare import prepare_training_anchor, make_model
from sacmp.train import batch_loss
from sacmp.rollout import load_checkpoint


class SegmentFixture(SimpleNamespace):
    def __len__(self):
        return len(self.q)


def synthetic_crystal(samples=36):
    torch.manual_seed(44)
    ref = torch.tensor([[0.,0.,0.], [1.6,0.,0.],
                        [.2,1.6,0.], [1.8,1.7,.1]])
    mass = torch.tensor([16.,28.,16.,28.])
    numbers = torch.tensor([8,14,8,14])
    offsets = torch.tensor(list(range(17))+[32,48],dtype=torch.float32)
    center = ref[None]+.015*torch.randn(samples,4,3)
    velocity = .006*torch.randn(samples,4,3)
    acceleration = .00025*torch.randn(samples,4,3)
    q = (center[:,None]+offsets[None,:,None,None]*velocity[:,None]
         +.5*offsets[None,:,None,None]**2*acceleration[:,None])
    v = velocity[:,None]+offsets[None,:,None,None]*acceleration[:,None]
    cells = torch.eye(3)[None].expand(samples,3,3).contiguous()*12.
    data = SegmentFixture(q=q, v=v, m=mass, z=numbers, cells=cells,
                           scales=dict(displacement_residual_rms=.015,
                                       velocity_delta_rms=.004))
    return data


def test_prepare_and_single_rollout_train_backward(tmp_path):
    data = synthetic_crystal()
    cfg = SACMPConfig(rank=5, hidden=24, radial_basis=6,
                      max_substep_fs=4., cutoff_a=4.5)
    anchor, report = prepare_training_anchor(data, cfg)
    assert anchor["initial_basis"].shape == (12, 5)
    assert np.isfinite(report["captured_mass_weighted_acceleration_variance"])
    assert report["training_samples"] == 36
    model = make_model(anchor, "cpu")
    model.train()
    batch = {"q":data.q[:2].clone(), "v":data.v[:2].clone(),
             "rollout_valid":torch.ones(2,dtype=torch.bool)}
    first = batch_loss(model, batch, [4, 8, 16], data.scales, False)
    assert torch.isfinite(first["loss"])
    first["loss"].backward()
    assert model.basis_raw.grad is not None
    assert torch.isfinite(model.basis_raw.grad).all()
    model.zero_grad(set_to_none=True)
    second = batch_loss(model, batch, [4, 8, 16], data.scales, True)
    assert torch.isfinite(second["loss"])
    second["loss"].backward()
    assert torch.isfinite(model.log_omega.grad).all()
    ckpt = tmp_path/"tiny.pt"
    torch.save(dict(bootstrap=anchor, model=model.state_dict(),
                    scales=data.scales, anchor_report=report), ckpt)
    restored, saved = load_checkpoint(ckpt, "cpu")
    with torch.no_grad():
        original = model.eval()(data.q[0,0],data.v[0,0],[4.])["positions"]
        after = restored(data.q[0,0],data.v[0,0],[4.])["positions"]
    torch.testing.assert_close(original, after, atol=1e-6, rtol=1e-6)
