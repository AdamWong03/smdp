import pytest
import torch

from train_relational import make_scheduler


def test_warmup_cosine_scheduler_reaches_frozen_endpoints():
    parameter = torch.nn.Parameter(torch.zeros(()))
    config = {
        "learning_rate": 3.0e-4,
        "total_optimizer_steps": 250_000,
        "scheduler": {
            "type": "warmup_cosine",
            "initial_lr": 1.0e-5,
            "warmup_steps": 5_000,
            "minimum_lr": 1.0e-5,
        },
    }
    optimizer = torch.optim.AdamW([parameter], lr=config["learning_rate"])
    scheduler, scheduler_type = make_scheduler(optimizer, config)

    assert scheduler_type == "warmup_cosine"
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-5)
    scale = scheduler.lr_lambdas[0]
    assert config["learning_rate"] * scale(5_000) == pytest.approx(3.0e-4)
    assert config["learning_rate"] * scale(250_000) == pytest.approx(1.0e-5)
