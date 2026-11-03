import pytest
import torch

from train_relational import dataset_settings, make_scheduler


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


def test_dataset_settings_support_water_without_changing_quartz_defaults():
    assert dataset_settings({}) == (
        'quartz30', {8: 0, 14: 1}, {'train': 5000, 'val': 1250, 'test': 1000})
    config = {'dataset_name': 'water5', 'atom_type_mapper': {'1': 0, '8': 1},
              'split_counts': {'train': 5000, 'val': 1250, 'test': 1000}}
    assert dataset_settings(config) == (
        'water5', {1: 0, 8: 1}, {'train': 5000, 'val': 1250, 'test': 1000})
