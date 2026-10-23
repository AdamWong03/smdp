"""Common, frozen validation protocol for R3-A/R3-B (never uses test data)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from evaluate_quartz import load_model
from periodic import PeriodicGraphBuilder
from reference_data import QuartzSegments
from train_quartz import batch_to, loss_for
from training_objectives import SegmentStats


def evaluate_checkpoint(checkpoint: Path, dataset, device, batch_size):
    model, config = load_model(checkpoint, device)
    builder = PeriodicGraphBuilder(model.config.cutoff_angstrom)
    valid_indices = torch.where(dataset.valid)[0].tolist()
    loader = DataLoader(Subset(dataset, valid_indices), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    stats = SegmentStats(**config['stats'])
    weight = float(config.get('r3_composition_weight', 0.0))
    samples = int(config.get('r3_composition_samples', 2))
    times = torch.tensor([0., 1., 4., 8., 12., 15., 16.], device=device)
    totals, n_total = {}, 0
    model.eval()
    with torch.inference_mode(False), torch.no_grad():
        for batch in loader:
            batch = batch_to(batch, device)
            loss, _ = loss_for(model, batch, times, stats, builder,
                               rollout_steps=16, composition_weight=weight,
                               composition_samples=samples)
            n = len(batch['q'])
            for key, value in loss.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    totals[key] = totals.get(key, 0.0) + float(value) * n
            n_total += n
    if not n_total:
        raise RuntimeError('No valid independent validation windows')
    return dict(checkpoint=str(checkpoint), model_config=config['model'],
                validation_samples=n_total, metrics={k:v/n_total for k,v in totals.items()},
                query_times_fs=times.tolist(), comparison_rollout_steps=16,
                composition_weight=weight, checkpoint_selection=config.get('selection'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data=QuartzSegments(args.data_root,'val',args.output.parent/'val_cache')
    rows=[evaluate_checkpoint(ckpt,data,args.device,args.batch_size)
          for ckpt in args.checkpoints]
    args.output.write_text(json.dumps(rows,indent=2))
    for x in rows: print(json.dumps(x),flush=True)


if __name__ == '__main__':
    main()
