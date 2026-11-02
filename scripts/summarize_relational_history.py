"""Summarize existing relational runs without reading test data or checkpoints."""
import argparse
import json
import math
from pathlib import Path


def load_json(path):
    return json.loads(path.read_text())


def summarize(name, run):
    records = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    metadata = load_json(run/'metadata.json')
    steps_per_epoch = metadata.get(
        'optimizer_steps_per_epoch',
        math.ceil(metadata['train_samples'] / metadata['config']['batch_size']))
    best = min(records, key=lambda item: item['validation']['selection_loss'])
    final = records[-1]
    steps = lambda item: item.get('optimizer_steps', item['epoch'] * steps_per_epoch)

    def point(item):
        validation = item['validation']
        return dict(
            epoch=item['epoch'], optimizer_steps=steps(item), lr=item['lr'],
            train_final_loss=item['train_final_loss'],
            train_proposal_loss=item['train_proposal_loss'],
            validation_final_loss=validation['selection_loss'],
            validation_proposal_loss=validation['proposal_loss'],
            validation_position_RMSE_A=validation['position_RMSE_A'],
            validation_velocity_RMSE_A_fs=validation['velocity_RMSE_A_fs'])

    return dict(
        name=name, epochs=len(records), parameters=metadata['parameters'],
        batch_size=metadata['config']['batch_size'], precision=metadata['config']['precision'],
        steps_per_epoch=steps_per_epoch, best=point(best), final=point(final),
        after_best=dict(
            extra_optimizer_steps=steps(final)-steps(best),
            train_final_loss_change=final['train_final_loss']-best['train_final_loss'],
            validation_final_loss_change=(final['validation']['selection_loss']-
                                          best['validation']['selection_loss']),
            train_proposal_loss_change=(final['train_proposal_loss']-
                                        best['train_proposal_loss']),
            validation_proposal_loss_change=(final['validation']['proposal_loss']-
                                             best['validation']['proposal_loss'])),
        missing_history_fields=[
            key for key in ['train_normalized_position_MSE', 'train_normalized_velocity_MSE']
            if key not in records[0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True,
                        help='NAME=/absolute/run/directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for value in args.run:
        name, path = value.split('=', 1)
        results.append(summarize(name, Path(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'runs': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
