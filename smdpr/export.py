"""Export compact real metrics/figures; never weights, trajectories or raw logs."""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = dict(updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), runs={})
    for config in sorted(args.root.rglob('config.json')):
        run = config.parent
        if 'physics' in run.relative_to(args.root).parts:
            continue
        name = str(run.relative_to(args.root))
        item = dict(server_path=str(run), history=[])
        for name in ['config', 'metadata', 'status', 'progress', 'queue_status',
                     'pipeline_status', 'validation_gate', 'formal_physics_status']:
            if (run/f'{name}.json').exists():
                item[name] = json.loads((run/f'{name}.json').read_text())
        if (run/'history.jsonl').exists():
            for row in (run/'history.jsonl').read_text().splitlines():
                try:
                    item['history'].append(json.loads(row))
                except json.JSONDecodeError:
                    # A simultaneously appended final line is not fabricated.
                    continue
        snapshot['runs'][name] = item
        # Directory structure distinguishes stages, sources and failed runs.
        for path in run.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.png', '.csv'} and any(
                    part in {'evaluation', 'physics'} for part in path.relative_to(run).parts):
                if 'spectral_cache' in path.parts:
                    continue
                dest = args.output/'results'/name/path.relative_to(run)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    (args.output/'training_snapshot.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, item in snapshot['runs'].items():
        rows = item['history']
        if not rows:
            continue
        x = [r['optimizer_steps'] for r in rows]
        final_logged = all('train_final_loss' in r for r in rows)
        axes[0].plot(x, [r['train_final_loss'] if final_logged else r['train_loss'] for r in rows],
                     '--', label=name+(' train final' if final_logged else ' train two-step total'))
        axes[0].plot(x, [r['validation']['selection_loss'] for r in rows], label=name+' val')
        if all('position_RMSE_A' in r['validation'] for r in rows):
            axes[1].plot(x, [r['validation']['position_RMSE_A'] for r in rows], label=name)
    axes[0].set(xlabel='Optimizer updates', ylabel='Normalized final endpoint loss', yscale='log')
    axes[1].set(xlabel='Optimizer updates', ylabel='Validation position RMSE (Å)', yscale='log')
    for ax in axes:
        ax.legend(fontsize=6)
        ax.grid(alpha=.2)
    fig.suptitle('SMDP-R actual training history — interim, not final results')
    fig.tight_layout()
    fig.savefig(args.output/'training_current.png', dpi=160)
    plt.close(fig)
    print(json.dumps(dict(updated_at=snapshot['updated_at'], runs=list(snapshot['runs']))))


if __name__=='__main__':
    main()
