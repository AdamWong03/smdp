import json
import sys

from smdpr.export import main


def test_export_distinguishes_stages_and_omits_incomplete_figures(tmp_path, monkeypatch):
    root, output = tmp_path/'runs', tmp_path/'report'
    run = root/'quartz'
    stage2 = run/'two_step'
    stage2.mkdir(parents=True)
    for directory in [run, stage2]:
        (directory/'config.json').write_text('{}')
    row = dict(optimizer_steps=625, train_final_loss=.002,
               validation=dict(selection_loss=.003, position_RMSE_A=.01))
    (run/'history.jsonl').write_text(json.dumps(row)+'\n')
    row = dict(optimizer_steps=625, train_loss=.002,
               validation=dict(selection_loss=.003))
    (stage2/'history.jsonl').write_text(json.dumps(row)+'\n')
    analysis = stage2/'physics/nvt/analysis'
    analysis.mkdir(parents=True)
    (analysis/'metrics.json').write_text('{"physical_analysis_completed": false}')
    (analysis/'incomplete.csv').write_text('partial')
    monkeypatch.setattr(sys, 'argv', ['export', '--root', str(root), '--output', str(output)])
    main()
    snapshot = json.loads((output/'training_snapshot.json').read_text())
    assert set(snapshot['runs']) == {'quartz', 'quartz/two_step'}
    assert (output/'results/quartz/two_step/physics/nvt/analysis/metrics.json').exists()
    assert not list(output.rglob('incomplete.csv'))
