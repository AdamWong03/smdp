"""No-overwrite execution guards; not dynamical experiments."""
import pytest
import rollout_quartz
import evaluate_quartz


def test_trajectory_existing_file_is_never_opened_for_write(tmp_path,monkeypatch):
    target=tmp_path/'nvt.h5';target.write_bytes(b'historical trajectory')
    monkeypatch.setattr('sys.argv',['rollout','--checkpoint','absent.pt','--initial','absent.h5',
        '--output',str(target),'--ensemble','nvt','--steps','937500'])
    with pytest.raises(FileExistsError):rollout_quartz.main()
    assert target.read_bytes()==b'historical trajectory'


def test_existing_test_metrics_are_not_replaced(tmp_path,monkeypatch):
    target=tmp_path/'test_metrics.json';target.write_text('{"historical": true}')
    monkeypatch.setattr('sys.argv',['evaluate','--checkpoint','absent.pt','--data-root','absent',
        '--output',str(tmp_path)])
    with pytest.raises(FileExistsError):evaluate_quartz.main()
    assert target.read_text()=='{"historical": true}'
