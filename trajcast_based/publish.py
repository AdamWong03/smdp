"""Publish small verified artifacts only. Never includes source trajectories/weights."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from trajcast_based.acquire import RUN


def main(push):
    repo=Path(__file__).resolve().parents[1]
    out=repo/'reports/trajcast_foundation_20261027'
    out.mkdir(parents=True,exist_ok=True)
    names=['upstream_manifest.json','interface_checks.json','environment.lock.txt',
           'official_test_metrics.json','original16.yaml','experiment_versions.json',
           'training_scheduler_check.json',
           'independent16/adaptation.json','independent16/cache_verification.json']
    tags=sorted(set(['official30_100ps','official30_15ns']) |
                {p.name for p in RUN.glob('official30_*') if p.is_dir()})
    for tag in tags:
        names += [f'{tag}/{name}' for name in ['run.json','physics.metrics.json','physics.vdos.csv',
                      'physics.rdf.csv','energy_conventions.json','interruption.json']]
    for name in names:
        src=RUN/name
        if src.exists():
            assert src.stat().st_size<2_000_000
            dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    status={'server_artifacts':str(RUN),'original16_trained':False,
            'adapter_implemented':False,'segment_implemented':False}
    for name in ['official_test_metrics.json','independent16/adaptation.json']:
        status[name]=(RUN/name).exists()
    for tag in tags:
        status[tag]=json.loads((RUN/tag/'run.json').read_text()) if (RUN/tag/'run.json').exists() else {'status':'not_started'}
        interruption=RUN/tag/'interruption.json'
        if interruption.exists():
            status[tag]['execution_status']='interrupted'
            status[tag]['interruption']=json.loads(interruption.read_text())
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    lines=['# TrajCast foundation — observed results','',
        'Pinned upstream, environment, protocol and limits: [audit](../../trajcast_based/AUDIT.md).',
        f'Large artifacts remain at `{RUN}`. Historical experiments are unchanged.','',
        'No adapter/segment training has been launched. Original16 configuration is not a trained baseline.','']
    if (RUN/'official_test_metrics.json').exists():
        m=json.loads((RUN/'official_test_metrics.json').read_text())
        lines+=['## Official 30 fs checkpoint / official 1000-sample test','',
            '| Metric | Measured | Paper Table S1 |','|---|---:|---:|',
            f"| Displacement MAE / Å | {m['displacement_MAE_A']:.9g} | 0.0014124 |",
            f"| Velocity MAE / Å/fs | {m['velocity_MAE_A_fs']:.9g} | 0.0001137 |",'']
    cache=RUN/'independent16/cache_verification.json'
    if cache.exists():
        verified=json.loads(cache.read_text())
        lines += ['## Independent 16 fs dataset — official loader roundtrip','',
            '| Split | Samples | Maximum absolute field difference |',
            '|---|---:|---:|']
        for split in ['train','val','test']:
            m=verified[split]
            lines += [f"| {split} | {m['samples']} | {m['roundtrip_max_abs_error']} |"]
        lines += ['', 'No cross-graph neighbor edges. Normalization uses training samples only.', '']
    for tag in tags:
        lines += [f'## {tag}','',f"Trajectory completed: {status[tag].get('trajectory_completed',False)}; physics completed: {status[tag].get('physical_analysis_completed',False)}.",'']
        if 'execution_status' in status[tag]:
            lines += [f"Execution status at publication: {status[tag]['execution_status']}.", '']
        if 'interruption' in status[tag]:
            lines += [status[tag]['interruption']['reason'], '']
        p=RUN/tag/'physics.metrics.json'
        if p.exists():
            m=json.loads(p.read_text())
            lines += [f"Temperature: {m['mean_temperature_k']:.3f} K.",
                f"VDOS overlap to published MD: {m['vdos_overlap_to_published_md']}.",
                f"Published TrajCast overlap (source CSV): {m['published_trajcast_overlap_to_published_md']}.",
                'See energy_conventions.json for raw/DSF-corrected means and absolute differences; centered similarity alone is insufficient.','']
    lines += ['## Outstanding physical-protocol limitation', '',
              'The 100 ps reconstructed Carré-convention potential-energy mean is about '
              '7.63 eV below the published reference mean, even after removing the known '
              'DSF self-energy constant. This difference is unresolved; it must not be '
              'dismissed as an arbitrary zero-point shift. The original Quartz LAMMPS '
              'input deck is not present in the pinned upstream release.', '',
              '100 ps is a preliminary physical validation, not 15 ns reproduction.',
              'GPU sharing and host load affect wall time; no uncontended speedup claim is made.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    if push:
        subprocess.run(['git','add',str(out.relative_to(repo))],cwd=repo,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=repo).returncode:
            subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost',
                'commit','-m','Report measured official TrajCast reproduction and adapter status'],cwd=repo,check=True)
        subprocess.run(['git','-c','http.version=HTTP/1.1','push','--ipv4','-u','origin','HEAD'],cwd=repo,check=True,timeout=120)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--push',action='store_true');main(p.parse_args().push)
