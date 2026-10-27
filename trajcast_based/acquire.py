"""Acquire and verify upstream assets without touching historical experiments."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/data1/wyd/research/foundation_mlip_adaptation')
RUN = ROOT / 'experiments/trajcast_based/quartz_20261027'
UPSTREAM = ROOT / 'upstream/trajcast-c52f107'
COMMIT = 'c52f107a02176671b21a420b2d4c67e0327b04f7'
REVISIONS = {'models': '621f5933a1020c68ab7f9bc5071b1362499bbfa7',
             'datasets': '71cdd22807033010bd0d7a852b84520763822e44'}


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl', '-4', '-fL', '--retry', '3', '--connect-timeout', '15',
                    '--max-time', '600', '-o', str(path), url], check=True)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    assert subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip()
    manifest = {'code_commit': COMMIT, 'code_license': 'Apache-2.0', 'assets': []}
    for kind, revision in REVISIONS.items():
        repo = f'ibm-research/trajcast.{kind}-arxiv2025'
        api = RUN / f'{kind}_metadata.json'
        download(f'https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true', api)
        meta = json.loads(api.read_text())
        assert meta['sha'] == revision and meta['cardData']['license'] == 'apache-2.0'
        for item in meta['siblings']:
            name = item['rfilename']
            if not (name.startswith('quartz/') or name in ['README.md', 'LICENSE']):
                continue
            target = RUN / 'official' / kind / name
            expected = item.get('lfs', {}).get('sha256')
            historical = ROOT / ('models/trajcast_quartz/official' if kind == 'models' else 'data/trajcast_quartz/official') / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reuse bytes ONLY after matching the upstream immutable artifact hash.
            if expected and historical.exists() and sha(historical) == expected and not target.exists():
                shutil.copy2(historical, target)
            if not target.exists():
                prefix = '' if kind == 'models' else 'datasets/'
                download(f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}', target)
            digest = sha(target)
            if expected:
                assert digest == expected, str(target)
            else:
                raw = target.read_bytes()
                git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
                assert git_blob == item['blobId'], str(target)
            manifest['assets'].append(dict(repository=repo, revision=revision, file=name,
                license=meta['cardData']['license'], bytes=target.stat().st_size, sha256=digest, local_path=str(target)))
    shutil.copy2(UPSTREAM / 'LICENSE', RUN / 'UPSTREAM_LICENSE')
    (RUN / 'upstream_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
