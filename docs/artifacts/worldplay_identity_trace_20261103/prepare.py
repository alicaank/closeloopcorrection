"""Freeze the full-horizon identity diagnostic without editing the failed pilot."""
import hashlib
import json
from pathlib import Path
import shutil
import time

R = Path(__file__).resolve().parent
W = R.parents[1]
OLD = R.parent / 'online_response_v2_resume_20261103'
assert not (R / 'manifest.json').exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent = json.loads((OLD / 'manifest.json').read_text())
for p, h in parent['hashes'].items():
    assert sha(p) == h, p
for name in ['frozen', 'hosts', 'batch']:
    shutil.copytree(OLD / name, R / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
for name in ['arms.json']:
    shutil.copy2(OLD / name, R / name)
for root in ['frozen', 'hosts']:
    for p in (R / root).rglob('*.py'):
        s = p.read_text()
        if str(OLD) in s:
            p.write_text(s.replace(str(OLD), str(R)))
shutil.copy2(W / 'camera-closed-loop/src/camera_loop/native_trace.py', R / 'frozen/camera_loop/native_trace.py')
(R / 'tests').mkdir()
shutil.copy2(W / 'camera-closed-loop/tests/test_native_trace.py', R / 'tests/test_native_trace.py')
(R / 'logs').mkdir()
group = next(g for g in parent['groups'] if g['model'] == 'worldplay' and g['seed'] == 7)
jobs = json.loads(json.dumps([j for j in group['jobs'] if j['arm'] in ['A', 'S']]).replace(str(OLD), str(R)))
for job in jobs:
    job['argv'][1] = str(R / 'trace_launcher.py')
assert len(jobs) == 2
hashes = {}
for p, h in parent['hashes'].items():
    q = Path(p.replace(str(OLD), str(R)))
    hashes[str(q if q.exists() else Path(p))] = sha(q if q.exists() else p)
for root in [R / 'frozen', R / 'hosts', R / 'tests']:
    for p in root.rglob('*.py'):
        hashes[str(p)] = sha(p)
for p in R.glob('*.py'):
    hashes[str(p)] = sha(p)
hashes[str(OLD / 'manifest.json')] = sha(OLD / 'manifest.json')
manifest = dict(protocol='worldplay_identity_trace_v1', created=time.time(),
                parent_manifest_sha256=sha(OLD / 'manifest.json'),
                scene='0238de14', seed=7, chunks=20, rgb_frames=317,
                jobs=jobs, hashes=hashes, weights=parent['weights'],
                scope='Diagnostic A/S only. Same model, seed, resolution, horizon, commands and observer. Read-only trace changes timing; no correction.',
                limits=dict(launch_free_gib=8, runtime_free_gib=4, host_free_gib=2, timeout_hours=8))
(R / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(jobs=len(jobs), bound_sources=len(hashes), manifest_sha256=sha(R / 'manifest.json'))))
