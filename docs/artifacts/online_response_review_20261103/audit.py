"""Read-only audit of completed response-v2 runs; no model inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path('/home/hj/Alican/analysis/online_response_v2_resume_20261103')
OUT = Path(__file__).resolve().parent
inputs = {}


def bind(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    inputs[str(path)] = h.hexdigest()


def read(path):
    bind(path)
    return json.loads(path.read_text())


def video(path):
    bind(path)
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    hashes, shape = [], None
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            shape = list(rgb.shape)
            hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    finally:
        cap.release()
    assert hashes, path
    return dict(frames=len(hashes), shape=shape, frame_sha256=hashes)


result = dict(scope='scene 0238de14 seed 7; completed pilot only', models={})
for model, arms, expected in [('astronex', ['A', 'S', 'C'], 637),
                              ('worldplay', ['A', 'S'], 317)]:
    records, decoded, summaries = {}, {}, {}
    for arm in arms:
        base = ROOT / 'runs' / model / arm / '0238de14_s7'
        records[arm] = read(base.with_suffix('.json'))['online_response']
        decoded[arm] = video(base.with_suffix('.mp4'))
        assert decoded[arm]['frames'] == expected
        events = records[arm]['events']
        summaries[arm] = dict(
            boundaries=len(events), accepted=sum(e['accepted'] for e in events),
            corrections=sum(e['applied'] for e in events),
            active_boundaries=sum(any(e['active_axes']) for e in events),
            fit_reason_axis_counts=dict(Counter(axis['reason'] for e in events
                for axis in e['response'].get('axes', []))),
            rejected_observations=[dict(frame_id=e['frame_id'],
                reasons=e['evidence'].get('reasons', [])) for e in events if not e['accepted']],
        )
    comparisons = {}
    for arm in arms[1:]:
        a, b = records['A'], records[arm]
        x, y = decoded['A']['frame_sha256'], decoded[arm]['frame_sha256']
        mismatch = [i for i, (u, v) in enumerate(zip(x, y)) if u != v]
        prefix_mismatch = [u['frame_id'] for u, v in zip(a['prefix_hashes'], b['prefix_hashes']) if u != v]
        comparisons['A/' + arm] = dict(
            exact_native_commands=a['native_commands'] == b['native_commands'],
            exact_plan=a['plan'] == b['plan'],
            prefix_receipt_count=[len(a['prefix_hashes']), len(b['prefix_hashes'])],
            raw_prefix_mismatch_boundaries=prefix_mismatch,
            exported_rgb_exact=not mismatch and len(x) == len(y),
            exported_mismatch_count=len(mismatch),
            first_exported_mismatch=mismatch[0] if mismatch else None,
            note='Exported MP4 is lossy; raw-prefix receipts independently locate generation mismatch.',
        )
    result['models'][model] = dict(arms=summaries, comparisons=comparisons, decoded=decoded)

score = read(ROOT / 'scores/astronex_s7.json')
keys = ['position_valid', 'camera_mean', 'camera_endpoint', 'rotation_mean',
        'rotation_endpoint', 'dino', 'lpips', 'path_length_ratio']
result['astronex_scores'] = {a: {k: s[k] for k in keys} for a, s in score['arms'].items()}
assert result['astronex_scores']['A'] == result['astronex_scores']['C']
assert all(c['exported_rgb_exact'] and c['exact_native_commands'] and
           not c['raw_prefix_mismatch_boundaries']
           for c in result['models']['astronex']['comparisons'].values())
assert all(v['corrections'] == 0 for m in result['models'].values() for v in m['arms'].values())
bind(ROOT / 'scores/astronex_s7.jpg')
read(ROOT / 'status.json')
manifest = read(ROOT / 'manifest.json')
source_paths = [
    ROOT / 'hosts/analysis/plan_base_restart_20261008/fair_common_20261025/wp_fair.py',
    ROOT / 'frozen/camera_loop/native_response_runtime.py',
    Path('/home/hj/Alican/sota_repos/worldplay/wan/inference/pipeline_wan_w_mem_relative_rope.py'),
]
for path in source_paths:
    bind(path)
    assert inputs[str(path)] == manifest['hashes'][str(path)], path
bind(Path(__file__).resolve())
result['inputs_sha256'] = inputs
(OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({m: d['comparisons'] for m, d in result['models'].items()}, indent=2))
