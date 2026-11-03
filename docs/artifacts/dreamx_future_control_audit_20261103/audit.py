"""CPU replay of completed moving-scene pilot; no model/gain fitting."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/hj/Alican')
BASE = ROOT / 'analysis/dreamx_future_command_20261103'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'source'))
from camera_loop.fractional_recovery import issue_fractional_future

hashes = {}


def track(path):
    path = Path(path)
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(track(path).read_text())


def main():
    manifest = read(BASE / 'manifest.json')
    for name in ('fractional_recovery.py', 'geometry.py', 'memory_rate.py'):
        path = track(BASE / 'source/camera_loop' / name)
        assert hashes[str(path.relative_to(ROOT))] == manifest['sources'][str(path)]
    path = track(BASE / 'future_hook.py')
    assert hashes[str(path.relative_to(ROOT))] == manifest['sources'][str(path)]
    rows = []
    for seed in (7, 107):
        key = f'dreamx_0238de14_s{seed}'
        run = BASE / 'runs' / key / 'F'
        score = read(BASE / 'scores' / key / 'score.json')
        assert score['comparison_valid']
        for kind in ('identity', 'attribution'):
            assert read(BASE / 'gates' / f'{key}_{kind}.json')['passed']
        with np.load(track(run / 'issued_commands.npz')) as z:
            plan, saved = z['plan'], z['commands']
        with np.load(track(BASE / 'scores' / key / 'F.npz')) as z:
            offline, offline_ids = z['c2w'], z['frame_ids']
        np.testing.assert_array_equal(offline_ids, np.arange(69))
        issued = plan.copy()
        events = read(run / 'future_events.json')
        boundaries = []
        for event in events:
            r = event['receipt']
            assert event['quality']['passed']
            ids = np.asarray(event['query_frames'])
            measured = np.asarray(event['observation'])
            b, anchor = r['available_frame'], r['memory_anchor_frame']
            end = r['native_camera_frames'][-1] + 1
            result = issue_fractional_future(
                plan, issued, measured, ids, next_end=end,
                available_frame=b, feedback_index=event['feedback_index'],
                nominal_boundaries=1, history_labels='commanded', native_stride=4,
                recovery_fraction=.5, speed_multiplier=2., memory_anchor_frame=anchor)
            for name in ('initial_residual', 'requested_recovery',
                         'realized_command_recovery', 'unserved_recovery'):
                if name in r:
                    np.testing.assert_allclose(result.receipt[name], r[name], rtol=0, atol=1e-14)
            issued = result.commands
            online_pose = measured[np.flatnonzero(ids == b)[0]]
            position_gap = np.linalg.norm(online_pose[:3, 3] - offline[b, :3, 3])
            rotation_gap = np.rad2deg(Rotation.from_matrix(
                online_pose[:3, :3].T @ offline[b, :3, :3]).magnitude())
            online_residual = plan[b, :3, 3] - online_pose[:3, 3]
            offline_residual = plan[b, :3, 3] - offline[b, :3, 3]
            denominator = np.linalg.norm(online_residual) * np.linalg.norm(offline_residual)
            steps = r.get('steps', [])
            boundaries.append(dict(
                available_frame=b, memory_anchor_frame=anchor,
                reason=r['reason'], native_input_changed=r['native_input_changed'],
                realized_command_fraction=r.get('realized_fraction'),
                requested_recovery_norm=float(np.linalg.norm(r['requested_recovery'])) if steps else None,
                unserved_recovery_norm=float(np.linalg.norm(r['unserved_recovery'])) if steps else None,
                speed_clipped_frames=sum(s['speed_clipped'] for s in steps),
                slew_clipped_frames=sum(s['slew_clipped'] for s in steps),
                proposed_max_speed_over_plan=max(s['speed'] for s in steps)/r['planned_speed_max'] if steps else None,
                online_offline_position_gap=float(position_gap),
                online_offline_rotation_gap_deg=float(rotation_gap),
                online_residual_norm=float(np.linalg.norm(online_residual)),
                offline_residual_norm=float(np.linalg.norm(offline_residual)),
                residual_direction_cosine=float(online_residual @ offline_residual / denominator) if denominator else None,
                passive_execution=event['passive_execution']))
        np.testing.assert_array_equal(issued, saved)
        rows.append(dict(key=key, command_replay_exact=True, boundaries=boundaries))
    result = dict(
        scope='Two completed moving-scene DreamX F runs; replay of frozen float64 issuance, not native bf16 encoder replay.',
        definition='Command recovery is the requested geometric endpoint change before bridging, not observed generator recovery. Speed ratios cover per-RGB-frame proposals, not physical speed.',
        observer_comparison='Online DA3 and offline VIPE poses at the same available RGB frame, each registered by its existing initial-anchor procedure. No additional trajectory alignment or scale fit. Neither observer is ground truth; causal windows and full-video fits differ.',
        cases=rows, input_sha256=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for row in rows:
        print(row['key'], 'exact replay:', row['command_replay_exact'])
        for b in row['boundaries']:
            print({k:v for k,v in b.items() if k != 'passive_execution'})


if __name__ == '__main__':
    main()
