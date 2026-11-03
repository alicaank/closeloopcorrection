"""Replay archived response fitting; audit observation decisions without inference."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

W = Path('/home/hj/Alican')
R = W / 'analysis/shared_controller_pilot_v2'
O = Path(__file__).resolve().parent
sys.path.insert(0, str(R / 'frozen'))
from camera_loop.adaptive import DirectionalResponse, RecoveryConfig

hashes = {}


def track(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    hashes[str(path.relative_to(W))] = h.hexdigest()
    return path


def read(path):
    return json.loads(track(path).read_text())


def main():
    manifest = read(R / 'manifest.json')
    for name in ['adaptive.py', 'geometry.py', 'alignment.py', 'evidence.py', 'quality.py']:
        track(R / 'frozen/camera_loop' / name)
    for name in ['lingbot_pair.py', 'sana_run.py', 'observer_worker.py', 'worker.py']:
        track(R / name)
    rows = []
    pairs = []
    for case in manifest['cases']:
        key = f"{case['model']}_{case['scene']}_s{case['seed']}"
        scores = read(R / 'scores' / f'{key}.json')
        paired_runs = {}
        for arm in ['B', 'C']:
            root = R / 'runs' / key / arm
            run = read(root / 'run.json')
            paired_runs[arm] = run
            track(root / 'frames.npz')
            meta = run['controller'] if case['model'] == 'lingbot' else read(root / 'controller.json')
            events = meta['controller_events']
            assert len(events) == len(run['events'])
            issued = np.asarray(meta['native_relative_inputs'])
            frame_ids = np.asarray(run['camera_frame_ids']) if case['model'] == 'lingbot' else np.arange(len(run['plan']))
            unit_scale = 1 / run['native_translation_normalizer'] if case['model'] == 'lingbot' else 1.
            response = DirectionalResponse(RecoveryConfig(**meta['config']))
            last_index = None
            last_pose = None
            last_clipped = False
            detail = []
            max_error = 0.
            for i, (event, raw) in enumerate(zip(events, run['events'])):
                evidence = read(root / 'observer' / f'evidence_{i:03d}.json')['quality']
                assert evidence == raw['measurement']['quality']
                q = event['evidence']
                assert all(q[k] == v for k, v in evidence.items())
                assert event['accepted'] == evidence['passed']
                assert event['frame_id'] == evidence['frame_id']
                boundary = int(np.flatnonzero(frame_ids == event['frame_id'])[0])
                accepted = event['accepted']
                expected_fit = [False] * 6
                eligible = False
                if arm == 'B':
                    reason = 'identity_response_arm'
                elif not accepted:
                    reason = 'observation_rejected'
                elif last_index != boundary - q['previous_chunk_size']:
                    reason = 'no_consecutive_accepted_pair'
                elif last_clipped:
                    reason = 'previous_chunk_clipped'
                else:
                    eligible = True
                    reason = 'response_motion_gates'
                    actual = np.eye(4)
                    for command in issued[last_index + 1:boundary + 1]:
                        actual = actual @ command
                    now = np.asarray(raw['measured']).copy()
                    now[:3, 3] *= unit_scale
                    seen = np.linalg.inv(last_pose) @ now
                    duration = frame_ids[boundary] - frame_ids[last_index]
                    scale = np.maximum(np.repeat(np.asarray(meta['native_speed_limits']) * duration, 3), 1e-9)
                    expected_fit = response.update(actual, seen, scale, event['confidence'])
                    if any(expected_fit):
                        reason = 'response_samples_updated'
                assert expected_fit == event['response_update_axes'], (key, arm, i)
                gains = response.gains() if arm == 'C' else np.ones(6)
                error = float(np.max(np.abs(gains - event['response_gains'])))
                max_error = max(max_error, error)
                assert error < 1e-10, (key, arm, i, error)
                if accepted:
                    last_index = boundary
                    last_pose = np.asarray(raw['measured']).copy()
                    last_pose[:3, 3] *= unit_scale
                else:
                    last_index = None
                    assert event['confidence'] == 0
                last_clipped = bool(np.any(event['clipped']))
                detail.append(dict(frame=event['frame_id'], accepted=accepted,
                    rejection_reasons=evidence['reasons'], confidence=event['confidence'],
                    clipped=last_clipped, response_eligible=eligible,
                    response_status=reason, response_update_axes=expected_fit,
                    fitted_sample_counts=response.count.tolist(), gains=gains.tolist(),
                    nonidentity_gain=bool(np.any(np.abs(gains - 1) > 1e-12))))
            score = scores['arms'][arm]
            rows.append(dict(id=key, model=case['model'], arm=arm, observations=len(detail),
                accepted=sum(e['accepted'] for e in detail),
                rejected=sum(not e['accepted'] for e in detail),
                rejection_reasons=dict(Counter(reason for e in detail for reason in e['rejection_reasons'])),
                clipped_chunks=sum(e['clipped'] for e in detail),
                eligible_response_pairs=sum(e['response_eligible'] for e in detail),
                response_update_chunks=sum(any(e['response_update_axes']) for e in detail),
                nonidentity_gain_chunks=sum(e['nonidentity_gain'] for e in detail),
                response_status_counts=dict(Counter(e['response_status'] for e in detail)),
                response_replay_max_abs_error=max_error,
                final_gains=detail[-1]['gains'], final_sample_counts=response.count.tolist(),
                scores={k: score[k] for k in ['position_valid', 'camera_mean', 'camera_endpoint', 'rotation_mean', 'rotation_endpoint']},
                config=meta['config'], events=detail))
        cmd = 'latent_commands' if case['model'] == 'lingbot' else 'commands'
        command_error = float(np.max(np.abs(np.asarray(paired_runs['B'][cmd]) - paired_runs['C'][cmd])))
        paths = [str((R / 'runs' / key / arm / 'frames.npz').relative_to(W)) for arm in ['B', 'C']]
        exact_rgb = hashes[paths[0]] == hashes[paths[1]]
        exact_K = np.array_equal(paired_runs['B']['K'], paired_runs['C']['K'])
        exact_plan = np.array_equal(paired_runs['B']['plan'], paired_runs['C']['plan'])
        if exact_rgb:
            assert command_error == 0 and exact_K and exact_plan
            assert rows[-1]['nonidentity_gain_chunks'] == 0
        metrics = ['camera_mean', 'camera_endpoint', 'rotation_mean', 'rotation_endpoint']
        raw_delta = {k: scores['arms']['C'][k] - scores['arms']['B'][k]
                     if scores['arms']['C'][k] is not None and scores['arms']['B'][k] is not None else None
                     for k in metrics}
        pairs.append(dict(id=key, exact_rgb_archive=exact_rgb, exact_intrinsics=exact_K,
            exact_reference_plan=exact_plan, saved_command_max_abs_difference=command_error,
            original_C_minus_B=raw_delta,
            identical_input_canonical_C_minus_B={k: 0. if scores['arms']['B'][k] is not None else None for k in metrics} if exact_rgb else None,
            interpretation='Identical RGB/K/reference: repeated observer differences are not a control effect.' if exact_rgb else 'Active response policy; paired efficacy requires its existing attribution and evaluation qualifications.'))
    summary = {}
    for model in ['lingbot', 'sana']:
        for arm in ['B', 'C']:
            subset = [r for r in rows if r['model'] == model and r['arm'] == arm]
            summary[f'{model}_{arm}'] = dict(runs=len(subset),
                **{key: sum(r[key] for r in subset) for key in ['observations', 'accepted', 'rejected', 'clipped_chunks', 'eligible_response_pairs', 'response_update_chunks', 'nonidentity_gain_chunks']},
                runs_with_nonidentity_gain=sum(r['nonidentity_gain_chunks'] > 0 for r in subset))
    result = dict(scope='All eight prespecified shared-controller pilot cases, B/C arms; archived evidence and exact response-fit replay only.',
        limitations=['Does not evaluate the stronger observer on historical legacy RGB.',
            'Does not isolate observation gating from the simultaneously changed recovery controller.',
            'Replays logged full-precision issued transforms, not final quantized native camera tensors.',
            'Response fit on passive closed-loop observations is not an identified differential actuation gain.',
            'A passed observation check does not prove visual scene preservation.'],
        summary=summary, runs=rows, pairs=pairs, input_sha256=hashes,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (O / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(summary=summary, input_hashes=len(hashes),
        max_response_replay_error=max(r['response_replay_max_abs_error'] for r in rows))))


if __name__ == '__main__':
    main()
