"""Read-only CPU audit of the 80 archived legacy comparisons; no model inference.

Run with baselines/venv_vmem/bin/python. Workspace sources are immutable inputs.
Reports descriptive changes, not significance tests or causal model gains.
"""
import ast
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

W = Path('/home/hj/Alican')
OUT = Path(__file__).resolve().parent
HASHES = {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    HASHES[str(path.relative_to(W))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def funcs(path, names, scope):
    raw = path.read_bytes()
    HASHES[str(path.relative_to(W))] = hashlib.sha256(raw).hexdigest()
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def metric(raw, horizon, key):
    return raw.get('horizons', {}).get(str(horizon), raw).get(key)


def stats(x):
    return dict(median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(np.max(x)))


def reversals(v):
    # Geometric diagnostic only; tiny changes are excluded at 1% of median norm.
    norms = np.linalg.norm(v, axis=1)
    floor = max(1e-8, float(np.median(norms)) * .01)
    return int(np.sum((np.sum(v[1:] * v[:-1], axis=1) < 0) & (norms[1:] > floor) & (norms[:-1] > floor)))


def main():
    manifest = read(W / 'analysis/legacy_rejoin_reassessment_20261103/result.json')
    snapshot_path = W / 'sites/camera-correction-results/dist/snapshot.json.gz'
    snapshot_raw = snapshot_path.read_bytes()
    snapshot_sha = hashlib.sha256(snapshot_raw).hexdigest()
    published = {p['id']: p for p in json.loads(gzip.decompress(snapshot_raw))['pairs']}
    cases, cohorts, vmem = [], [], []
    for cohort in manifest['rows']:
        group = []
        for src in cohort['sources']:
            raw = read(src['path'])
            assert HASHES[str(Path(src['path']).relative_to(W))] == src['sha256']
            p = published[src['pair']]
            arms = {a: raw['arms'][p['arms'][a]['source_arm']] for a in ['baseline', 'correction']}
            row = {k: p[k] for k in ['id', 'model', 'scene', 'seed', 'cohort', 'horizon', 'estimator', 'units']}
            row.update(source=src, arm_keys={a: p['arms'][a]['source_arm'] for a in arms}, metrics={})
            for key in ['camera_mean', 'camera_endpoint', 'rotation_mean', 'rotation_endpoint', 'lpips', 'dino']:
                a, b = [metric(arms[t], p['horizon'], key) for t in ['baseline', 'correction']]
                for t, value in zip(['baseline', 'correction'], [a, b]):
                    pub = p['arms'][t]['metrics'].get(key)
                    assert (value is None and pub is None) or np.isclose(value, pub, rtol=1e-12, atol=1e-12), (p['id'], key)
                row['metrics'][key] = dict(baseline=a, correction=b, delta=None if a is None or b is None else b-a,
                                          percent=None if a is None or b is None or a == 0 else 100*(b/a-1))
            row['position_regression'] = row['metrics']['camera_mean']['delta'] is not None and row['metrics']['camera_mean']['delta'] > 0
            row['mean_improves_endpoint_worsens'] = (row['metrics']['camera_mean']['delta'] or 0) < 0 and (row['metrics']['camera_endpoint']['delta'] or 0) > 0
            # Reference motion is a descriptive context, never used to fit the controller.
            trajectory = Path(raw['batch']) / 'trajectories' / (p['scene'] + '.npz')
            HASHES[str(trajectory.relative_to(W))] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
            with np.load(trajectory, allow_pickle=False) as z:
                plan = np.linalg.inv(z['w2c_overrun' if 'w2c_overrun' in z else 'w2c'].astype(float))[:p['horizon']+1]
            steps = np.linalg.norm(np.diff(plan[:, :3, 3], axis=0), axis=1)
            row['reference_motion'] = dict(frames=len(plan), path_length=float(steps.sum()), net_translation=float(np.linalg.norm(plan[-1,:3,3]-plan[0,:3,3])), max_step=float(steps.max()))
            row['halves'] = {}
            for a, v in arms.items():
                ids = np.asarray(v['camera_frame_ids'])
                error = np.asarray(v['position_errors'], dtype=float)
                valid = v.get('position_valid', True)
                selected = ids <= p['horizon']
                measured = np.asarray(v['measured_c2w'])
                if valid:
                    assert np.allclose(np.linalg.norm(measured[selected,:3,3]-plan[ids[selected],:3,3],axis=1), error[selected], atol=1e-6)
                rot = np.degrees(Rotation.from_matrix(measured[selected,:3,:3].transpose(0,2,1) @ plan[ids[selected],:3,:3]).magnitude())
                assert np.allclose(rot, np.asarray(v['rotation_errors'])[selected], atol=2e-3)
                row['halves'][a] = {label: float(np.mean(error[mask])) if valid and np.isfinite(error[mask]).all() else None
                                   for label, mask in [('first', ids <= p['horizon']/2), ('second', (ids > p['horizon']/2) & (ids <= p['horizon']))]}
            if p['model'] == 'sana':
                directory = Path(arms['correction']['source'])
                run = read(directory/'run.json')
                calibration = read(directory/'scale_calibration.json')
                unit = calibration['observer_pose_scale']
                v = arms['correction']; indices = {f:i for i,f in enumerate(v['camera_frame_ids'])}
                offline = np.asarray(v['measured_c2w']); disagreements = []
                for ev in run['events']:
                    f = ev['observed_frame']
                    if f not in indices or f > p['horizon']: continue
                    measured = np.asarray(ev['measured']).copy(); measured[:3,3] /= unit
                    other = offline[indices[f]]
                    disagreements.append(dict(frame=f, position=float(np.linalg.norm(measured[:3,3]-other[:3,3])) if v.get('position_valid',True) else None,
                                              rotation_deg=float(np.degrees(Rotation.from_matrix(measured[:3,:3].T @ other[:3,:3]).magnitude()))))
                row['online_offline_disagreement'] = disagreements
                row['online_offline_caution'] = 'DA3 versus VIPE at exact common frames and declared scale; neither observer is ground truth.'
            if p['model'] == 'vmem':
                vmem.append(dict(id=p['id'], arms={k: v['horizons'][str(p['horizon'])] for k,v in raw['arms'].items()}))
            cases.append(row)
            group.append(row)
        valid_group = [r['metrics']['camera_mean'] for r in group if r['metrics']['camera_mean']['delta'] is not None]
        group_change = 100*(sum(r['correction'] for r in valid_group)/sum(r['baseline'] for r in valid_group)-1)
        assert np.isclose(group_change, cohort['metrics']['camera_mean']['change_percent'], atol=1e-10)
        cohorts.append(dict(model=cohort['model'], cohort=cohort['cohort'], seed=cohort['seed'], pairs=len(group),
                            valid_position=sum(r['metrics']['camera_mean']['delta'] is not None for r in group),
                            position_regressions=sum(r['position_regression'] for r in group),
                            mean_improves_endpoint_worsens=sum(r['mean_improves_endpoint_worsens'] for r in group),
                            mean_position_change_percent=group_change))

    # Replay DreamX's exact stored legacy curve and the camera encoder's measured
    # temporary anchor. Differences of saved command labels omit this anchor.
    root = W / 'analysis/plan_base_restart_20261008/new_models_loop_20261028'
    native = root.parent / 'model_readiness_20261027/repos/dreamx'
    scope = dict(np=np, Rotation=Rotation, Slerp=Slerp)
    funcs(root/'common.py', ['rejoin'], scope)
    funcs(native/'utils/trajectory_processor.py', ['Camera'], scope)
    funcs(native/'inference_ar_forcing.py', ['get_relative_pose'], scope)
    # Re-execute only the pure audit functions; do not run the old script's
    # module-level writer or overwrite its archived outputs.
    scope.update(json=json)
    funcs(W/'analysis/lingbot_failure_audit_20261030/audit.py', ['angular','audit'], scope)
    lingbot = []
    for row in cases:
        if row['model'] != 'lingbot': continue
        raw = read(row['source']['path']); pair = {}
        for arm in ['baseline','correction']:
            path = Path(raw['arms'][arm]['output_directory'])/'run.json'
            read(path)
            pair[arm] = scope['audit'](path)
        assert pair['baseline']['events'][0]['prefix_sha256'] == pair['correction']['events'][0]['prefix_sha256']
        lingbot.append(dict(id=row['id'], initial_prefix_identical=True, arms=pair))
    for path in [root/'dreamx_run.py', native/'camera_input.py', native/'pipeline/pipeline_causal_camera.py']:
        HASHES[str(path.relative_to(W))] = hashlib.sha256(path.read_bytes()).hexdigest()
    dream = []
    for row in cases:
        if row['model'] != 'dreamx':
            continue
        raw = read(row['source']['path'])
        run_path = Path(raw['arms']['correction']['output_directory'])/'run.json'
        run = read(run_path)
        plan = np.asarray(run['plan']); saved = np.asarray(run['commands']); commands = plan.copy()
        ids = np.asarray(run['native_camera_frame_ids']); keep = ids <= row['horizon']
        actual = np.diff(saved[ids,:3,3], axis=0)
        recorded = actual.copy(); nominal = np.diff(plan[ids,:3,3], axis=0)
        encoder_max_error = 0.
        for ev in run['events']:
            future = ev['future_camera_frame_ids']; prev = ev['observed_frame']; m = np.asarray(ev['measured'])
            desired = scope['rejoin'](m, commands[prev], plan[future])
            assert np.allclose(desired, saved[future], rtol=0, atol=1e-12)
            commands[future] = desired
            seq = np.concatenate([m[None], desired])
            # Actual released Camera + float32 relative-pose construction, CPU.
            rows = np.zeros((len(seq),19)); rows[:,7:] = np.linalg.inv(seq)[:,:3].reshape(-1,12)
            relative = scope['get_relative_pose']([scope['Camera'](v.tolist()) for v in rows])
            expected = np.linalg.inv(m) @ seq
            encoder_max_error = max(encoder_max_error, float(np.max(np.abs(relative[1:] - expected[1:]))))
            for j, frame in enumerate(future):
                idx = int(np.where(ids==frame)[0][0])-1
                actual[idx] = seq[j+1,:3,3]-seq[j,:3,3]
        assert np.allclose(commands, saved, rtol=0, atol=1e-12)
        actual, recorded, nominal = [v[keep[1:]] for v in [actual,recorded,nominal]]
        den = np.linalg.norm(nominal,axis=1); valid = den > max(1e-8, float(np.median(den))*.01)
        dream.append(dict(id=row['id'], run=str(run_path), command_replay_max_error=float(np.max(np.abs(commands-saved))),
                          native_float32_relative_pose_max_error=encoder_max_error,
                          stored_label_reversals=reversals(recorded), measured_anchor_step_reversals=reversals(actual),
                          target_step_reversals=reversals(nominal),
                          stored_label_speed_ratio=stats(np.linalg.norm(recorded[valid],axis=1)/den[valid]),
                          measured_anchor_speed_ratio=stats(np.linalg.norm(actual[valid],axis=1)/den[valid]),
                          scope='World-gauge intended native-step translations, using measured anchor for first step of each chunk; released float32 relative-pose construction verified. Not generated motion or archived BF16 tensor replay.'))
    result = dict(scope='80 historical comparisons in 10 heterogeneous cohorts; descriptive audit, not a pooled benchmark.',
                  published_snapshot_sha256=snapshot_sha, pairs=len(cases), valid_position=sum(r['metrics']['camera_mean']['delta'] is not None for r in cases),
                  position_regressions=sum(r['position_regression'] for r in cases),
                  mean_improves_endpoint_worsens=sum(r['mean_improves_endpoint_worsens'] for r in cases),
                  cohorts=cohorts, cases=cases, dreamx_native_transition_replay=dream, lingbot_native_replay=lingbot, vmem_four_arms=vmem,
                  input_sha256=HASHES, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['pairs','valid_position','position_regressions','mean_improves_endpoint_worsens','cohorts']},indent=2))
    for r in dream: print(r['id'], 'label/native reversals',r['stored_label_reversals'],r['measured_anchor_step_reversals'],'native ratios',r['measured_anchor_speed_ratio'])


if __name__ == '__main__':
    main()
