"""CPU audit of saved VMem four-arm runs; no generation or inferred retrieval.

Run with /home/hj/Alican/baselines/venv_vmem/bin/python vmem_replay.py.
The optional output argument defaults to vmem_replay.json beside this script.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

W = Path('/home/hj/Alican')
A = W / 'analysis/plan_base_restart_20261008'
ROOT = A / 'continuous_eval_20261029'
HASHES = {}


def track(path):
    path = Path(path)
    HASHES[str(path.relative_to(W))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(track(path).read_text())


def stats(values):
    values = np.asarray(values, float)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), max=float(values.max()))


def main():
    source = track(A / 'new_models_loop_20261028/common.py')
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rejoin')
    scope = dict(np=np, Rotation=Rotation, Slerp=Slerp)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    rejoin = scope['rejoin']
    for p in [A/'long_domain_20261028/vmem_run.py',
              A/'long_domain_20261028/score_long_vggt_legacy.py',
              A/'fair_common_20261025/fair_common.py',
              W/'baselines/vmem/modeling/pipeline.py',
              W/'baselines/vmem/configs/inference/inference.yaml']:
        track(p)
    queue = read(ROOT/'queue_status.json')
    rows = []
    for scorepath in sorted((ROOT/'scores').glob('vmem_*.json')):
        score = read(scorepath)
        scene = score['scene']
        with np.load(track(Path(score['batch'])/'trajectories'/f'{scene}.npz')) as z:
            reference = np.linalg.inv(z['w2c'].astype(float))
        first = None
        for arm, scored in score['arms'].items():
            raw = read(ROOT/'raw'/f'vmem_{scene}'/arm/f'{scene}_s7.json')
            plan, command, measured = [np.asarray(raw[k], float) for k in ['planned', 'commanded', 'measured']]
            assert plan.shape == command.shape == measured.shape == (41, 4, 4)
            assert np.array_equal(plan, reference)
            assert np.isfinite(measured).all()
            settings = raw['settings']
            assert settings['scale_mode'] == 'window' and settings['nms'] == 'cfg'
            assert settings['first_step'] == 4
            job = queue['jobs'][f'vmem_{scene}_{arm}']
            # One queue entry was left 'running'; use the saved completion line,
            # metadata and complete scores, while preserving that provenance gap.
            log = track(Path(job['log']))
            completion = json.loads(log.read_text().strip().splitlines()[-1])
            assert completion['arm'] == arm and completion['scene'] == scene
            assert completion['frames'] == 40 and completion['scale_mode'] == 'window'
            argv = job['argv']
            assert argv[argv.index('--scale_mode')+1] == 'window'
            assert argv[argv.index('--nms')+1] == 'cfg'
            if first is None:
                first = measured[:5].copy()
            assert np.array_equal(first, measured[:5]), (scene, arm, 'observer prefix')
            replay = plan.copy()
            chunks = []
            for f in range(1, 41, 4):
                hi = f+4
                m = measured[f-1]
                correcting = raw['arm']['replan'] and f > 1
                physical = rejoin(m, plan[f-1], plan[f:hi]) if correcting else plan[f:hi].copy()
                bridge = np.eye(4)
                if correcting and raw['arm']['write'] != 'measured':
                    bridge = replay[f-1] @ np.linalg.inv(m)
                replay[f:hi] = bridge @ physical
                assert np.allclose(replay[f:hi], command[f:hi], atol=1e-12, rtol=0), (scene, arm, f)
                if correcting:
                    assert np.allclose(physical[-1], plan[hi-1], atol=1e-12, rtol=0)
                    # Latest-memory-relative SE(3) target is preserved by the bridge.
                    label = m if raw['arm']['write'] == 'measured' else replay[f-1]
                    relative_error = float(np.abs(np.linalg.inv(label) @ replay[f:hi]
                                                  - np.linalg.inv(m) @ physical).max())
                    assert relative_error < 1e-11
                else:
                    relative_error = None
                # Intended translations before window normalization; not native tensor gain.
                steps = np.diff(np.concatenate([m[None, :3, 3], physical[:, :3, 3]]), axis=0)
                nominal = np.linalg.norm(np.diff(plan[f-1:hi, :3, 3], axis=0), axis=1)
                assert np.all(nominal > 0)
                chunks.append(dict(first_frame=f, correcting=correcting,
                                   intended_step_over_plan=(np.linalg.norm(steps, axis=1)/nominal).tolist(),
                                   latest_label_relative_matrix_max_error=relative_error,
                                   plan_residual_before=float(np.linalg.norm(m[:3,3]-plan[f-1,:3,3])),
                                   stored_label_residual_before=float(np.linalg.norm(m[:3,3]-command[f-1,:3,3]))))
            offline = np.asarray(scored['measured_c2w'])
            assert scored['camera_frame_ids'] == list(range(41))
            pe = np.linalg.norm(offline[:, :3, 3]-plan[:, :3, 3], axis=1)
            assert np.allclose(pe, scored['position_errors'], atol=1e-8, rtol=0)
            assert abs(pe[1:].mean()-scored['horizons']['40']['camera_mean']) < 1e-12
            online_error = np.linalg.norm(measured[:, :3, 3]-plan[:, :3, 3], axis=1)
            disagreement = np.linalg.norm(measured[:, :3, 3]-offline[:, :3, 3], axis=1)
            ratios = [v for c in chunks if c['correcting'] for v in c['intended_step_over_plan']]
            rows.append(dict(scene=scene, arm=arm, settings=settings,
                             archived_queue_state=job['state'], completion_log=str(log.relative_to(W)),
                             score=scored['horizons']['40'], chunks=chunks,
                             corrected_step_over_plan=stats(ratios) if ratios else None,
                             online_position=stats(online_error[1:]),
                             online_endpoint=float(online_error[-1]),
                             online_offline_disagreement=stats(disagreement[1:]),
                             online_offline_endpoint=float(disagreement[-1]),
                             logged_endpoint_command_residual=float(np.linalg.norm(measured[-1,:3,3]-command[-1,:3,3])),
                             command_replay_max_abs=float(np.abs(replay-command).max())))
    assert len(rows) == 40
    arms = ['baseline', 'replan_commanded', 'write_measured', 'replan_measured']
    means = {a: float(np.mean([r['score']['camera_mean'] for r in rows if r['arm'] == a])) for a in arms}
    by_scene = {(r['scene'], r['arm']):r for r in rows}
    scenes = sorted({r['scene'] for r in rows})
    command_given_memory = [by_scene[s,'replan_measured']['score']['camera_mean']
                            - by_scene[s,'write_measured']['score']['camera_mean'] for s in scenes]
    result = dict(scope='40 archived runs, ten scenes, seed 7, 40 generated views; CPU replay only',
                  normalization='All archived argv and metadata: window, nms=cfg; not global_demo.',
                  observer_prefix='All four saved DA3 pose prefixes equal through frame 4 in every scene; not raw-RGB identity qualification.',
                  limitations=['Retrieved context IDs, per-window scaling factors and conditioning tensors were not archived.',
                               'Step ratios are intended pre-normalization geometry, not effective native amplitudes or differential gains.',
                               'DA3 online and VGGT offline poses share the declared anchor gauge; neither is ground truth.',
                               'Current code hashes identify inspected implementations, not a frozen generation-time source snapshot.'],
                  means=means,
                  mean_change_percent={a:100*(means[a]/means['baseline']-1) for a in arms},
                  command_added_to_memory=dict(mean_delta=float(np.mean(command_given_memory)),
                                               lower_cases=sum(x<0 for x in command_given_memory), total=10),
                  rows=rows, input_sha256=HASHES)
    out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(runs=len(rows), hashes=len(HASHES), means=means,
                          command_added_to_memory=result['command_added_to_memory'],
                          max_replay_error=max(r['command_replay_max_abs'] for r in rows))))


if __name__ == '__main__':
    main()
