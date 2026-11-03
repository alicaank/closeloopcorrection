"""Matched exported RGB diagnostic; never reconstructs original raw observer RGB."""
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/home/hj/Alican')
HERE = Path(__file__).resolve().parent
PILOT = ROOT / 'analysis/dreamx_future_command_20261103'
GEOMETRY = ROOT / 'analysis/plan_base_restart_20261008/continuous_eval_20261029/scale_20261030/correction_v3_20261030'


def selected_ids(query):
    """Mirror the one-call worker's context-plus-query selection."""
    q = np.asarray(query, dtype=int)
    q = q[q > 0]
    assert 0 < len(q) <= 8 and np.all(np.diff(q) > 0)
    context = np.arange(1, q[0])
    if 1 + len(context) + len(q) <= 16:
        return np.r_[0, context, q]
    picks = np.linspace(0, len(context) - 1, 15 - len(q)).round().astype(int)
    return np.r_[0, context[picks], q]


def validate_selection():
    # Exercise the actual frozen selection code with image IDs; no model weights.
    spec = importlib.util.spec_from_file_location('native_selection', GEOMETRY / 'snapshots/realized_pose.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    for seed in (7, 107):
        events = json.loads((PILOT / 'runs' / f'dreamx_0238de14_s{seed}' / 'F/future_events.json').read_text())
        for e in events:
            q = np.asarray(e['query_frames']); q = q[q > 0]
            original_ids = [*range(1, q[0]), *q]
            class Capture:
                def inference(self, image, **kwargs):
                    self.ids = np.array([int(np.asarray(x)[0, 0, 0]) for x in image])
                    class Pred: pass
                    p = Pred(); p.extrinsics = np.repeat(np.eye(4)[None, :3], len(image), axis=0)
                    p.depth = np.ones((len(image), 2, 2)); return p
            capture = Capture()
            frame = lambda i: np.full((2, 2, 3), i, np.uint8)
            module.realized_poses(capture, frame(0), np.ones((2, 2)),
                                  [frame(i) for i in original_ids], np.eye(3), max_views=16, tail=len(q))
            ids = selected_ids(e['query_frames'])
            np.testing.assert_array_equal(ids, capture.ids)
            assert len(ids) == 16 and ids[-1] == e['receipt']['available_frame']
            checks.append(dict(seed=seed, available_frame=int(ids[-1]), frame_ids=ids.tolist()))
    return checks


def gap(a, b, target):
    ra, rb = target[:3, 3] - a[:3, 3], target[:3, 3] - b[:3, 3]
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return dict(position=float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())),
                residual_cosine=float(ra @ rb / den) if den else None)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--validate', action='store_true'); args = p.parse_args()
    if args.validate:
        print(json.dumps(dict(passed=True, cases=validate_selection(), gpu_inference=False))); return
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    sys.path[:0] = [str(PILOT / 'source'), str(GEOMETRY)]
    from score_memory_pair import load_frames, estimate
    from guarded_observer import realized_poses, native
    import torch
    import os
    os.environ['GV_OBSERVER_QUALITY'] = 'enforce'
    out = HERE / 'results'; out.mkdir(exist_ok=False)
    report = dict(protocol=manifest['protocol'], windows=[], limitations=manifest['limitations'])
    batch = Path(manifest['batch'])
    with np.load(batch / 'trajectories/0238de14.npz') as z:
        K, depth, target = z['intrinsics'][0], z['anchor_depth'], np.linalg.inv(z['w2c_overrun'].astype(float))
    for seed in (7, 107):
        key = f'dreamx_0238de14_s{seed}'
        run = PILOT / 'runs' / key / 'F'
        frames, fmt = load_frames(run, '0238de14')
        assert fmt == 'lossless_ffv1' and frames.shape == (69, *depth.shape, 3)
        with np.load(PILOT / 'scores' / key / 'F.npz') as z: full = z['c2w']
        events = json.loads((run / 'future_events.json').read_text())
        for event in events:
            ids = selected_ids(event['query_frames']); b = int(ids[-1])
            folder = out / f'{key}_f{b}'; folder.mkdir()
            selected = frames[ids].copy()
            row = dict(case=key, available_frame=b, selected_frame_ids=ids.tolist(),
                       selected_rgb_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                       causal_prefix_rgb_sha256=hashlib.sha256(frames[:b+1].tobytes()).hexdigest())
            model = native.load_da3('cuda')
            try:
                poses, info = realized_poses(model, selected[0], depth, list(selected[1:]), K, max_views=16, tail=15)
                da3 = np.asarray(poses)
                np.savez_compressed(folder / 'da3.npz', c2w=da3, frame_ids=ids[1:])
                row['da3'] = info
            except Exception as exc:
                da3 = None; row['da3'] = dict(passed=False, error=repr(exc), quality=getattr(exc, 'quality', None))
            finally:
                del model; gc.collect(); torch.cuda.empty_cache()
            for mode, rgb in [('selected', selected), ('causal_prefix', frames[:b+1])]:
                try:
                    poses, info = estimate(rgb, K, depth, folder / f'vipe_{mode}.npz')
                    row['vipe_' + mode] = info
                    row['vipe_' + mode + '_vs_full'] = gap(poses[-1], full[b], target[b]) if info['registration']['valid'] else None
                    row['da3_vs_vipe_' + mode] = gap(da3[-1], poses[-1], target[b]) if da3 is not None and info['registration']['valid'] else None
                except Exception as exc:
                    row['vipe_' + mode] = dict(passed=False, error=repr(exc))
            if da3 is not None:
                original = np.asarray(event['observation'])[-1]
                row['rerun_da3_vs_original_online'] = gap(da3[-1], original, target[b])
            report['windows'].append(row)
            (out / 'partial.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
