"""DA3 pose-measurement worker for spatia_fprime.py.

Runs in venv_vmem: venv_spatia cannot import DA3 (missing evo, moviepy, e3nn, gsplat, pycolmap); venv_vmem is the venv in
which vmem_fprime_fixed.py already runs DA3 from the Lyra-2 source tree (memorizon_plugins/realized_pose.load_da3).

Measurement = memorizon_plugins/realized_pose.realized_poses, the same algorithm as gv_memorizon2.da3_window used by
vmem_fprime_fixed.measure_new: DA3 on [anchor + evenly spaced earlier frames + query tail], intrinsics given, NO extrinsics
given; poses re-expressed relative to the anchor view; scale fixed once per DA3 call by the median ratio of the given anchor
depth to DA3's anchor depth (anchor_depth = ones -> 1/median(DA3 anchor depth) = anchor-median-depth units).
The query frames are measured in chunks of `tail` (one DA3 call each), context = every earlier generated frame.

Protocol (one JSON object per line): worker prints  READY {...}  once DA3 is loaded (on CPU); then for each stdin line
  {"cmd": "measure", "req": <npz>, "out": <npz>}  -> reply {"ok": true, ...} after writing <out> (poses (Q,4,4), NaN = none)
  {"cmd": "quit"}                                -> exits
req npz: anchor (H,W,3) u8, frames (N,H,W,3) u8 with frames[i] = video frame i+1, K (3,3) pixel K of these images,
         query (Q,) video frame indices >= 1, anchor_depth (h,w) f32 or size-0 (= keep DA3's own units), max_views, tail.
DA3 is moved to the GPU for each request and back to the CPU afterwards (vmem_fprime_fixed.py:98), so it holds no GPU
memory while Spatia denoises or MapAnything / the point renderer run.
"""
import json, os, sys, time, traceback
import numpy as np
import torch

MP = '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins'
if MP not in sys.path:
    sys.path.insert(0, MP)
from realized_pose import load_da3, realized_poses          # noqa: E402


def measure(da3, anchor, frames, K, query, anchor_depth, max_views, tail):
    """Returns (poses (Q,4,4) NaN where DA3 gave none, per-call info list)."""
    query = [int(q) for q in query]
    out = np.full((len(query), 4, 4), np.nan); calls = []
    for c0 in range(0, len(query), tail):
        chunk = query[c0:c0 + tail]
        ctx = [frames[i - 1] for i in range(1, chunk[0])]                     # every earlier generated frame
        views = ctx + [frames[q - 1] for q in chunk]
        t0 = time.time()
        poses, info = realized_poses(da3, anchor, anchor_depth, views, K, max_views=max_views, tail=len(chunk))
        for k, q in enumerate(chunk):
            p = poses[len(ctx) + k]
            if p is not None and np.isfinite(p).all():
                out[c0 + k] = p
        calls.append(dict(frames=[chunk[0], chunk[-1]], n=len(chunk), scale=float(info['scale']), views=int(info['views']),
                          sec=round(time.time() - t0, 2)))
    return out, calls


class _FakeDA3:
    """GV_DA3_FAKE=1 (protocol test only, no weights): identity cameras, unit depth."""
    def to(self, *a, **k):
        return self

    def inference(self, image, **kw):
        class P:
            pass
        p = P(); n = len(image); h, w = np.asarray(image[0]).shape[:2]
        p.extrinsics = np.tile(np.eye(4)[:3], (n, 1, 1)); p.depth = np.ones((n, h, w), np.float32); return p


def main():
    proto = os.fdopen(os.dup(1), 'w', buffering=1)                            # protocol channel
    os.dup2(2, 1)                                                               # any library print -> stderr
    sys.stdout = sys.stderr
    fake = os.environ.get('GV_DA3_FAKE') == '1'
    dev = 'cuda' if torch.cuda.is_available() and not fake else 'cpu'
    t0 = time.time(); da3 = _FakeDA3() if fake else load_da3('cpu')
    proto.write('READY ' + json.dumps(dict(load_sec=round(time.time() - t0, 1), device=dev, pid=os.getpid(), fake=fake)) + '\n')
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get('cmd') == 'quit':
                break
            z = np.load(req['req'])
            ad = z['anchor_depth']; ad = ad if ad.size else None
            t1 = time.time()
            if dev == 'cuda':
                torch.cuda.reset_peak_memory_stats(); da3.to('cuda')
            try:
                poses, calls = measure(da3, z['anchor'], z['frames'], z['K'], z['query'], ad, int(z['max_views']), int(z['tail']))
            finally:
                if dev == 'cuda':
                    da3.to('cpu'); torch.cuda.empty_cache()
            tmp = req['out'] + '.tmp.npz'
            np.savez(tmp, poses=poses); os.replace(tmp, req['out'])
            proto.write(json.dumps(dict(ok=True, sec=round(time.time() - t1, 2), calls=calls, n_valid=int(np.isfinite(poses[:, 0, 0]).sum()),
                                        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == 'cuda' else 0.)) + '\n')
        except Exception as e:                                                  # keep serving; the driver decides
            traceback.print_exc()
            proto.write(json.dumps(dict(ok=False, error=f'{type(e).__name__}: {e}')) + '\n')


if __name__ == '__main__':
    main()
