"""F' on released VMem (surfel-indexed view memory on Stable Virtual Camera; baselines/venv_vmem; official inference config:
50 steps, CFG 2, 4 context + 4 target frames per step, CUT3R surfels). Same dev scenes / planned paths as Memorizon F'.

Frames are 576x576 centre crops of the 448x256 protocol frame (as baselines/run_vmem.py): anchor cropped to 256x256, poses =
anchor-relative OpenCV c2w in the anchor-depth gauge converted to VMem's OpenGL convention, intrinsics normalised to the crop.
VMem's natural step = 4 new frames (first step 7). F' is applied at that step:
  * measure: DA3 command-free on [cropped anchor + up to 15 earlier generated frames + the new frames] (da3_window, tail = new
    frames, scale from the stored anchor depth cropped the same way);
  * write='measured': the new frames' poses in VMem's state (self.c2ws, used for context camera conditioning AND for placing
    their surfels) are replaced by the measured poses BEFORE the surfels of those frames are built (construct_and_store_scene
    is wrapped);
  * replan=true: the next step's targets are re-planned from the measured pose of the last frame, rejoining the plan by SE(3)
    interpolation over the step (as F').
Output: <out>/<arm>/<scene>_s<seed>.npz (frames 1..N at 256x256 crop) + .json (planned, commanded, measured per frame).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
VM = Path('@WORKSPACE@/baselines/vmem'); sys.path.insert(0, str(VM))
MP = '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins'; sys.path.insert(0, MP)
from realized_pose import load_da3
from gv_memorizon2 import da3_window
from gv_memorizon import se3_interp
BATCH = Path('@WORKSPACE@/analysis/plan_base_restart_20261008/lyra2_plugins/batch_dev10')
COMP = Path('@WORKSPACE@/training_runs/base_restart_20261008/dataset_v3/components')
GL = np.diag([1.0, -1.0, -1.0, 1.0])


class Run:
    def __init__(self, pipe, da3, scene, arm, seed, n_frames):
        self.pipe, self.da3, self.arm, self.seed, self.n = pipe, da3, arm, seed, n_frames
        z = np.load(BATCH / 'trajectories' / f'{scene}.npz')
        self.plan = np.linalg.inv(z['w2c'].astype(np.float64))[:n_frames + 1]             # OpenCV c2w per frame, gauge
        K = z['intrinsics'][0].astype(np.float64); Wd, Hd = int(z['image_width']), int(z['image_height']); self.x0 = x0 = (Wd - Hd) // 2
        self.Kc = K.copy(); self.Kc[0, 2] -= x0                                             # pixel K of the 256x256 crop
        Kn = np.eye(3); Kn[0, 0] = K[0, 0] / Hd; Kn[1, 1] = K[1, 1] / Hd; Kn[0, 2] = (K[0, 2] - x0) / Hd; Kn[1, 2] = K[1, 2] / Hd
        self.Kn = Kn.astype(np.float32)
        idx = json.load(open(BATCH / 'index.json')); sid = [s for s in idx['scenes'] if s['name'] == scene][0]['stream_id']
        self.adepth = np.load(COMP / sid / 'anchor.npz')['depth'][:, x0:x0 + Hd]
        img = Image.open(BATCH / 'images' / f'{scene}.png').convert('RGB').crop((x0, 0, x0 + Hd, Hd)); self.anchor = np.asarray(img)
        self.image = torch.from_numpy(np.asarray(img.resize((576, 576), Image.BICUBIC))).permute(2, 0, 1).float()[None].cuda() / 127.5 - 1
        self.cmd = self.plan.copy(); self.measured = [None] * (n_frames + 1); self.measured[0] = np.eye(4); self.events = []

    def frames256(self, pil):
        return [np.asarray(f.convert('RGB').resize((256, 256), Image.BICUBIC)) for f in pil]

    def measure_new(self, k):
        """DA3 on the k newest frames (pipe.pil_frames[-k:]) -> measured OpenCV c2w (gauge) for frames len-k..len-1."""
        fr = self.frames256(self.pipe.pil_frames[1:]); n = len(fr)
        self.da3.to('cuda'); est = da3_window(self.da3, self.anchor, self.adepth, fr, self.Kc, max_views=16, tail=k); self.da3.to('cpu'); torch.cuda.empty_cache()
        out = {}
        for j in range(n - k, n):
            p = est['poses'][j]
            if p is not None and np.isfinite(p).all():
                out[j + 1] = p
        return out

    def hook(self, orig):
        def wrapped(*a, **kw):
            k = len(self.pipe.pil_frames) - self._done
            meas = self.measure_new(k)
            for f, p in meas.items():
                self.measured[f] = p
                if self.arm.get('write') == 'measured':
                    self.pipe.c2ws[f] = (p @ GL).astype(self.pipe.c2ws[f].dtype)     # pipe.c2ws[0] = anchor; index f = frame f
            self.events.append(dict(kind='step', frames=[self._done, len(self.pipe.pil_frames) - 1],
                                    residual=[float(np.linalg.norm(meas[f][:3, 3] - self.cmd[f][:3, 3])) if f in meas else None for f in range(self._done, len(self.pipe.pil_frames))]))
            self._done = len(self.pipe.pil_frames)
            return orig(*a, **kw)
        return wrapped

    @torch.no_grad()
    def run(self):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        pipe = self.pipe; pipe.reset()
        pipe.initialize(self.image, (self.plan[0] @ GL).astype(np.float32), self.Kn)
        orig = pipe.construct_and_store_scene; pipe.construct_and_store_scene = self.hook(orig); self._done = 1
        try:
            f = 1
            while f <= self.n:
                k = 7 if f == 1 else 4; hi = min(f + k, self.n + 1)
                if self.arm.get('replan') and f > 1 and self.measured[f - 1] is not None:
                    delta = self.measured[f - 1] @ np.linalg.inv(self.cmd[f - 1])
                    for i, j in enumerate(range(f, hi)):
                        self.cmd[j] = se3_interp(delta, 1. - (i + 1) / (hi - f)) @ self.plan[j]
                    self.events.append(dict(kind='replan', frame=f, position_error=float(np.linalg.norm(delta[:3, 3]))))
                c2ws = [(self.cmd[j] @ GL).astype(np.float32) for j in range(f, hi)]
                pipe.generate_trajectory_frames(c2ws, [self.Kn] * len(c2ws))
                f = hi
        finally:
            pipe.construct_and_store_scene = orig
        fr = self.frames256(pipe.pil_frames[1:self.n + 1])
        return np.stack(fr), dict(planned=self.plan.tolist(), commanded=self.cmd.tolist(), crop_x0=self.x0,
                                  measured=[None if m is None else m.tolist() for m in self.measured], events=self.events)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=['1740e916', '38dcc531']); p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    p.add_argument('--frames', type=int, default=80); a = p.parse_args()
    arms = json.load(open(a.arms)); a.out.mkdir(parents=True, exist_ok=True)
    os.chdir(VM)
    from omegaconf import OmegaConf
    from modeling.pipeline import VMemPipeline
    cfg = OmegaConf.load('configs/inference/inference.yaml'); cfg.inference.visualize = False
    pipe = VMemPipeline(cfg, 'cuda'); da3 = load_da3('cpu')
    for arm in arms:
        for scene in a.scenes:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.npz'
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
                frames, log = Run(pipe, da3, scene, arm, seed, a.frames).run()
                np.savez_compressed(dst, frames=frames)
                json.dump(dict(arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, **log), open(dst.with_suffix('.json'), 'w'))
                res = [x for e in log['events'] if e['kind'] == 'step' for x in e['residual'] if x is not None]
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(time.time() - t0, 1), frames=len(frames),
                                      residual_end=round(res[-1], 3) if res else None, peak=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)


if __name__ == '__main__':
    main()
