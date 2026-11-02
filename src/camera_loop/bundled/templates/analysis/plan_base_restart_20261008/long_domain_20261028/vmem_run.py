"""Fair-inference copy of model_controls_20261024/vmem_fprime_fixed.py (released VMem, SEVA-based, 576x576) on batch_ours10.
venv_vmem; cwd is switched to @WORKSPACE@/baselines/vmem (configs/inference/inference.yaml).

Changes vs vmem_fprime_fixed.py:
  * inputs: batch_ours10; the dataset's 864x480 frame 0 prepared the official way for every video (app.py:107-111
    load_img_and_K + transform_img_and_K(576x576, mode='crop'): area resize to 1037x576, centre crop; tensor_to_pil into
    pipeline.initialize as app.py:184-190) instead of a 256x256 crop of the 448x256 image upsampled to 576; intrinsics = our true
    K through the same transform (transform_img_and_K's own K branch, cross-checked with fair_common.k_vmem), normalised by 576;
  * defaults = the official navigation-demo settings the 10-24 control reproduced exactly: --first_step 4 (Navigator call size),
    --nms 0 (Navigator passes use_non_maximum_suppression=False for moves, navigation.py:187,236; our walking paths translate in
    every call), --autocast 1 (app.py:156), --scale_mode window (native normalization recomputed for each conditioning window; no frozen demo factor).
  * 160 frames = 40 calls of 4 (frames 1..160), saved with frame 0 = the processed input at the native 576x576 (before: npz at
    256x256) as a crf-12 mp4 + json with per-frame poses and the output K (fair_common.save_run); DA3 measurement on native
    frames with the anchor gauge depth resampled to the 576 crop;
  * custom-image mode (--image/--traj_npy) removed; scene-major loop (both arms of a scene before the next scene); disk floor.
No text input (VMem is image + camera only). Remaining official-code patches in the repo (git diff, baselines/vmem): SD2.1 VAE
from a local copy (autoencoder.py:13), NMS threshold computed at the first multi-frame step (pipeline.py:678-707), retrieval
fallback when no surfel is visible (pipeline.py:663-666), context padding when retrieval returns < 4 frames (pipeline.py:762-767)."""
import argparse, contextlib, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
VM = Path('@WORKSPACE@/baselines/vmem'); sys.path.insert(0, str(VM))
MP = '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins'; sys.path.insert(0, MP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'fair_common_20261025'))
from realized_pose import load_da3
import fair_common as fc
from causal_units import capture_native_normalizer
fc.BATCH=Path(os.environ['LONG_BATCH']);fc.N_FRAMES=40
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'new_models_loop_20261028'))
from common import rejoin

GL = np.diag([1.0, -1.0, -1.0, 1.0])
SIDE = 576


class Run:
    def __init__(self, pipe, da3, name, arm, seed, n_frames, opt):
        from utils import load_img_and_K, transform_img_and_K, tensor_to_pil
        self.pipe, self.da3, self.arm, self.seed, self.n, self.opt = pipe, da3, arm, seed, n_frames, opt
        sc = fc.scene(name); self.plan = sc['c2w'][:n_frames + 1].copy()                 # OpenCV c2w per frame, gauge
        img, _ = load_img_and_K(sc['image_path'], None, K=None, device='cuda')
        img, Kt = transform_img_and_K(img, (SIDE, SIDE), mode='crop', K=torch.as_tensor(sc['K'], dtype=torch.float32)[None].to(img.device))
        self.image = tensor_to_pil(img[0])                                              # app.py: the navigator gets tensor_to_pil(video[0])
        self.K = Kt[0].double().cpu().numpy(); Kf, _ = fc.k_vmem(sc['K'], Image.open(sc['image_path']).size, (SIDE, SIDE))
        assert np.allclose(self.K, Kf, atol=1e-3), f'K mismatch {self.K} vs {Kf}'
        self.Kn = self.K.copy(); self.Kn[0] /= SIDE; self.Kn[1] /= SIDE; self.Kn = self.Kn.astype(np.float32)   # VMem takes normalised K
        self.anchor = np.asarray(self.image.convert('RGB'))
        self.adepth = fc.remap_depth(sc['depth'], sc['K'], self.K, (SIDE, SIDE))
        self.cmd = self.plan.copy(); self.measured = [None] * (self.n + 1); self.measured[0] = np.eye(4); self.events = []
        self.nms = None if opt.nms == 'cfg' else bool(int(opt.nms))

    def frames_native(self, pil):
        return [np.asarray(f.convert('RGB')) for f in pil]

    def measure_new(self, k):
        fr = self.frames_native(self.pipe.pil_frames[1:]); n = len(fr)
        self.da3.to('cuda'); est = fc.da3_window(self.da3, self.anchor, self.adepth, fr, self.K, max_views=16, tail=k); self.da3.to('cpu'); torch.cuda.empty_cache()
        return {j + 1: est['poses'][j] for j in range(n - k, n) if est['poses'][j] is not None and np.isfinite(est['poses'][j]).all()}

    def hook(self, orig):
        def wrapped(*a, **kw):
            k = len(self.pipe.pil_frames) - self._done
            with torch.random.fork_rng(devices=[0]):
                meas = self.measure_new(k) if self.arm['name']!='native' else {}
            for f, p in meas.items():
                self.measured[f] = p
                if self.arm.get('write') == 'measured':
                    self.pipe.c2ws[f] = (p @ GL).astype(self.pipe.c2ws[f].dtype)
            self.events.append(dict(kind='step', frames=[self._done, len(self.pipe.pil_frames) - 1],
                                    residual=[float(np.linalg.norm(meas[f][:3, 3] - self.cmd[f][:3, 3])) if f in meas else None for f in range(self._done, len(self.pipe.pil_frames))]))
            self._done = len(self.pipe.pil_frames)
            return orig(*a, **kw)
        return wrapped

    @torch.no_grad()
    def run(self):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        pipe = self.pipe; pipe.reset()
        pipe.get_translation_scaling_factor = capture_native_normalizer(
            pipe.get_translation_scaling_factor, self.events.append, lambda: len(pipe.pil_frames)-1)
        ac = torch.autocast('cuda') if self.opt.autocast else contextlib.nullcontext()
        orig = pipe.construct_and_store_scene
        try:
            with ac:
                pipe.initialize(self.image, (self.plan[0] @ GL).astype(np.float32), self.Kn)
                pipe.construct_and_store_scene = self.hook(orig); self._done = 1
                f = 1
                while f <= self.n:
                    k = self.opt.first_step if f == 1 else 4; hi = min(f + k, self.n + 1)
                    every = int(self.arm.get('every', 0))                                 # 10-26 damped arm: correct every `every` frames (0 = every call)
                    due = every == 0 or (f - 1) % every == 0
                    if self.arm.get('replan') and f>1 and due and self.measured[f-1] is None:raise RuntimeError('Missing feedback pose')
                    if self.arm.get('replan') and f > 1 and due and self.measured[f - 1] is not None:
                        measured=self.measured[f-1];future=self.plan[f:hi];gain=float(self.arm.get('gain',1.))
                        if gain==0:
                            desired=future.copy()
                            assert np.array_equal(rejoin(self.plan[f-1],self.plan[f-1],future),future)
                        else:
                            assert gain==1., 'No evaluation-set gain tuning'
                            desired=rejoin(measured,self.plan[f-1],future)
                            if self.arm.get('write')!='measured':desired=(self.cmd[f-1]@np.linalg.inv(measured))@desired
                        self.cmd[f:hi]=desired
                        self.events.append(dict(kind='replan',frame=f,position_error=float(np.linalg.norm(measured[:3,3]-self.plan[f-1,:3,3])),applied_fraction=gain,span=hi-f))
                    c2ws = [(self.cmd[j] @ GL).astype(np.float32) for j in range(f, hi)]
                    pipe.generate_trajectory_frames(c2ws, [self.Kn] * len(c2ws), use_non_maximum_suppression=self.nms)
                    f = hi
        finally:
            pipe.construct_and_store_scene = orig
            if 'get_translation_scaling_factor' in pipe.__dict__:
                del pipe.get_translation_scaling_factor
        fr = self.frames_native(pipe.pil_frames[1:self.n + 1])
        return fr, dict(planned=self.plan.tolist(), commanded=self.cmd.tolist(), pose_frame_ids=list(range(self.n + 1)),
                        measured=[None if m is None else m.tolist() for m in self.measured], events=self.events,
                        settings=dict(first_step=self.opt.first_step, nms=self.opt.nms, autocast=self.opt.autocast, scale_mode=self.opt.scale_mode,
                                      side=SIDE, frames=self.n, steps=pipe.config.model.inference_num_steps, cfg=pipe.config.model.cfg))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--arm', nargs='*', default=None); p.add_argument('--scenes', nargs='+', default=None); p.add_argument('--seeds', nargs='+', type=int, default=[7])
    p.add_argument('--frames', type=int, default=fc.N_FRAMES); p.add_argument('--first_step', type=int, default=4, choices=[4, 7])
    p.add_argument('--nms', default='0', choices=['cfg', '0', '1']); p.add_argument('--autocast', type=int, default=1)
    p.add_argument('--scale_mode', default='window', choices=['window'])
    p.add_argument('--min_free_gib', type=float, default=10.); a = p.parse_args()
    arms = [x for x in json.load(open(a.arms)) if not a.arm or x['name'] in a.arm]; a.out = a.out.resolve(); a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes or [s['name'] for s in fc.index()['scenes']]
    os.chdir(VM)
    from omegaconf import OmegaConf
    from modeling.pipeline import VMemPipeline
    cfg = OmegaConf.load('configs/inference/inference.yaml'); cfg.inference.visualize = False; cfg.model.cfg=3.0
    pipe = VMemPipeline(cfg, 'cuda'); da3 = load_da3('cpu')
    for scene in scenes:                                                                  # scene-major: paired arms finish together
        for arm in arms:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.mp4'
                if dst.exists():
                    continue
                if fc.free_gib() < a.min_free_gib:
                    print(json.dumps(dict(stop='disk floor', free_gib=round(fc.free_gib(), 2))), flush=True); sys.exit(3)
                t0 = time.time(); torch.cuda.reset_peak_memory_stats()
                r = Run(pipe, da3, scene, arm, seed, a.frames, a); frames, log = r.run()
                log = dict(model='vmem', arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, peak_GiB=torch.cuda.max_memory_allocated() / 2**30, **log)
                fc.save_run(dst, frames, r.anchor, r.K, log, preprocess='vmem')
                res = [x for e in log['events'] if e['kind'] == 'step' for x in e['residual'] if x is not None]
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(log['seconds'], 1), frames=len(frames), scale_mode=a.scale_mode,
                                      residual_end=round(res[-1], 3) if res else None, peak=round(log['peak_GiB'], 1))), flush=True)


if __name__ == '__main__':
    main()
