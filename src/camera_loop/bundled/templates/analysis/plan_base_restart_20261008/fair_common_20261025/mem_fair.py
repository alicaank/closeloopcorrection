"""Fair-inference copy of memorizon_plugins/gv_memorizon2.py (released Memorizon, 4-step distilled) on batch_ours10, F' arms only.

Changes vs gv_memorizon2.py (everything else, incl. the rollout math, is the same code path):
  * inputs: batch_ours10 (our 10 dev videos) instead of batch_dev10; the anchor is the dataset's 864x480 frame 0 passed to the
    official MemorizonPipeline.encode_image (centre-crop to 864:480 + LANCZOS resize = identity here) instead of a 448x256
    image upsampled to 864x480; intrinsics = the true 864x480 pixel K (before: the 448 K scaled anisotropically);
  * prompt: the scene's dataset caption with Memorizon's required perspective prefix (memorizon/caption.py) instead of the
    generic batch prompt;
  * frames 0..160 saved at the native 864x480 (before: npz resized to 448x256) as a crf-12 mp4, frame 0 = the processed input
    frame, + json with per-frame poses and the output K (fair_common.save_run); the DA3 measurement runs on the native frames with the anchor gauge depth resampled to 864x480;
  * only the arm keys of the F' 2x2 (write commanded|measured, replan) are kept; cfg / bestof / hybrid / inject / geoinit /
    redenoise / reproject / gradguide / gate branches and the --ckpt fine-tune path are removed (never used by these arms);
  * per-run disk floor check; --arm filter.
Settings = official (pipeline.py __call__ with the checkpoint config): 4 steps, guidance 0 (default_guidance_scale 0.0 ->
B=1, no negative prompt), memory_size 6, chunk 4 latents, 10 chunks = 1 + 40 latents = frames 0..160, seed via a CPU generator.
Camera units: runtime corresponding-pixel metric depth/reference depth, divided by the native training divisor 4; missing metric geometry is rejected.
Usage (venv_memorizon, cwd @WORKSPACE@/baselines):
  mem_fair.py --out RUNS/memorizon --arms arms_4.json [--arm NAME ...] [--scenes ...] [--seeds 7]"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'fair_common_20261025'))
from causal_units import metric_prediction_scale

import numpy as np, torch
from PIL import Image

ROOT = Path('@WORKSPACE@/baselines'); sys.path.insert(0, str(ROOT / 'memorizon'))
from memorizon import MemorizonPipeline
from memorizon.model import MemorizonModel
from memorizon.retrieval import Frustum, select_bank
from memorizon.layout import Layout, rope_index, attention_mask
from memorizon.camera import camera_transforms
from memorizon.wan.transformer import frame_mask_to_block_mask
MP = '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins'; sys.path.insert(0, MP)
from realized_pose import load_da3
from gv_memorizon import StreamDecoder                     # causal Wan VAE decode, one latent at a time (same math as pipeline.decode)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_common as fc

TRAJ_SCALE = 4.0                                           # memorizon units = metres / 4 (README trajectory format)


class Roller:
    @torch.no_grad()
    def __init__(self, pipe, da3, name, arm, seed, memory_size=6, num_steps=4):
        self.pipe, self.da3, self.arm, self.seed = pipe, da3, arm, seed
        sc = fc.scene(name); self.name = name
        self.plan = sc['c2w'][0:161:4].copy()                                          # planned c2w per latent j (frame 4j), gauge
        W, H = pipe.model.config.image_width, pipe.model.config.image_height
        img = Image.open(sc['image_path']).convert('RGB')
        self.K, box = fc.k_memorizon(sc['K'], img.size, (W, H))                         # encode_image geometry (identity at 864x480)
        self.anchor = np.asarray(img.crop(box).resize((W, H), Image.LANCZOS))           # = the pixels encode_image VAE-encodes
        self.anchor_depth = fc.remap_depth(sc['depth'], sc['K'], self.K, (H, W))
        self.intr = (self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])
        self.prompt = sc['prompts']['memorizon']; self.text = pipe.encode_prompt(self.prompt).to(pipe.device, pipe.dtype)
        da3.to('cuda'); pm = da3.inference(image=[self.anchor], intrinsics=self.K[None].astype(np.float32), infer_gs=False, process_res=504, export_dir=None)
        da3.to('cpu'); torch.cuda.empty_cache()
        self.unit, self.unit_calibration = metric_prediction_scale(da3, pm, self.anchor, self.anchor_depth, self.K, divisor=TRAJ_SCALE)
        self.metric = self.unit_calibration["metres_per_reference_unit"]
        self.dit, self.C = pipe.model.model, pipe.model.config.chunk_size
        self.frustum = Frustum.from_intrinsics(list(self.intr)); self.memory_size = memory_size
        first = pipe.encode_image(img)                                                  # official crop / resize / VAE encode
        _, self.ch, _, self.h, self.w = first.shape; _, p_h, p_w = self.dit.config.patch_size; self.tpf = (self.h // p_h) * (self.w // p_w)
        self.sched = pipe.model.noise_scheduler; self.sched.set_timesteps(num_steps); self.timesteps = self.sched.timesteps.to(pipe.device)
        self.intr_b = torch.as_tensor(np.asarray(self.intr, dtype=np.float32))[None].to(pipe.device)
        self.gen = torch.zeros((1, self.ch, 41, self.h, self.w), device=pipe.device, dtype=pipe.dtype); self.gen[:, :, 0] = first[:, :, 0]
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.dec = StreamDecoder(pipe); self.frames = list(self.dec(self.gen[:, :, :1]))    # native 864x480 uint8
        self.cmd = self.plan.copy(); self.mem = self.plan.copy(); self.measured = [None] * 41; self.measured[0] = np.eye(4)
        self.residual = [None] * 41; self.events = []; self.g_prev = None

    @torch.no_grad()
    def denoise(self, ci, cams_np):
        """gv_memorizon2.Roller.denoise with cfg_w=1, no x_init / anchor / inject (= MemorizonPipeline.rollout at B=1)."""
        C, q_lo, q_hi = self.C, 1 + ci * self.C, 1 + (ci + 1) * self.C
        k_mem = 0 if self.arm.get('bank') == 'drop' else self.memory_size
        n_rec = min(C, q_lo); recent = np.arange(q_lo - n_rec, q_lo)
        cand = np.array([i for i in range(1, max(1, q_lo - n_rec))], dtype=int)
        bank = select_bank(cams_np, cand, cams_np[q_hi - 1][None], self.frustum, k=k_mem, q0=q_lo, chunk=C, cap=max(k_mem, 1)) if k_mem else np.zeros(0, int)
        lay = Layout(n_bank=len(bank), n_recent=n_rec, chunk=C, n_query_chunks=1); F_, n_cond = lay.total, lay.n_cond
        rope = torch.as_tensor(rope_index(lay), dtype=torch.long, device=self.pipe.device)
        mask = (attention_mask(lay, bank, q_lo, cams_np[q_lo:q_hi], cams_np[bank], frustum=self.frustum, k=self.memory_size) if len(bank) else attention_mask(lay))
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.pipe.device); ctx_mask = frame_mask_to_block_mask(mask[:n_cond, :n_cond], self.tpf)
        idx = np.concatenate([[0], bank, recent]).astype(int)
        cams = torch.as_tensor(np.concatenate([cams_np[idx], cams_np[q_lo:q_hi]]), dtype=torch.float32, device=self.pipe.device)[None]
        x = torch.randn((1, self.ch, F_, self.h, self.w), generator=self.rng).to(self.pipe.device, self.pipe.dtype)
        cond = self.gen[:, :, idx]; x[:, :, :n_cond] = cond
        cam_ctx = camera_transforms(self.dit, cams[:, :n_cond], self.intr_b); cam_q = camera_transforms(self.dit, cams[:, n_cond:], self.intr_b)
        kv = self.dit.init_kv_cache(1, F_, self.h, self.w)
        with torch.autocast('cuda', torch.bfloat16):
            self.dit(cond, torch.zeros((1, n_cond), device=self.pipe.device), self.text, rope_index=rope[:n_cond], attention_mask=ctx_mask, camera=cam_ctx, kv_cache=kv, kv_cache_range=(0, n_cond))
            for layer in kv:
                layer.pop('cross_attn_key', None); layer.pop('cross_attn_value', None)
            for t in self.timesteps:
                ts = t.repeat(1, C).clone()
                pred = self.dit(x[:, :, n_cond:], ts, self.text, rope_index=rope, camera=cam_q, kv_cache=kv, kv_cache_range=(0, F_))
                x[:, :, n_cond:] = self.sched.step_diff_noise_level(pred, ts, x[:, :, n_cond:]).to(x)
        out = x[0:1, :, n_cond:].clone(); del kv, x, cond, cam_ctx, cam_q; torch.cuda.empty_cache()
        return out, bank

    @torch.no_grad()
    def measure(self, q_lo, q_hi):
        lat_frames = [self.frames[4 * j] for j in range(1, q_hi)]                       # latent j <-> frame 4j (its last RGB frame)
        torch.cuda.empty_cache(); self.da3.to('cuda')
        est = fc.da3_window(self.da3, self.anchor, self.anchor_depth, lat_frames, self.K, max_views=16, tail=4)
        self.da3.to('cpu'); torch.cuda.empty_cache()
        for j in range(q_lo, q_hi):
            p = est['poses'][j - 1]
            if p is not None and np.isfinite(p).all():
                self.measured[j] = p; self.residual[j] = float(np.linalg.norm(p[:3, 3] - self.cmd[j][:3, 3]))
        return est

    @torch.no_grad()
    def run(self):
        arm, C = self.arm, self.C
        for ci in range(10):
            q_lo, q_hi = 1 + ci * C, 1 + (ci + 1) * C
            if arm.get('replan') and ci > 0 and self.measured[q_lo - 1] is not None:       # F': start at the measured pose, rejoin the plan
                delta = self.measured[q_lo - 1] @ np.linalg.inv(self.cmd[q_lo - 1])
                for i, j in enumerate(range(q_lo, q_hi)):
                    self.cmd[j] = fc.se3_interp(delta, 1. - (i + 1) / C) @ self.plan[j]
                self.events.append(dict(kind='replan', chunk=ci, position_error=float(np.linalg.norm(delta[:3, 3]))))
            self.mem[q_lo:q_hi] = self.cmd[q_lo:q_hi]
            cams_np = self.mem.copy(); cams_np[:, :3, 3] *= self.unit
            lat, bank = self.denoise(ci, cams_np)
            self.gen[:, :, q_lo:q_hi] = lat; self.frames += list(self.dec(lat))
            est = self.measure(q_lo, q_hi)
            for j in range(q_lo, q_hi):
                if self.measured[j] is not None and arm.get('write') == 'measured':
                    self.mem[j] = self.measured[j]
            a, b = self.measured[q_lo - 1], self.measured[q_hi - 1]
            if a is not None and b is not None:
                self.g_prev = float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / max(np.linalg.norm(self.cmd[q_hi - 1][:3, 3] - self.cmd[q_lo - 1][:3, 3]), 1e-6))
            self.events.append(dict(kind='chunk', chunk=ci, g=self.g_prev, bank=[int(x) for x in bank], da3_scale=est['scale'],
                                    residual=[self.residual[j] for j in range(q_lo, q_hi)]))
        return self.frames, dict(unit_calibration=self.unit_calibration, metric_anchor_depth_m=self.metric, unit=self.unit, prompt=self.prompt, events=self.events, pose_frame_ids=[4 * j for j in range(41)],
                                 commanded=self.cmd.tolist(), planned=self.plan.tolist(), measured=[None if m is None else m.tolist() for m in self.measured])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--arm', nargs='*', default=None, help='subset of arm names'); p.add_argument('--scenes', nargs='+', default=None)
    p.add_argument('--seeds', nargs='+', type=int, default=[7]); p.add_argument('--steps', type=int, default=None, help='default: checkpoint config (4)')
    p.add_argument('--memory_size', type=int, default=6); p.add_argument('--min_free_gib', type=float, default=10.)
    a = p.parse_args()
    arms = [x for x in json.load(open(a.arms)) if not a.arm or x['name'] in a.arm]; a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes or [s['name'] for s in fc.index()['scenes']]
    pipe = MemorizonPipeline(MemorizonModel.from_pretrained(str(ROOT / 'models/memorizon_local'))); da3 = load_da3('cpu')
    cfg = pipe.model.config; steps = a.steps or getattr(cfg, 'default_num_steps', 4)
    assert getattr(cfg, 'default_guidance_scale', 0.0) == 0.0, 'distilled checkpoint expected (no CFG)'
    for arm in arms:
        for scene in scenes:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.mp4'
                if dst.exists():
                    continue
                if fc.free_gib() < a.min_free_gib:
                    print(json.dumps(dict(stop='disk floor', free_gib=round(fc.free_gib(), 2))), flush=True); sys.exit(3)
                t0 = time.time(); torch.cuda.reset_peak_memory_stats()
                try:
                    r = Roller(pipe, da3, scene, arm, seed, memory_size=a.memory_size, num_steps=steps); frames, log = r.run()
                except Exception as e:
                    import traceback; tb = traceback.format_exc().strip().splitlines()
                    print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, error=repr(e)[:200], where=[l.strip() for l in tb if 'mem_fair.py' in l][-2:],
                                          peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True); torch.cuda.empty_cache(); continue
                log = dict(model='memorizon', arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                           settings=dict(steps=steps, guidance=0.0, memory_size=a.memory_size, chunks=10, hw=list(r.anchor.shape[:2])), **log)
                fc.save_run(dst, frames[1:], r.anchor, r.K, log, preprocess='memorizon')
                res = [x for e in log['events'] if e['kind'] == 'chunk' for x in e['residual'] if x is not None]
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(log['seconds'], 1), frames=len(frames) - 1, residual_end=round(res[-1], 3) if res else None,
                                      peak=round(log['peak_GiB'], 1))), flush=True)


if __name__ == '__main__':
    main()
