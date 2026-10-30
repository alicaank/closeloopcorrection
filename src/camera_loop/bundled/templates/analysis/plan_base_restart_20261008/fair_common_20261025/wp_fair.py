"""Fair-inference copy of worldplay_plugins/wp_fprime.py (HY-WorldPlay-5B, Wan2.2-5B 4-step distilled) on batch_ours10.
venv_worldplay; launch: cd @WORKSPACE@/sota_repos/worldplay && torchrun --nproc_per_node=1 --master_port=29541 wp_fair.py ...

Changes vs wp_fprime.py:
  * inputs: batch_ours10; image_path = the dataset's 864x480 frame 0, prepared by the pipeline's own resize_center_crop_to_tensor
    (LANCZOS resize to cover 960x544 -> 979x544, centre crop) instead of a 448x256 image upscaled; intrinsics = our true K mapped
    through that resize/crop (fair_common.k_worldplay), normalised as hyvideo pose_to_input does (fx/2cx, fy/2cy, 0.5, 0.5);
  * prompt: the scene's dataset caption (WorldPlay has no caption template; prompt rewriting is off by default in run.sh);
  * length: 11 chunks (44 latents = frames 0..172) and frames 1..160 kept, so every model is scored on 160 frames (10 chunks give
    only 156); the poses of latents 41..43 (frames 164..172, only used by the discarded part of the last chunk) come from the
    same recorded path (batch w2c_overrun);
  * frames 0..160 saved at the native 544x960 (before: npz at 448x256) as a crf-12 mp4, frame 0 = the processed input frame,
    + json with per-frame poses and the output K (fair_common.save_run); DA3 measurement
    on native frames with the anchor gauge depth resampled to 544x960;
  * synthetic / action_zero / prope_static / cmd_gain controls removed; per-run disk floor check; --arm filter.
Settings kept from wan/generate.py: guidance 1 (no CFG), few-step 4, use_memory, context_window_length 16, first_chunk_size 4.
Deviation kept from before: 544x960 instead of the official 704x1280 (OOM on 32 GB at chunk 5; peak 28.0 GB at 544x960).
Camera units: --unit WorldPlay translation units per anchor-depth unit (1.0 as in the 10-23 runs)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
WP = '@WORKSPACE@/sota_repos/worldplay'; sys.path[:0] = [WP, WP + '/wan']
MP = '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins'; sys.path.insert(0, MP)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wan.inference.pipeline_wan_w_mem_relative_rope import WanPipeline
from wan.models.dits.arwan_w_action_w_mem_relative_rope import WanTransformer3DModel
from wan.distributed import maybe_init_distributed_environment_and_model_parallel
from wan.models.par_vae.tools import DistController
from wan.models.par_vae.context_parallel.wrapper_vae import DistWrapper
from wan.inference.helper import MyVAE
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation
from realized_pose import load_da3
import fair_common as fc

MODEL_ID = '@WORKSPACE@/baselines/models/Wan2.2-TI2V-5B-Diffusers'
WDIR = '@WORKSPACE@/baselines/models/worldplay5b'
H, W, C = 544, 960, 4
MAP = {(0, 0, 0, 0): 0, (1, 0, 0, 0): 1, (0, 1, 0, 0): 2, (0, 0, 1, 0): 3, (0, 0, 0, 1): 4,
       (1, 0, 1, 0): 5, (1, 0, 0, 1): 6, (0, 1, 1, 0): 7, (0, 1, 0, 1): 8}


def action_label(prev_c2w, c2w):
    """verbatim wp_fprime.py: hyvideo.generate.pose_to_input action rule for one latent step (translation class * 9 + rotation)."""
    rel = np.linalg.inv(prev_c2w) @ c2w; t = rel[:3, 3]; n = np.linalg.norm(t); tr = [0, 0, 0, 0]; ro = [0, 0, 0, 0]
    if n > 1e-4:
        ang = np.degrees(np.arccos(np.clip(t / n, -1, 1)))
        if ang[2] < 60: tr[0] = 1
        elif ang[2] > 120: tr[1] = 1
        if ang[0] < 60: tr[2] = 1
        elif ang[0] > 120: tr[3] = 1
    e = Rotation.from_matrix(rel[:3, :3]).as_euler('xyz', degrees=True)
    if e[1] > 5e-2: ro[0] = 1
    elif e[1] < -5e-2: ro[1] = 1
    if e[0] > 5e-2: ro[2] = 1
    elif e[0] < -5e-2: ro[3] = 1
    return MAP[tuple(tr)] * 9 + MAP[tuple(ro)]


class WorldPlay:
    """verbatim wp_fprime.WorldPlay (model loading)."""
    def __init__(self):
        os.environ.setdefault('RANK', '0'); os.environ.setdefault('LOCAL_RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1')
        torch.cuda.set_device(0); self.device = torch.device('cuda:0')
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        maybe_init_distributed_environment_and_model_parallel(1, sp_size=1, distributed_init_method='env://')
        self.vae = MyVAE.from_pretrained(MODEL_ID, subfolder='vae', torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        self.pipe = WanPipeline.from_pretrained(MODEL_ID, vae=self.vae, transformer=None, torch_dtype=torch.bfloat16)
        self.pipe.dist_vae = DistWrapper(self.vae, DistController(0, 0, 1, None), None, 4)
        cfg = WanTransformer3DModel.load_config(WDIR + '/wan_transformer')
        tr = WanTransformer3DModel.from_config(cfg).to(torch.bfloat16); tr.add_discrete_action_parameters()
        tr.load_state_dict(load_file(WDIR + '/generator_bf16.safetensors'), strict=True)
        self.pipe.transformer = tr.to(torch.bfloat16).eval().requires_grad_(False)
        self.text_encoder = self.pipe.text_encoder; self.pipe.text_encoder = None
        self.pipe.to(self.device); self.emb = {}

    @torch.no_grad()
    def prompt(self, text):
        if text not in self.emb:
            self.text_encoder.to(self.device); self.pipe.text_encoder = self.text_encoder
            pe, ne = self.pipe.encode_prompt(prompt=text, negative_prompt='', do_classifier_free_guidance=False, max_sequence_length=512, device=self.device)
            self.pipe.text_encoder = None; self.text_encoder.to('cpu'); torch.cuda.empty_cache()
            self.emb[text] = (pe, ne)
        return self.emb[text]


class Roller:
    def __init__(self, wp, da3, name, arm, seed, chunks, unit):
        self.wp, self.da3, self.arm, self.seed, self.chunks, self.unit = wp, da3, arm, seed, chunks, unit
        sc = fc.scene(name); n_lat = 4 * chunks
        self.plan = sc['c2w_ext'][0:4 * (n_lat - 1) + 1:4].copy()                    # c2w per latent j (frame 4j), gauge
        assert len(self.plan) == n_lat, f'{len(self.plan)} poses for {n_lat} latents'
        self.text = sc['prompts']['worldplay']; self.image = sc['image_path']
        src = Image.open(self.image).convert('RGB')
        self.K, (nw, nh, l, t) = fc.k_worldplay(sc['K'], src.size, (W, H))
        x = wp.pipe.resize_center_crop_to_tensor(self.image, (W, H))                    # the pipeline's own preprocessing ([C, H, W] in [-1, 1])
        self.anchor = ((x.float() + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
        ref = np.asarray(src.resize((nw, nh), Image.LANCZOS).crop((l, t, l + W, t + H)))
        assert np.abs(self.anchor.astype(int) - ref.astype(int)).max() <= 1, 'k_worldplay geometry disagrees with the pipeline'
        self.anchor_depth = fc.remap_depth(sc['depth'], sc['K'], self.K, (H, W))
        self.cmd = self.plan.copy(); self.mem = self.cmd.copy(); self.measured = [None] * n_lat; self.measured[0] = np.eye(4); self.events = []
        k = self.K.copy(); k[0, 0] /= k[0, 2] * 2; k[1, 1] /= k[1, 2] * 2; k[0, 2] = .5; k[1, 2] = .5; self.Kn = k   # pose_to_input normalisation

    def wp_rows(self, lo, hi):
        """viewmats (w2c, WorldPlay units), Ks and actions for latents lo..hi-1 from self.mem (verbatim wp_fprime minus controls)."""
        c2w = self.mem[lo:hi].copy(); c2w[:, :3, 3] *= self.unit; w2c = np.linalg.inv(c2w)
        act = [0 if j == 0 else action_label(self.mem[j - 1], self.mem[j]) for j in range(lo, hi)]
        f = lambda x: torch.as_tensor(np.asarray(x)).to(self.wp.device)
        return (f(w2c).float()[None], f(np.repeat(self.Kn[None], hi - lo, 0)).float()[None], torch.as_tensor(act)[None].to(self.wp.device))

    @torch.no_grad()
    def run(self):
        wp, pipe = self.wp, self.wp.pipe; pe, ne = wp.prompt(self.text); frames = []
        torch.manual_seed(self.seed); n_lat = 4 * self.chunks; nf = 4 * (n_lat - 1) + 1
        args = dict(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne, height=H, width=W, num_frames=nf, num_inference_steps=4,
                    guidance_scale=1.0, few_step=True, first_chunk_size=C, return_dict=False, image_path=self.image,
                    use_memory=True, context_window_length=16, generator=torch.Generator(device='cpu').manual_seed(self.seed))
        pipe.ctx = None
        for ci in range(self.chunks):
            lo, hi = 4 * ci, 4 * ci + 4
            if self.arm.get('replan') and ci > 0 and self.measured[lo - 1] is not None:
                delta = self.measured[lo - 1] @ np.linalg.inv(self.cmd[lo - 1])
                for i, j in enumerate(range(lo, hi)):
                    self.cmd[j] = fc.se3_interp(delta, 1. - (i + 1) / C) @ self.plan[j]
                self.events.append(dict(kind='replan', chunk=ci, position_error=float(np.linalg.norm(delta[:3, 3]))))
            self.mem[lo:hi] = self.cmd[lo:hi]
            vm, ks, act = self.wp_rows(lo, hi)
            pipe(**args, chunk_i=ci, viewmats=vm, Ks=ks, action=act, output_type='latent')
            for i in range(4):
                v = pipe.decode_next_latent(output_type='np')                          # [1, f, H, W, 3] in [0, 1]
                frames += [(x * 255).round().clip(0, 255).astype(np.uint8) for x in np.asarray(v)[0]]
            lat = [frames[4 * j] for j in range(1, hi)]                                 # latent j end frame = frame 4j; frames[0] = image
            self.da3.to('cuda'); est = fc.da3_window(self.da3, self.anchor, self.anchor_depth, lat, self.K, max_views=16, tail=4); self.da3.to('cpu'); torch.cuda.empty_cache()
            for j in range(max(1, lo), hi):
                p = est['poses'][j - 1]
                if p is not None and np.isfinite(p).all():
                    self.measured[j] = p
            if self.arm.get('write') == 'measured':
                for j in range(max(1, lo), hi):
                    if self.measured[j] is not None:
                        self.mem[j] = self.measured[j]
                vm2, _, act2 = self.wp_rows(lo, hi)                                      # overwrite PRoPE/retrieval poses + actions
                pipe.ctx['viewmats'][:, lo:hi] = vm2.to(pipe.ctx['viewmats'].dtype); pipe.ctx['action'][:, lo:hi] = act2.to(pipe.ctx['action'].dtype)
            r = [None if self.measured[j] is None else float(np.linalg.norm(self.measured[j][:3, 3] - self.cmd[j][:3, 3])) for j in range(max(1, lo), hi)]
            self.events.append(dict(kind='chunk', chunk=ci, residual=r, da3_scale=est['scale'], peak_GiB=torch.cuda.max_memory_allocated() / 2**30))
        return frames, dict(planned=self.plan.tolist(), commanded=self.cmd.tolist(), unit=self.unit, prompt=self.text, pose_frame_ids=[4 * j for j in range(n_lat)],
                            measured=[None if m is None else m.tolist() for m in self.measured], events=self.events)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--arm', nargs='*', default=None); p.add_argument('--scenes', nargs='+', default=None); p.add_argument('--seeds', nargs='+', type=int, default=[7])
    p.add_argument('--chunks', type=int, default=11); p.add_argument('--unit', type=float, default=1.0)
    p.add_argument('--min_free_gib', type=float, default=10.); a = p.parse_args()
    assert 16 * a.chunks - 3 >= fc.N_FRAMES + 1, 'need >= 161 frames (0..160)'
    arms = [x for x in json.load(open(a.arms)) if not a.arm or x['name'] in a.arm]; a.out.mkdir(parents=True, exist_ok=True)
    scenes = a.scenes or [s['name'] for s in fc.index()['scenes']]
    wp = WorldPlay(); da3 = load_da3('cpu')
    for arm in arms:
        for scene in scenes:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.mp4'
                if dst.exists():
                    continue
                if fc.free_gib() < a.min_free_gib:
                    print(json.dumps(dict(stop='disk floor', free_gib=round(fc.free_gib(), 2))), flush=True); sys.exit(3)
                t0 = time.time(); torch.cuda.reset_peak_memory_stats()
                r = Roller(wp, da3, scene, arm, seed, a.chunks, a.unit); frames, log = r.run()
                log = dict(model='worldplay', arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                           settings=dict(steps=4, guidance=1.0, few_step=True, use_memory=True, context_window_length=16, chunks=a.chunks, hw=[H, W],
                                         generated_frames=len(frames) - 1, kept_frames=fc.N_FRAMES), **log)
                fc.save_run(dst, frames[1:], r.anchor, r.K, log, preprocess='worldplay')
                res = [x for e in log['events'] if e['kind'] == 'chunk' for x in e['residual'] if x is not None]
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(log['seconds'], 1), frames=len(frames) - 1,
                                      residual_end=round(res[-1], 3) if res else None, peak=round(log['peak_GiB'], 1))), flush=True)


if __name__ == '__main__':
    main()
