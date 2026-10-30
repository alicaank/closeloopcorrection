"""F' on released HY-WorldPlay-5B (Wan2.2-5B, 4-step distilled; baselines/venv_worldplay; launch with torchrun --nproc_per_node=1).

Same protocol as memorizon_plugins/gv_memorizon2.py on the dev scenes (batch_dev10):
  * planned path = recording, per latent j (frame 4j), anchor-depth gauge, OpenCV c2w; WorldPlay chunk c (c >= 1) = latents
    4c..4c+3; chunk 0 = image latent + latents 1..3. 10 chunks = latents 0..39 (frames 0..156).
  * after every chunk the latent-end frames are measured command-free with DA3 (da3_window, anchor + <=16 views, tail 4,
    scale from the stored anchor depth) -> measured c2w of the chunk's latents.
  * arm keys: write='measured' -> the chunk's viewmats (PRoPE camera of context tokens AND FOV memory retrieval) and its
    discrete actions are overwritten with the measured poses; replan=true -> the next chunk's command is re-planned from
    the measured pose of the last latent and rejoins the plan by SE(3) interpolation (as F'); cmd_gain=k -> commanded
    translations x k (control).
  * --unit: WorldPlay translation units per gauge unit (default 1.0; calibrate on the call-1 executed ratio).
WorldPlay settings kept from wan/generate.py: guidance 1, few-step (4), use_memory, context_window_length 16, first_chunk_size 4,
544x960 (704x1280 OOMs on 32 GB at chunk 5). Local repo patches (sota_repos/worldplay, git diff): SDPA fallback for
sageattention; tokens/latent from resolution.
Output: <out>/<arm>/<scene>_s<seed>.npz (frames 1..156 at 448x256) + .json (planned, commanded, measured, events).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
WP = '@WORKSPACE@/sota_repos/worldplay'; sys.path[:0] = [WP, WP + '/wan']
MP = '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins'; sys.path[:0] = [MP]
from wan.inference.pipeline_wan_w_mem_relative_rope import WanPipeline
from wan.models.dits.arwan_w_action_w_mem_relative_rope import WanTransformer3DModel
from wan.distributed import maybe_init_distributed_environment_and_model_parallel
from wan.models.par_vae.tools import DistController
from wan.models.par_vae.context_parallel.wrapper_vae import DistWrapper
from wan.inference.helper import MyVAE
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation
from realized_pose import load_da3
from gv_memorizon2 import da3_window
from gv_memorizon import se3_interp
BATCH = Path('@WORKSPACE@/analysis/plan_base_restart_20261008/lyra2_plugins/batch_dev10')
COMP = Path('@WORKSPACE@/training_runs/base_restart_20261008/dataset_v3/components')
MODEL_ID = '@WORKSPACE@/baselines/models/Wan2.2-TI2V-5B-Diffusers'
WDIR = '@WORKSPACE@/baselines/models/worldplay5b'
H, W, C = 544, 960, 4
MAP = {(0, 0, 0, 0): 0, (1, 0, 0, 0): 1, (0, 1, 0, 0): 2, (0, 0, 1, 0): 3, (0, 0, 0, 1): 4,
       (1, 0, 1, 0): 5, (1, 0, 0, 1): 6, (0, 1, 1, 0): 7, (0, 1, 0, 1): 8}


def action_label(prev_c2w, c2w):
    """hyvideo.generate.pose_to_input action rule for one latent step: translation class * 9 + rotation class."""
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


def small(f):
    return np.asarray(Image.fromarray(f).resize((448, 256), Image.BICUBIC))


class WorldPlay:
    def __init__(self):
        os.environ.setdefault('RANK', '0'); os.environ.setdefault('LOCAL_RANK', '0'); os.environ.setdefault('WORLD_SIZE', '1')
        torch.cuda.set_device(0); self.device = torch.device('cuda:0')
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        maybe_init_distributed_environment_and_model_parallel(1, sp_size=1, distributed_init_method='env://')
        self.vae = MyVAE.from_pretrained(MODEL_ID, subfolder='vae', torch_dtype=torch.bfloat16).eval().requires_grad_(False)
        self.pipe = WanPipeline.from_pretrained(MODEL_ID, vae=self.vae, transformer=None, torch_dtype=torch.bfloat16)   # transformer replaced below
        self.pipe.dist_vae = DistWrapper(self.vae, DistController(0, 0, 1, None), None, 4)
        cfg = WanTransformer3DModel.load_config(WDIR + '/wan_transformer')
        tr = WanTransformer3DModel.from_config(cfg).to(torch.bfloat16); tr.add_discrete_action_parameters()   # real init: keeps non-persistent rope buffers
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
    def __init__(self, wp, da3, scene, arm, seed, chunks, unit):
        self.wp, self.da3, self.arm, self.seed, self.chunks, self.unit = wp, da3, arm, seed, chunks, unit
        z = np.load(BATCH / 'trajectories' / f'{scene}.npz')
        self.plan = np.linalg.inv(z['w2c'].astype(np.float64))[0:161:4]               # c2w per latent j (frame 4j), gauge
        self.K = z['intrinsics'][0].astype(np.float64)
        if arm.get('synthetic'):                                    # {'dir': [x, y, z], 'speed': units/latent}: straight line, no rotation
            v = np.asarray(arm['synthetic']['dir'], float); v = v / np.linalg.norm(v) * float(arm['synthetic']['speed'])
            self.plan = np.tile(np.eye(4), (41, 1, 1)); self.plan[:, :3, 3] = np.arange(41)[:, None] * v[None]
        idx = json.load(open(BATCH / 'index.json')); sid = [s for s in idx['scenes'] if s['name'] == scene][0]['stream_id']
        self.text = idx['prompt']; self.anchor_depth = np.load(COMP / sid / 'anchor.npz')['depth']
        self.image = str(BATCH / 'images' / f'{scene}.png'); self.anchor = np.asarray(Image.open(self.image).convert('RGB'))
        self.cmd = self.plan.copy()
        if arm.get('cmd_gain'):
            self.cmd[:, :3, 3] *= float(arm['cmd_gain'])
        self.mem = self.cmd.copy(); self.measured = [None] * 41; self.measured[0] = np.eye(4); self.events = []
        # WorldPlay normalized intrinsics (pose_to_input: fx/(2cx), fy/(2cy), principal point at the centre)
        k = self.K.copy(); k[0, 0] /= k[0, 2] * 2; k[1, 1] /= k[1, 2] * 2; k[0, 2] = .5; k[1, 2] = .5; self.Kn = k

    def wp_rows(self, lo, hi):
        """viewmats (w2c, WorldPlay units), Ks and actions for latents lo..hi-1 from self.mem."""
        c2w = self.mem[lo:hi].copy(); c2w[:, :3, 3] *= self.unit; w2c = np.linalg.inv(c2w)
        act = [0 if j == 0 else action_label(self.mem[j - 1], self.mem[j]) for j in range(lo, hi)]
        # carrier ablations (speed-limit diagnosis): action_zero -> every discrete action = class 0 (no move, no turn), PRoPE
        # poses intact; prope_static -> every camera (PRoPE + FOV retrieval) = the anchor pose, discrete actions intact.
        if self.arm.get('action_zero'):
            act = [0] * (hi - lo)
        if self.arm.get('prope_static'):
            w2c = np.repeat(np.eye(4)[None], hi - lo, 0)
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
                    self.cmd[j] = se3_interp(delta, 1. - (i + 1) / C) @ self.plan[j]
                self.events.append(dict(kind='replan', chunk=ci, position_error=float(np.linalg.norm(delta[:3, 3]))))
            self.mem[lo:hi] = self.cmd[lo:hi]
            vm, ks, act = self.wp_rows(lo, hi)
            pipe(**args, chunk_i=ci, viewmats=vm, Ks=ks, action=act, output_type='latent')
            for i in range(4):
                v = pipe.decode_next_latent(output_type='np')                          # [1, f, H, W, 3] in [0, 1]
                frames += [small((x * 255).round().clip(0, 255).astype(np.uint8)) for x in np.asarray(v)[0]]
            # measure latents 1..hi-1 (latent j end frame = frame 4j); frames[0] = image
            lat = [frames[4 * j] for j in range(1, hi)]
            self.da3.to('cuda'); est = da3_window(self.da3, self.anchor, self.anchor_depth, lat, self.K, max_views=16, tail=4); self.da3.to('cpu'); torch.cuda.empty_cache()
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
            self.events.append(dict(kind='chunk', chunk=ci, residual=r, peak_GiB=torch.cuda.max_memory_allocated() / 2**30))
        return np.stack(frames[1:]), dict(planned=self.plan[:n_lat].tolist(), commanded=self.cmd[:n_lat].tolist(), unit=self.unit,
                                          measured=[None if m is None else m.tolist() for m in self.measured[:n_lat]], events=self.events)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=['1740e916', '38dcc531']); p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19])
    p.add_argument('--chunks', type=int, default=10); p.add_argument('--unit', type=float, default=1.0); a = p.parse_args()
    arms = json.load(open(a.arms)); a.out.mkdir(parents=True, exist_ok=True)
    wp = WorldPlay(); da3 = load_da3('cpu')
    for arm in arms:
        for scene in a.scenes:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.npz'
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
                frames, log = Roller(wp, da3, scene, arm, seed, a.chunks, a.unit).run()
                np.savez_compressed(dst, frames=frames)
                json.dump(dict(arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, **log), open(dst.with_suffix('.json'), 'w'))
                res = [x for e in log['events'] if e['kind'] == 'chunk' for x in e['residual'] if x is not None]
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(time.time() - t0, 1), frames=len(frames),
                                      residual_end=round(res[-1], 3) if res else None, peak=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)


if __name__ == '__main__':
    main()
