"""GaussVerse ideas plugged into released Memorizon (Wan2.2-TI2V-5B, native latent bank + PRoPE camera branch, 4-step distilled).

Same rollout as memorizon.pipeline.MemorizonPipeline.rollout (10 chunks x 4 latents = 160 frames + anchor), plus hooks:
  write   : 'commanded' (stock) | 'measured'  -- after each chunk the cameras stored for its latents (used later for
            bank retrieval, attention mask and PRoPE of bank/recent entries) are replaced by the realized pose measured
            on the generated frame (fixed-anchor-depth SIFT PnP, our anchor-depth gauge, never the command).
  replan  : F' -- the next chunk's commands start at the measured last pose and rejoin the plan by SE(3) interpolation
            over that chunk (last latent of the next chunk = planned pose exactly).
  gate    : F' uncertainty gate -- latents whose |measured - commanded| position residual > tau (anchor units) are
            excluded from bank candidates (never written).
  bank    : 'full' | 'drop' (memory_size 0) | 'appearance' (bank tokens removed from the PRoPE camera branch only; main
            attention still reads them = our APPEARANCE mode) | 'wrongcam' (bank cameras perturbed by a fixed SE(3)
            offset: correct appearance, wrong placement).
Measurement runs in every arm (passive in 'commanded'), so residuals are logged everywhere. A PnP failure = no
measurement for that latent (it keeps its commanded camera; replanning skips the correction); failures are counted.
Outputs per (arm, scene, seed): frames npz at 448x256 (frames 1..160), events json.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'fair_common_20261025'))
from causal_units import metric_prediction_scale

import numpy as np, torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('@WORKSPACE@/baselines'); sys.path.insert(0, str(ROOT / 'memorizon'))
from memorizon import MemorizonPipeline
from memorizon.model import MemorizonModel
from memorizon.retrieval import Frustum, select_bank
from memorizon.layout import Layout, rope_index, attention_mask
from memorizon.camera import camera_transforms
from memorizon.wan.transformer import frame_mask_to_block_mask
sys.path.insert(0, str(Path(__file__).parent))
from realized_pose import load_da3, realized_poses   # command-free DA3 multi-view, anchor-depth gauge (validated on recorded frames)

BATCH = Path('@WORKSPACE@/analysis/plan_base_restart_20261008/lyra2_plugins/batch_dev10')
COMP = Path('@WORKSPACE@/training_runs/base_restart_20261008/dataset_v3/components')
W, H, TRAJ_SCALE = 864, 480, 4.0      # memorizon units = our anchor-depth gauge / 4 (as in baselines/run_memorizon.py)


def se3_interp(delta, w):
    rot = Rotation.from_matrix(delta[:3, :3])
    r = Slerp([0., 1.], Rotation.concatenate([Rotation.identity(), rot]))(w).as_matrix()
    out = np.eye(4); out[:3, :3] = r; out[:3, 3] = w * delta[:3, 3]
    return out


def wrong_offset(deg=15., shift=0.3):
    m = np.eye(4); m[:3, :3] = Rotation.from_euler('y', deg, degrees=True).as_matrix(); m[0, 3] = shift
    return m


class StreamDecoder:
    """Causal Wan VAE decode, one latent at a time, persistent cache across chunks (same math as pipeline.decode)."""
    def __init__(self, pipe):
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
        self.pipe, self.unpatchify, self.i = pipe, unpatchify, 0
        pipe.vae.clear_cache()

    @torch.no_grad()
    def __call__(self, latents):     # [1,48,n,h,w] new latents only
        vae, out = self.pipe.vae, []
        z = vae.post_quant_conv(latents.to(self.pipe.device, torch.float32) * self.pipe.lat_std + self.pipe.lat_mean)
        patch = getattr(vae.config, 'patch_size', None)
        for k in range(z.shape[2]):
            vae._conv_idx = [0]
            o = vae.decoder(z[:, :, k:k + 1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=(self.i == 0)).to('cpu')
            if patch is not None:
                o = self.unpatchify(o, patch_size=patch)
            v = o[0].permute(1, 2, 3, 0).clamp(-1, 1)
            out.append(((v + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()); self.i += 1
        return np.concatenate(out, 0)


def small(f):
    return np.asarray(Image.fromarray(f).resize((448, 256), Image.BICUBIC))


@torch.no_grad()
def rollout(pipe, da3, scene, arm, seed, num_chunks=10, memory_size=6, num_steps=4):
    z = np.load(BATCH / 'trajectories' / f'{scene}.npz')
    plan = np.linalg.inv(z['w2c'].astype(np.float64))[0:161:4]          # planned c2w per latent, our gauge
    K448 = z['intrinsics'][0].astype(np.float64)
    stream = json.load(open(BATCH / 'index.json'))
    sid = [s for s in stream['scenes'] if s['name'] == scene][0]['stream_id']
    anchor_depth = np.load(COMP / sid / 'anchor.npz')['depth']
    anchor448 = np.asarray(Image.open(BATCH / 'images' / f'{scene}.png').convert('RGB'))
    intr = (K448[0, 0] * W / 448, K448[1, 1] * H / 256, K448[0, 2] * W / 448, K448[1, 2] * H / 256)
    # Memorizon was trained on metric poses (metres / 4). Our gauge unit = anchor median depth, so convert once per
    # scene with DA3's metric anchor depth (single image, no future RGB, no command): metres = gauge * metric_depth.
    torch.cuda.empty_cache(); da3.to('cuda')
    pm = da3.inference(image=[anchor448], intrinsics=K448[None].astype(np.float32), infer_gs=False, process_res=504, export_dir=None)
    da3.to('cpu'); torch.cuda.empty_cache()
    unit, unit_calibration = metric_prediction_scale(da3, pm, anchor448, anchor_depth, K448, divisor=TRAJ_SCALE)
    metric = unit_calibration["metres_per_reference_unit"]

    cmd = plan.copy()                      # what is commanded (replanning edits future rows)
    mem = plan.copy()                      # what is stored for written latents (measured or commanded)
    measured = [None] * len(plan); residual = [None] * len(plan); excluded = set()
    events = []
    dit, C = pipe.model.model, pipe.model.config.chunk_size
    frustum = Frustum.from_intrinsics(list(intr))
    first = pipe.encode_image(Image.fromarray(anchor448).resize((W, H), Image.BICUBIC))
    text = pipe.encode_prompt(stream['prompt']).to(pipe.device, pipe.dtype)
    _, ch, _, h, w = first.shape; _, p_h, p_w = dit.config.patch_size; tpf = (h // p_h) * (w // p_w)
    sched = pipe.model.noise_scheduler; sched.set_timesteps(num_steps); timesteps = sched.timesteps.to(pipe.device)
    intr_b = torch.as_tensor(np.asarray(intr, dtype=np.float32))[None].to(pipe.device)
    end = 1 + num_chunks * C
    gen = torch.zeros((1, ch, end, h, w), device=pipe.device, dtype=pipe.dtype); gen[:, :, 0] = first[:, :, 0]
    rng = torch.Generator(device='cpu').manual_seed(seed)
    dec = StreamDecoder(pipe); frames = [small(f) for f in dec(gen[:, :, :1])]
    k_mem = 0 if arm.get('bank') == 'drop' else memory_size
    for ci in range(num_chunks):
        q_lo, q_hi = 1 + ci * C, 1 + (ci + 1) * C
        # ---- F' replanning: start this chunk from the measured last pose, rejoin the plan at its last latent
        if arm.get('replan') and ci > 0 and measured[q_lo - 1] is not None:
            delta = measured[q_lo - 1] @ np.linalg.inv(cmd[q_lo - 1])
            for i, j in enumerate(range(q_lo, q_hi)):
                cmd[j] = se3_interp(delta, 1. - (i + 1) / C) @ plan[j]
            events.append(dict(kind='replan', chunk=ci, position_error=float(np.linalg.norm(delta[:3, 3])),
                               rotation_error_deg=float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))))
        mem[q_lo:q_hi] = cmd[q_lo:q_hi]
        n_rec = min(C, q_lo); recent = np.arange(q_lo - n_rec, q_lo)
        cand = np.array([i for i in range(1, max(1, q_lo - n_rec)) if i not in excluded], dtype=int)
        cams_np = mem.copy()
        cams_np[:, :3, 3] *= unit
        bank = select_bank(cams_np, cand, cams_np[q_hi - 1][None], frustum, k=k_mem, q0=q_lo, chunk=C, cap=max(k_mem, 1)) if k_mem else np.zeros(0, int)
        lay = Layout(n_bank=len(bank), n_recent=n_rec, chunk=C, n_query_chunks=1)
        F_, n_cond = lay.total, lay.n_cond
        rope = torch.as_tensor(rope_index(lay), dtype=torch.long, device=pipe.device)
        mask = (attention_mask(lay, bank, q_lo, cams_np[q_lo:q_hi], cams_np[bank], frustum=frustum, k=memory_size)
                if len(bank) else attention_mask(lay))
        mask = torch.as_tensor(mask, dtype=torch.bool, device=pipe.device)
        ctx_mask = frame_mask_to_block_mask(mask[:n_cond, :n_cond], tpf)
        idx = np.concatenate([[0], bank, recent]).astype(int)
        ctx_cams = cams_np[idx].copy()
        if arm.get('bank') == 'wrongcam' and len(bank):
            off = wrong_offset(); off[:3, 3] *= unit
            for b in range(1, 1 + len(bank)):
                ctx_cams[b] = off @ ctx_cams[b]
        cams = torch.as_tensor(np.concatenate([ctx_cams, cams_np[q_lo:q_hi]]), dtype=torch.float32, device=pipe.device)[None]
        x = torch.randn((1, ch, F_, h, w), generator=rng).to(pipe.device, pipe.dtype)
        cond = gen[:, :, idx]; x[:, :, :n_cond] = cond
        cam_ctx = camera_transforms(dit, cams[:, :n_cond], intr_b); cam_q = camera_transforms(dit, cams[:, n_cond:], intr_b)
        kv = dit.init_kv_cache(1, F_, h, w)
        with torch.autocast('cuda', torch.bfloat16):
            dit(cond, torch.zeros((1, n_cond), device=pipe.device), text, rope_index=rope[:n_cond], attention_mask=ctx_mask,
                camera=cam_ctx, kv_cache=kv, kv_cache_range=(0, n_cond))
            for layer in kv:
                layer.pop('cross_attn_key', None); layer.pop('cross_attn_value', None)
                if arm.get('bank') == 'appearance' and len(bank) and 'prope_key' in layer:
                    keep = torch.cat([torch.arange(0, tpf), torch.arange((1 + len(bank)) * tpf, n_cond * tpf)]).to(pipe.device)
                    layer['prope_key'] = layer['prope_key'][:, :, keep]; layer['prope_value'] = layer['prope_value'][:, :, keep]
            for t in timesteps:
                ts = t.repeat(1, C).clone()
                pred = dit(x[:, :, n_cond:], ts, text, rope_index=rope, camera=cam_q, kv_cache=kv, kv_cache_range=(0, F_))
                x[:, :, n_cond:] = sched.step_diff_noise_level(pred, ts, x[:, :, n_cond:]).to(x)
        gen[:, :, q_lo:q_hi] = x[0:1, :, n_cond:]
        del kv, x, cond, cam_ctx, cam_q; torch.cuda.empty_cache()
        new = [small(f) for f in dec(gen[:, :, q_lo:q_hi])]; frames += new
        # ---- measure realized pose of each new latent (its last frame), write / gate
        lat_frames = [frames[4 * j] for j in range(1, q_hi)]          # latent j <-> frame 4j (its last RGB frame)
        torch.cuda.empty_cache(); da3.to('cuda')
        poses, est = realized_poses(da3, anchor448, anchor_depth, lat_frames, K448)
        da3.to('cpu'); torch.cuda.empty_cache()
        for i, j in enumerate(range(q_lo, q_hi)):
            pose = poses[j - 1]
            if pose is not None and np.isfinite(pose).all():
                measured[j] = pose; residual[j] = float(np.linalg.norm(measured[j][:3, 3] - cmd[j][:3, 3]))
                if arm.get('write') == 'measured':
                    mem[j] = measured[j]
                if arm.get('gate') is not None and residual[j] > float(arm['gate']):
                    excluded.add(j)
        events.append(dict(kind='chunk', chunk=ci, bank=[int(b) for b in bank], estimator=est,
                           measured_valid=[measured[j] is not None for j in range(q_lo, q_hi)],
                           residual=[residual[j] for j in range(q_lo, q_hi)], excluded=sorted(int(e) for e in excluded)))
    frames = np.stack(frames)
    return frames, dict(unit_calibration=unit_calibration, metric_anchor_depth_m=metric, events=events, commanded=cmd.tolist(), planned=plan.tolist(),
                        measured=[None if m is None else m.tolist() for m in measured])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', type=Path, required=True); p.add_argument('--scenes', nargs='+', default=['1740e916', '38dcc531'])
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 11, 19]); a = p.parse_args()
    arms = json.load(open(a.arms)); a.out.mkdir(parents=True, exist_ok=True)
    CK = ROOT / 'models/memorizon_local'
    pipe = MemorizonPipeline(MemorizonModel.from_pretrained(str(CK))); da3 = load_da3('cpu')
    for arm in arms:
        for scene in a.scenes:
            for seed in a.seeds:
                dst = a.out / arm['name'] / f'{scene}_s{seed}.npz'
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); t0 = time.time()
                frames, log = rollout(pipe, da3, scene, arm, seed)
                np.savez_compressed(dst, frames=frames[1:161])
                json.dump(dict(arm=arm, scene=scene, seed=seed, seconds=time.time() - t0, **log), open(dst.with_suffix('.json'), 'w'))
                res = [r for e in log['events'] if e['kind'] == 'chunk' for r in e['residual'] if r is not None]
                nval = sum(m is not None for m in log['measured'][1:])
                print(json.dumps(dict(arm=arm['name'], scene=scene, seed=seed, sec=round(time.time() - t0, 1), pnp_valid=f'{nval}/40',
                                      residual_median=None if not res else round(float(np.median(res)), 3))), flush=True)


if __name__ == '__main__':
    main()
