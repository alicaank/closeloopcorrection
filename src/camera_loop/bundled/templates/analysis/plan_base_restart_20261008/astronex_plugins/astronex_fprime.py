"""F' (closed-loop camera control) on Astronex-World 1.0, causal student, consumer config (inference/causal_consumer.yaml,
23.2 GB peak). Same 10 dev videos and recorded paths as our 864 runs (inputs/ from export_dev10.py, anchor 832x480 crop).

Astronex takes one OpenCV w2c + normalised K per latent frame (PRoPE in all 30 layers; 0.08 units = one 'w' step) and
generates 8-latent blocks (block 1 = reference + 7). Here: 1 + 31 latents = 4 blocks = 125 RGB frames; camera of latent j =
recorded pose of RGB frame 4j, translation x --scale (anchor-depth units -> Astronex units, calibrated with --arm open).

Hook: inference/pipelines/causal_inference.py iterates its blocks with `tqdm.tqdm(all_num_frames)`; that call is replaced
by a generator that runs between blocks with the sampler's own locals (output latents, viewmats, current_start_frame):
  * the finished block is stream-decoded with the VAE's causal cache (WAN_SKIP_DECODE=1 skips the end-of-run decode);
  * measure: DA3 on [anchor + every latent end so far] (realized_poses, as closed_loop_864) -> anchor units;
  * write (arms write/both): viewmats of the finished block's latents := measured poses (history labels, PRoPE re-applies
    them from viewmats every block; no K/V re-forward);
  * replan (arms replan/both): next block's commands = plan with the measured start offset removed over the block
    (offset-preserving, as the Memorizon/VMem/WorldPlay harnesses: exact start -> plan unchanged).
Output per video: <out>/<sid>.npz (RGB frames 0,2,4,... uint8) + <sid>.json (planned / commanded / measured per latent).
Usage (venv_astronex): astronex_fprime.py --arm open|measure|write|replan|both --scale S --out DIR [--videos ...]"""
import argparse, json, os, runpy, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

REPO = Path('@WORKSPACE@/sota_repos/astronex_world'); WEIGHTS = Path('@WORKSPACE@/sota_repos/Astronex')
HERE = Path(__file__).resolve().parent; NLAT = 32; BLOCK = 8
ARMS = {'open': dict(measure=0, write=0, replan=0), 'measure': dict(measure=1, write=0, replan=0), 'write': dict(measure=1, write=1, replan=0),
        'replan': dict(measure=1, write=0, replan=1), 'both': dict(measure=1, write=1, replan=1)}
CTX = dict(videos=[], i=-1, arm=None, scale=1., da3=None, out=None)


def se3_interp(delta, w):
    r = Slerp([0., 1.], Rotation.concatenate([Rotation.identity(), Rotation.from_matrix(delta[:3, :3])]))(w).as_matrix()
    out = np.eye(4); out[:3, :3] = r; out[:3, 3] = w * delta[:3, 3]
    return out


def w2c_astro(c2w):
    c = np.asarray(c2w, np.float64).copy(); c[:3, 3] *= CTX['scale']
    return np.linalg.inv(c)


def block_done(L, v, st, final=False):
    pipe, out, vm = L['self'], L['output'], L['viewmats']
    cur = int(L.get('cache_start_frame', L['current_start_frame']))           # output index of the next block
    if int(L['current_start_frame']) != cur:
        raise RuntimeError('camera and latent indices disagree')
    t0 = time.perf_counter()
    if st['prev'] == 0:
        pipe.vae.reset_stream()
    gdev = next(pipe.generator.parameters()).device                                     # 10-23 smokes: even one latent's decode needs ~10 GB next to the
    pipe.generator.to('cpu'); torch.cuda.empty_cache()                                   # two CFG KV caches -> park the 5B denoiser on the CPU meanwhile
    for j in range(st['prev'], cur):                                                    # stream latent by latent (same causal cache, same result)
        px = pipe.vae.decode_to_pixel(out[:, j:j + 1], use_cache=True)
        st['frames'].append(((px[0].permute(0, 2, 3, 1).float() * .5 + .5).clamp(0, 1) * 255).round().byte().cpu().numpy()); del px
    torch.cuda.empty_cache()
    st['t_decode'] += time.perf_counter() - t0; done0 = st['prev']; st['prev'] = cur
    gen = np.concatenate(st['frames']); arm = ARMS[CTX['arm']]
    try:
        _measure_and_act(L, v, st, gen, arm, vm, cur, done0, final)
    finally:
        pipe.generator.to(gdev); torch.cuda.empty_cache()


def _measure_and_act(L, v, st, gen, arm, vm, cur, done0, final):
    if arm['measure']:
        t1 = time.perf_counter(); lat = [gen[4 * j] for j in range(1, cur)]
        from realized_pose import realized_poses
        CTX['da3'].to('cuda'); ps, _ = realized_poses(CTX['da3'], gen[0], None, lat, v['K'], max_views=16, tail=min(BLOCK, len(lat))); CTX['da3'].to('cpu')
        torch.cuda.empty_cache(); st['t_measure'] += time.perf_counter() - t1
        for j in range(max(1, done0), cur):
            p = ps[j - 1]
            if p is not None and np.isfinite(np.asarray(p)).all():
                q = np.asarray(p, np.float64).copy(); q[:3, 3] /= v['mpu']; v['measured'][j] = q
                if arm['write']:
                    vm[0, j] = torch.from_numpy(w2c_astro(q)).to(vm)
        if arm['replan'] and not final and v['measured'].get(cur - 1) is not None:
            delta = v['measured'][cur - 1] @ np.linalg.inv(v['cmd'][cur - 1]); n = min(BLOCK, NLAT - cur)
            for i, j in enumerate(range(cur, cur + n)):
                v['cmd'][j] = se3_interp(delta, 1. - (i + 1) / n) @ v['plan'][j]; vm[0, j] = torch.from_numpy(w2c_astro(v['cmd'][j])).to(vm)
            v['events'].append(dict(block_end=cur, position_offset=float(np.linalg.norm(delta[:3, 3]))))
    if final:
        sid = v['sid']; np.savez_compressed(CTX['out'] / f'{sid}.npz', frames=gen[::2])
        log = dict(arm=CTX['arm'], scale=CTX['scale'], sid=sid, n_rgb=int(len(gen)), planned=[p.tolist() for p in v['plan']],
                   commanded=[c.tolist() for c in v['cmd']], measured={str(k): m.tolist() for k, m in v['measured'].items()}, events=v['events'],
                   seconds=round(time.time() - st['t_start'], 1), decode_s=round(st['t_decode'], 1), measure_s=round(st['t_measure'], 1),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        json.dump(log, open(CTX['out'] / f'{sid}.json', 'w'))
        print(json.dumps(dict(sid=sid, arm=CTX['arm'], frames=len(gen), sec=log['seconds'], peak=log['peak_vram_gb'], events=len(v['events']))), flush=True)


def hook(L, final=False):
    """Called by the patched sampler at the start of every block after the first, and once after the last block."""
    key = id(L['output'])
    if CTX.get('key') != key:                                                           # a new video (new inference call)
        CTX['key'] = key; CTX['i'] += 1; torch.cuda.reset_peak_memory_stats()
        CTX['st'] = dict(prev=0, frames=[], t_decode=0., t_measure=0., t_start=time.time())
    block_done(L, CTX['videos'][CTX['i']], CTX['st'], final)


def patch_sampler():
    """causal_consumer.yaml has no denoising_step_list, so sample.py builds CausalDiffusionInferencePipeline (UniPC, 8 steps).
    A copy of its module with two hook calls (block start, after the loop) replaces it; the repo file is untouched."""
    import types, importlib
    src = (REPO / 'inference/pipelines/causal_diffusion_inference.py').read_text()
    a = "        for current_num_frames in all_num_frames:\n            apply_event(current_start_frame)\n"
    b = "        for _bi, current_num_frames in enumerate(all_num_frames):\n            if _bi:\n                __import__('astronex_hook').hook(locals())\n            apply_event(current_start_frame)\n"
    c = "        # Step 4: Decode the output\n"
    if src.count(a) != 1 or src.count(c) != 1:
        raise RuntimeError('sampler source changed: hook points not found')
    src = src.replace(a, b).replace(c, "        __import__('astronex_hook').hook(locals(), final=True)\n" + c)
    name = 'inference.pipelines.causal_diffusion_inference'; mod = types.ModuleType(name); mod.__file__ = str(REPO / 'inference/pipelines/causal_diffusion_inference.py')
    mod.__package__ = 'inference.pipelines'; exec(compile(src, mod.__file__, 'exec'), mod.__dict__); sys.modules[name] = mod
    pk = importlib.import_module('inference.pipelines'); pk.CausalDiffusionInferencePipeline = mod.CausalDiffusionInferencePipeline
    sys.modules['astronex_hook'] = types.SimpleNamespace(hook=hook)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True, choices=list(ARMS)); ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--videos', nargs='*', default=None); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--traj', default=None, help="diagnostic: Astronex's own trajectory string (e.g. 'w*31') instead of the recorded path; plan = its c2w / scale")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); CTX.update(arm=a.arm, scale=a.scale, out=a.out)
    vids = a.videos or sorted(p.stem for p in (HERE / 'inputs').glob('*.npz')); rows = []; work = a.out / '_inputs'; work.mkdir(exist_ok=True)
    for sid in vids:
        if (a.out / f'{sid}.json').exists():
            continue
        z = np.load(HERE / 'inputs' / f'{sid}.npz'); P = z['c2w']; K = z['K']; plan = [P[4 * j].copy() for j in range(NLAT)]
        if a.traj:                                                                       # 10-23 diagnostic: native Astronex motion
            sys.path.insert(0, str(REPO)); from utils.camera_trajectory import parse_trajectory
            C = np.linalg.inv(parse_trajectory(a.traj).astype(np.float64))[:NLAT]; plan = []
            for c in C:
                c = c.copy(); c[:3, 3] /= a.scale; plan.append(c)
        Image.fromarray(z['anchor']).save(work / f'{sid}.png')
        Kn = np.array([[K[0, 0] / 832, 0, K[0, 2] / 832], [0, K[1, 1] / 480, K[1, 2] / 480], [0, 0, 1]], np.float32)
        np.savez(work / f'{sid}_cam.npz', viewmats=np.stack([w2c_astro(c) for c in plan]).astype(np.float32), Ks=np.stack([Kn] * NLAT))
        rows.append(dict(prompt=str(z['prompt']), image_path=str(work / f'{sid}.png'), camera_path=str(work / f'{sid}_cam.npz'), output_name=f'{sid}.mp4', num_output_frames=NLAT - 1))
        CTX['videos'].append(dict(sid=sid, plan=plan, cmd=[c.copy() for c in plan], measured={}, events=[], K=K, mpu=float(z['mpu'])))
    if not rows:
        print('nothing to do'); return
    (work / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    import astronex_env
    cfg = astronex_env.resolve_config(str(REPO / 'inference/causal_consumer.yaml'), str(work / '_cfg.yaml'), weights_dir=str(WEIGHTS))
    os.environ.update(ASTRONEX_WEIGHTS=str(WEIGHTS), WAN_SKIP_DECODE='1')
    if ARMS[a.arm]['measure']:
        from realized_pose import load_da3
        CTX['da3'] = load_da3('cpu')
    patch_sampler()
    sys.argv = ['inference.sample', '--config_path', cfg, '--checkpoint_path', str(WEIGHTS), '--data_path', str(work / 'manifest.jsonl'),
                '--wbench_manifest', str(work / 'manifest.jsonl'), '--output_folder', str(work / 'sample_out'), '--num_output_frames', str(NLAT - 1),
                '--seed', str(a.seed), '--i2v']
    runpy.run_module('inference.sample', run_name='__main__')


if __name__ == '__main__':
    main()
