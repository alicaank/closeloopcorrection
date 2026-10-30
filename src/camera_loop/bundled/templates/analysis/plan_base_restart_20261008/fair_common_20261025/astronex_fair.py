"""Fair-inference copy of astronex_plugins/astronex_fprime.py (Astronex-World 1.0 causal student, inference/causal_consumer.yaml)
on batch_ours10. venv_astronex; env as astronex_plugins/queue_astronex.sh (PYTHONPATH with the Astronex repo).

Changes vs astronex_fprime.py:
  * image: the dataset's 864x480 frame 0 handed to sample.py, whose own I2V transform (sample.py:538-551: torchvision
    Resize((480, 832)), an anisotropic stretch) prepares it, instead of our pre-cropped 832x480 centre crop; intrinsics = our
    true K through that stretch (fair_common.k_astronex), normalised by 832 / 480 as before;
  * length: 48 latents (6 blocks of 8; generate.py snaps 1 + frames to a block multiple) = frames 0..188, frames 1..160 kept
    (40 latents would give 156); latents 41..47 take the same recorded path (batch w2c_overrun), never scored;
  * all frames kept (before: every 2nd, npz) at the native 832x480 as a crf-12 mp4 with frame 0 = the processed input frame,
    + json with per-frame poses and the output K (fair_common.save_run);
  * arms named as the other harnesses (arms_4.json: baseline / write_measured / replan_commanded / replan_measured =
    measure / write / replan / both before); output <out>/<arm>/<name>_s<seed>.mp4 + .json;
  * the latent-only dump of sample.py:921-925 (WAN_SKIP_DECODE, ~7 MB per video) is not written (disk floor; frames are
    stream-decoded in the hook as before); disk floor check before each video.
Unchanged: causal consumer config (UniPC 8 steps, CFG 3.0, window 20 + sink 4, ref_cond_scale 1.1), --seed, the measurement
(DA3 realized_poses on [decoded frame 0 + latent ends], metric / metres_per_anchor_unit), write / replan logic, --scale
(anchor-depth units -> Astronex units; astronex_calib.py picks it with queue_astronex.sh's rule).
Usage: astronex_fair.py --arms arms_4.json --arm baseline --scale S --out RUNS/astronex [--videos ...]"""
import argparse, json, os, runpy, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_common as fc

REPO = Path('@WORKSPACE@/sota_repos/astronex_world'); WEIGHTS = Path('@WORKSPACE@/sota_repos/Astronex')
NLAT = 48; BLOCK = 8; OUT_W, OUT_H = 832, 480
CTX = dict(videos=[], i=-1, arm=None, flags=None, scale=1., da3=None, out=None, seed=7, min_free=10.)


def w2c_astro(c2w):
    c = np.asarray(c2w, np.float64).copy(); c[:3, 3] *= CTX['scale']
    return np.linalg.inv(c)


def block_done(L, v, st, final=False):
    """verbatim astronex_fprime.block_done: stream-decode the finished block (VAE causal cache), denoiser parked on the CPU."""
    pipe, out, vm = L['self'], L['output'], L['viewmats']
    cur = int(L.get('cache_start_frame', L['current_start_frame']))
    if int(L['current_start_frame']) != cur:
        raise RuntimeError('camera and latent indices disagree')
    t0 = time.perf_counter()
    if st['prev'] == 0:
        pipe.vae.reset_stream()
    gdev = next(pipe.generator.parameters()).device
    pipe.generator.to('cpu'); torch.cuda.empty_cache()
    for j in range(st['prev'], cur):
        px = pipe.vae.decode_to_pixel(out[:, j:j + 1], use_cache=True)
        st['frames'].append(((px[0].permute(0, 2, 3, 1).float() * .5 + .5).clamp(0, 1) * 255).round().byte().cpu().numpy()); del px
    torch.cuda.empty_cache()
    st['t_decode'] += time.perf_counter() - t0; done0 = st['prev']; st['prev'] = cur
    gen = np.concatenate(st['frames'])
    try:
        _measure_and_act(L, v, st, gen, CTX['flags'], vm, cur, done0, final)
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
                v['cmd'][j] = fc.se3_interp(delta, 1. - (i + 1) / n) @ v['plan'][j]; vm[0, j] = torch.from_numpy(w2c_astro(v['cmd'][j])).to(vm)
            v['events'].append(dict(block_end=cur, position_offset=float(np.linalg.norm(delta[:3, 3]))))
    if final:
        name = v['name']; dst = CTX['out'] / CTX['arm']['name'] / f"{name}_s{CTX['seed']}.mp4"
        log = dict(model='astronex', arm=CTX['arm'], scene=name, seed=CTX['seed'], scale=CTX['scale'], n_rgb=int(len(gen)), prompt=v['prompt'],
                   planned=[p.tolist() for p in v['plan']], commanded=[c.tolist() for c in v['cmd']], pose_frame_ids=[4 * j for j in range(NLAT)],
                   measured=[None if v['measured'].get(j) is None else v['measured'][j].tolist() for j in range(NLAT)], events=v['events'],
                   seconds=round(time.time() - st['t_start'], 1), decode_s=round(st['t_decode'], 1), measure_s=round(st['t_measure'], 1),
                   peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   settings=dict(config='inference/causal_consumer.yaml', latents=NLAT, hw=[OUT_H, OUT_W], kept_frames=fc.N_FRAMES))
        fc.save_run(dst, list(gen[1:]), v['frame0'], v['K'], log, preprocess='astronex')
        print(json.dumps(dict(scene=name, arm=CTX['arm']['name'], frames=len(gen) - 1, sec=log['seconds'], peak=log['peak_GiB'], events=len(v['events']))), flush=True)


def hook(L, final=False):
    """Called by the patched sampler at the start of every block after the first, and once after the last block."""
    key = id(L['output'])
    if CTX.get('key') != key:                                                           # a new video (new inference call)
        if fc.free_gib() < CTX['min_free']:
            print(json.dumps(dict(stop='disk floor', free_gib=round(fc.free_gib(), 2))), flush=True); os._exit(3)
        CTX['key'] = key; CTX['i'] += 1; torch.cuda.reset_peak_memory_stats()
        CTX['st'] = dict(prev=0, frames=[], t_decode=0., t_measure=0., t_start=time.time())
    block_done(L, CTX['videos'][CTX['i']], CTX['st'], final)


def patch_sampler():
    """verbatim astronex_fprime.patch_sampler: a copy of causal_diffusion_inference with two hook calls replaces the module."""
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
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', type=Path, required=True); ap.add_argument('--arm', required=True)
    ap.add_argument('--scale', type=float, required=True); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--videos', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--min_free_gib', type=float, default=10.)
    a = ap.parse_args(); a.out = a.out.resolve(); arm = [x for x in json.load(open(a.arms)) if x['name'] == a.arm][0]
    flags = dict(measure=1, write=int(arm.get('write') == 'measured'), replan=int(bool(arm.get('replan'))))
    CTX.update(arm=arm, flags=flags, scale=a.scale, out=a.out, seed=a.seed, min_free=a.min_free_gib)
    from torchvision import transforms
    resize = transforms.Resize((OUT_H, OUT_W))                                         # = sample.py's I2V transform before ToTensor
    vids = a.videos or [s['name'] for s in fc.index()['scenes']]; rows = []; work = a.out / '_inputs' / arm['name']; work.mkdir(parents=True, exist_ok=True)
    for name in vids:
        if (a.out / arm['name'] / f'{name}_s{a.seed}.mp4').exists():
            continue
        sc = fc.scene(name); src = Image.open(sc['image_path']).convert('RGB')
        K, _ = fc.k_astronex(sc['K'], src.size, (OUT_W, OUT_H)); plan = [sc['c2w_ext'][4 * j].copy() for j in range(NLAT)]
        Kn = np.array([[K[0, 0] / OUT_W, 0, K[0, 2] / OUT_W], [0, K[1, 1] / OUT_H, K[1, 2] / OUT_H], [0, 0, 1]], np.float32)
        np.savez(work / f'{name}_cam.npz', viewmats=np.stack([w2c_astro(c) for c in plan]).astype(np.float32), Ks=np.stack([Kn] * NLAT))
        rows.append(dict(prompt=sc['prompts']['astronex'], image_path=sc['image_path'], camera_path=str(work / f'{name}_cam.npz'),
                         output_name=f'{name}.mp4', num_output_frames=NLAT - 1))
        CTX['videos'].append(dict(name=name, plan=plan, cmd=[c.copy() for c in plan], measured={}, events=[], K=K, mpu=float(sc['metres_per_anchor_unit']),
                                  frame0=np.asarray(resize(src)), prompt=sc['prompts']['astronex']))
    if not rows:
        print('nothing to do'); return
    if fc.free_gib() < a.min_free_gib:
        print(json.dumps(dict(stop='disk floor', free_gib=round(fc.free_gib(), 2)))); sys.exit(3)
    (work / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    import astronex_env
    cfg = astronex_env.resolve_config(str(REPO / 'inference/causal_consumer.yaml'), str(work / '_cfg.yaml'), weights_dir=str(WEIGHTS))
    os.environ.update(ASTRONEX_WEIGHTS=str(WEIGHTS), WAN_SKIP_DECODE='1')
    from realized_pose import load_da3
    CTX['da3'] = load_da3('cpu')
    patch_sampler()
    import random, torch.utils.data as tud

    class SeedPerVideoLoader(tud.DataLoader):
        """sample.py seeds once per process (sample.py:322-324), so a video's noise would depend on its manifest position (and
        change when a resumed run skips finished videos). Every video here starts from the RNG state the first one had, i.e.
        the state of the official one-video-per-process generate.py call (set_seed(seed) + the same model-loading draws)."""
        def __iter__(self):
            state = None
            for b in super().__iter__():
                if state is None:
                    state = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
                else:
                    torch.set_rng_state(state[0]); torch.cuda.set_rng_state_all(state[1]); np.random.set_state(state[2]); random.setstate(state[3])
                yield b
    tud.DataLoader = SeedPerVideoLoader                                                  # sample.py imports DataLoader after this point
    _save = torch.save                                                                  # drop sample.py's latent-only dump (disk floor)
    torch.save = lambda obj, f, *aa, **kw: None if str(f).endswith('_latents.pt') and str(work) in str(f) else _save(obj, f, *aa, **kw)
    sys.argv = ['inference.sample', '--config_path', cfg, '--checkpoint_path', str(WEIGHTS), '--data_path', str(work / 'manifest.jsonl'),
                '--wbench_manifest', str(work / 'manifest.jsonl'), '--output_folder', str(work / 'sample_out'), '--num_output_frames', str(NLAT - 1),
                '--seed', str(a.seed), '--i2v']
    runpy.run_module('inference.sample', run_name='__main__')


if __name__ == '__main__':
    main()
