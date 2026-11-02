"""Astronex native rollout with causal geometry-unit calibration.

Target poses are supplied in reference-anchor units. A runtime provider must
produce native-unit depth from the initial input; this launcher never fits scale
from recorded motion or reads prepared native_w2c_latent arrays. Without such
geometry the unit mapping is unavailable, and generation is not launched.
Native consumer settings, latent/RGB indexing, and observer schedule are retained.
"""
import argparse, json, os, runpy, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'fair_common_20261025'))
import fair_common as fc
from causal_units import runtime_anchor_scale, require_runtime_provider
fc.BATCH=Path(os.environ['LONG_BATCH']);fc.N_FRAMES=76
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'new_models_loop_20261028'))
from common import PoseWorker,rejoin
import atexit
WORKER=None
def close_worker():
    if WORKER is not None:WORKER.close()
atexit.register(close_worker)

REPO = Path('@WORKSPACE@/sota_repos/astronex_world'); WEIGHTS = Path('@WORKSPACE@/sota_repos/Astronex')
NLAT = 24; BLOCK = 8; OUT_W, OUT_H = 832, 480
CTX = dict(videos=[], i=-1, arm=None, flags=None, scale=None, da3=None, out=None, seed=7, min_free=10.)


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
    global WORKER
    if v.get('response') is not None:
        runtime=v['response'];n=min(BLOCK,NLAT-cur)
        commands,event,measured=runtime.observe(gen,cur-1,cur+n)
        v['measured'][cur-1]=measured;v['events'].append(event)
        if n:
            v['cmd'][cur:cur+n]=list(commands)
            if event['commands_differ_from_bootstrap']:
                vm[0,cur:cur+n]=torch.as_tensor(np.linalg.inv(commands)).to(vm)
    elif arm['measure'] and not final:
        if WORKER is None:WORKER=PoseWorker(CTX['out']/'observer'/v['name'])
        prefix=gen.copy();prefix[0]=v['frame0']
        with torch.random.fork_rng(devices=[0]):
            measured,info=WORKER.measure(prefix,v['K'],v['anchor_depth'])
        v['measured'][cur-1]=measured
        if arm['replan']:
            n=min(BLOCK,NLAT-cur);future=np.asarray(v['plan'][cur:cur+n])
            if CTX['arm']['name']=='sham':
                commands=rejoin(v['plan'][cur-1],v['plan'][cur-1],future)
                assert np.array_equal(commands,future)
                # Exact identity means leave native camera tensors untouched.
            else:
                desired=rejoin(measured,v['plan'][cur-1],future)
                commands=(v['cmd'][cur-1]@np.linalg.inv(measured))@desired
                for i,j in enumerate(range(cur,cur+n)):
                    v['cmd'][j]=commands[i];vm[0,j]=torch.from_numpy(w2c_astro(commands[i])).to(vm)
        v['events'].append(dict(block_end=cur,observed_frame=4*(cur-1),applied=arm['replan'] and CTX['arm']['name']!='sham',measured=measured.tolist(),measurement=info))
    if final:
        name = v['name']; dst = CTX['out'] / CTX['arm']['name'] / f"{name}_s{CTX['seed']}.mp4"
        log = dict(model='astronex', arm=CTX['arm'], scene=name, seed=CTX['seed'], scale=CTX['scale'], unit_calibration=v['unit_calibration'], n_rgb=int(len(gen)), prompt=v['prompt'],
                   planned=[p.tolist() for p in v['plan']], commanded=[c.tolist() for c in v['cmd']], pose_frame_ids=[4 * j for j in range(NLAT)],
                   measured=[None if v['measured'].get(j) is None else v['measured'][j].tolist() for j in range(NLAT)], events=v['events'],
                   seconds=round(time.time() - st['t_start'], 1), decode_s=round(st['t_decode'], 1), measure_s=round(st['t_measure'], 1),
                   peak_GiB=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   settings=dict(config='inference/causal_consumer.yaml', latents=NLAT, hw=[OUT_H, OUT_W], kept_frames=fc.N_FRAMES))
        if v.get('response') is not None:
            log['native_commanded']=log.pop('commanded');log['online_response']=v['response'].metadata();v['response'].close()
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
    global NLAT
    ap = argparse.ArgumentParser(); ap.add_argument('--arms', type=Path, required=True); ap.add_argument('--arm', required=True)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--videos', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--min_free_gib', type=float, default=10.)
    ap.add_argument('--online-response',action='store_true');ap.add_argument('--latents',type=int,default=24)
    a = ap.parse_args(); a.out = a.out.resolve(); arm = [x for x in json.load(open(a.arms)) if x['name'] == a.arm][0]
    flags = dict(measure=int(arm['name']!='native'), write=int(arm.get('write') == 'measured'), replan=int(bool(arm.get('replan'))))
    CTX.update(arm=arm, flags=flags, scale=None, out=a.out, seed=a.seed, min_free=a.min_free_gib)
    if a.online_response:
        if arm['name'] not in ('A','S','C') or arm.get('write')=='measured':raise ValueError('Online response requires command-only A/S/C arms')
        if a.latents<16 or a.latents%8:raise ValueError('Complete Astronex blocks required')
        NLAT=a.latents;fc.N_FRAMES=4*(NLAT-1)
    else:
        if a.latents!=24:raise ValueError('Custom duration requires online-response mode')
        require_runtime_provider('astronex')
    from torchvision import transforms
    resize = transforms.Resize((OUT_H, OUT_W))                                         # = sample.py's I2V transform before ToTensor
    vids = a.videos or [s['name'] for s in fc.index()['scenes']]; rows = []; work = a.out / '_inputs' / arm['name']; work.mkdir(parents=True, exist_ok=True)
    for name in vids:
        if (a.out / arm['name'] / f'{name}_s{a.seed}.mp4').exists():
            continue
        sc = fc.scene(name); src = Image.open(sc['image_path']).convert('RGB')
        K, _ = fc.k_astronex(sc['K'], src.size, (OUT_W, OUT_H)); plan = [sc['c2w_ext'][4 * j].copy() for j in range(NLAT)]
        Kn = np.array([[K[0, 0] / OUT_W, 0, K[0, 2] / OUT_W], [0, K[1, 1] / OUT_H, K[1, 2] / OUT_H], [0, 0, 1]], np.float32)
        frame0 = np.asarray(resize(src))
        anchor_depth = fc.remap_depth(sc['depth'],sc['K'],K,(OUT_H,OUT_W))
        response=None
        if a.online_response:
            from camera_loop.native_response_runtime import BoundaryRuntime
            response=BoundaryRuntime(plan,np.arange(NLAT)*4,model='astronex',arm=arm['name'],
                out=a.out/arm['name']/f'{name}_s{a.seed}_response',anchor=frame0,K=K,depth=anchor_depth,workspace='@WORKSPACE@')
            native=np.linalg.inv(response.control.commands)
            unit_calibration=dict(status='physical_units_unidentified',mode='online_command_response')
        else:
            CTX['scale'], unit_calibration = runtime_anchor_scale('astronex', frame0, anchor_depth, K,
                receipt=work/f'{name}_s{a.seed}_online_units.json')
            native = np.stack([w2c_astro(c) for c in plan])
        np.savez(work/f'{name}_cam.npz',viewmats=native,Ks=np.stack([Kn]*NLAT))
        rows.append(dict(prompt=sc['prompts']['astronex'], image_path=sc['image_path'], camera_path=str(work / f'{name}_cam.npz'),
                         output_name=f'{name}.mp4', num_output_frames=NLAT - 1))
        CTX['videos'].append(dict(name=name, plan=plan, response=response, cmd=[c.copy() for c in (plan if response is None else response.control.commands)], measured={}, events=[], K=K, unit_calibration=unit_calibration,
                                  anchor_depth=anchor_depth, frame0=frame0, prompt=sc['prompts']['astronex']))
    if not rows:
        print('nothing to do'); return
    if fc.free_gib() < a.min_free_gib:
        print(json.dumps(dict(stop='disk floor', free_gib=round(fc.free_gib(), 2)))); sys.exit(3)
    (work / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'astronex_plugins'))
    import astronex_env
    cfg = astronex_env.resolve_config(str(REPO / 'inference/causal_consumer.yaml'), str(work / '_cfg.yaml'), weights_dir=str(WEIGHTS))
    os.environ.update(ASTRONEX_WEIGHTS=str(WEIGHTS), WAN_SKIP_DECODE='1')
    assert len(rows)==1, 'one video per process preserves its native-unit conversion'
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
