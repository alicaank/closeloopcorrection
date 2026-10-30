"""History-momentum test on WorldPlay-5B (as memorizon_plugins/calib_20261016/history_probe.py): with the SAME command and the
SAME initial noise, does the executed camera motion of a chunk follow the motion already present in the history latents?

Main trajectory = WorldPlay baseline at native scale (unit 1, recorded plan, commanded write). At probed chunks (3, 5, 7, 9),
from the same state:
  gen        : generated history (baseline)
  rec_recent : the 4 most recent history latents replaced by the recording's latents (true motion, poses exact)
  rec_all    : every history latent 1..lo-1 replaced by the recording's latents
Recording latents: one causal VAE encode of recorded frames 0..156 at 960x544 (DL3DV-480P source upscaled; softer than
generated frames: disclosed confound). Camera conditioning (viewmats/actions) is identical across branches. Each branch's
new chunk is decoded with a decoder rebuilt from that branch's own history. DA3 (da3_window) on [anchor + subsampled history
latent-end frames of that branch + latents lo-5..lo-1 + 4 new]: executed motion of the chunk (start = latent lo-1) and the
history's own motion over its last 4 latents (lo-5 -> lo-1). JSON only.
Launch: torchrun --nproc_per_node=1 (baselines/venv_worldplay).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from wp_fprime import WorldPlay, Roller, small, load_da3, da3_window, BATCH, H, W, C
sys.path.insert(0, '@WORKSPACE@'); sys.path.insert(0, '@WORKSPACE@/training_runs/base_restart_20261008')
from all_scene_dataset import DATA, RecordedStream
from gaussverse.data.dl3dv import crop_and_resize
PROBE = (3, 5, 7, 9); BR = ('gen', 'rec_recent', 'rec_all')


class Hist(Roller):
    def norm_stats(self):
        v = self.wp.vae; m = torch.tensor(v.config.latents_mean).view(1, -1, 1, 1, 1); s = torch.tensor(v.config.latents_std).view(1, -1, 1, 1, 1)
        return m.to(self.wp.device), s.to(self.wp.device)

    @torch.no_grad()
    def encode_recording(self, scene, n_frames):
        idx = {s['name']: s for s in json.load(open(BATCH / 'index.json'))['scenes']}
        streams = {json.loads(l)['stream_id']: json.loads(l) for l in (DATA / 'corpus/streams.jsonl').open()}
        st = RecordedStream(streams[idx[scene]['stream_id']]); big = []; self.rec448 = []
        for i in range(n_frames):
            with Image.open(st.paths[i]) as im:
                big.append(np.asarray(crop_and_resize(im.convert('RGB'), W, H)))
            self.rec448.append(st.frame(i))
        x = torch.from_numpy(np.stack(big).astype(np.float32) / 127.5 - 1).permute(3, 0, 1, 2)[None].to(self.wp.device, torch.bfloat16)
        v = self.wp.vae; v.clear_cache()
        z = v.encode(x, return_dict=False)[0].mode()                                              # posterior mode (pipeline samples; mode = deterministic)
        v.clear_cache(); m, s = self.norm_stats()
        self.rec = ((z.float() - m) / s).to(torch.bfloat16); del x, z; torch.cuda.empty_cache()

    @torch.no_grad()
    def decode_all(self, lat, upto):
        """Decode latents 0..upto-1 from scratch with the causal decoder; returns latent-end frames (448) for latents 0..upto-1."""
        v = self.wp.vae; m, s = self.norm_stats(); ends = []
        for j in range(upto):
            z = lat[:, :, j:j + 1].float() * s + m
            out = v.decode(z.to(torch.bfloat16), is_first_chunk=(j == 0))[0]                   # [1,3,f,H,W] in [-1,1]
            f = ((out[0, :, -1].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            ends.append(small(f))
        v.clear_cache(); return ends

    @torch.no_grad()
    def run_hist(self, log=print):
        wp, pipe = self.wp, self.wp.pipe; pe, ne = wp.prompt(self.text); n_lat = 4 * self.chunks; nf = 4 * (n_lat - 1) + 1
        self.encode_recording(self.scene, nf)
        args = dict(prompt=None, prompt_embeds=pe, negative_prompt_embeds=ne, height=H, width=W, num_frames=nf, num_inference_steps=4,
                    guidance_scale=1.0, few_step=True, first_chunk_size=C, return_dict=False, image_path=self.image,
                    use_memory=True, context_window_length=16, generator=torch.Generator(device='cpu').manual_seed(self.seed))
        pipe.ctx = None; recs = []
        for ci in range(self.chunks):
            lo, hi = 4 * ci, 4 * ci + 4; self.mem[lo:hi] = self.cmd[lo:hi]; vm, ks, act = self.wp_rows(lo, hi)
            if ci not in PROBE:
                pipe(**args, chunk_i=ci, viewmats=vm, Ks=ks, action=act, output_type='latent'); continue
            t0 = time.time(); saved = pipe.ctx['latents'].clone(); res = {}
            for b in BR:
                lat = saved.clone()
                if b == 'rec_recent':
                    lat[:, :, lo - 4:lo] = self.rec[:, :, lo - 4:lo].to(lat)
                elif b == 'rec_all':
                    lat[:, :, 1:lo] = self.rec[:, :, 1:lo].to(lat)
                pipe.ctx['latents'] = lat
                pipe(**args, chunk_i=ci, viewmats=vm, Ks=ks, action=act, output_type='latent')
                res[b] = pipe.ctx['latents'].clone()
            pipe.ctx['latents'] = res['gen']                                                      # main trajectory continues as baseline
            est = {}
            for b in BR:
                ends = self.decode_all(res[b], hi)                                                 # latent-end frames 0..hi-1 of this branch
                views_hist = ends[1:lo]; new = ends[lo:hi]
                if b != 'gen':                                                                     # recorded history shows the recording
                    k0 = 1 if b == 'rec_all' else lo - 4
                    views_hist = ends[1:k0] + [self.rec448[4 * j] for j in range(k0, lo)]
                frames = views_hist + new
                self.da3.to('cuda'); e = da3_window(self.da3, self.anchor, self.anchor_depth, frames, self.K, max_views=16, tail=9); self.da3.to('cpu'); torch.cuda.empty_cache()
                P = e['poses']; n = len(frames)
                est[b] = dict(start=P[n - 5].tolist() if P[n - 5] is not None else None, end=P[n - 1].tolist() if P[n - 1] is not None else None,
                              hist5=P[n - 9].tolist() if n >= 9 and P[n - 9] is not None else None)
            pl = self.plan; cmd = np.linalg.norm(pl[hi - 1][:3, 3] - pl[lo - 1][:3, 3]); hcmd = np.linalg.norm(pl[lo - 1][:3, 3] - pl[lo - 5][:3, 3])
            row = dict(chunk=ci, planned_step=float(cmd), planned_hist_step=float(hcmd), est=est, seconds=time.time() - t0)
            T = lambda p: np.array(p)[:3, 3]
            row['exec_ratio'] = {b: (float(np.linalg.norm(T(est[b]['end']) - T(est[b]['start'])) / max(cmd, 1e-6)) if est[b]['end'] and est[b]['start'] else None) for b in BR}
            row['hist_ratio'] = {b: (float(np.linalg.norm(T(est[b]['start']) - T(est[b]['hist5'])) / max(hcmd, 1e-6)) if est[b]['hist5'] and est[b]['start'] else None) for b in BR}
            recs.append(row); log(json.dumps(dict(chunk=ci, exec=row['exec_ratio'], hist=row['hist_ratio'], sec=round(row['seconds'], 1))))
        return recs


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', required=True); p.add_argument('--seeds', nargs='+', type=int, default=[7]); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); wp = WorldPlay(); da3 = load_da3('cpu')
    for scene in a.scenes:
        for seed in a.seeds:
            dst = a.out / f'{scene}_s{seed}.json'
            if dst.exists():
                continue
            t0 = time.time(); r = Hist(wp, da3, scene, dict(name='hist_baseline'), seed, 10, 1.0); r.scene = scene
            recs = r.run_hist(log=lambda s: print(scene, seed, s, flush=True))
            json.dump(dict(scene=scene, seed=seed, unit=1.0, branches=BR, seconds=time.time() - t0, chunks=recs), open(dst, 'w'))
            print(scene, seed, 'done', round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
