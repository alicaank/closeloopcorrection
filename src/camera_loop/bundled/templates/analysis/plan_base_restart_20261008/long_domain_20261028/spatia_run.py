"""spatia_fair.py = copy of spatia_plugins/spatia_fprime.py with ONLY the output format changed (lead spec 10-25, fair external
comparison): instead of a frames npz, every arm writes <out>/<arm>/<scene>_s<seed>.mp4 (frames 0..num_frames-1 at Spatia's native
1248x704, frame 0 = the PROCESSED start image, i.e. the PIL stretch Spatia feeds in, not Spatia's decode of it; libx264 crf 12,
preset slow, yuv420p, 16 fps, via fair_common.write_mp4) + the same json plus K_out / out_hw / preprocess for the generic scorers
(fair_common_20261025/score_ws_fair.py, score_vggt_fair.py). Imports (fprime_poses) and the DA3 worker still come from
spatia_plugins/; the original file is untouched.

spatia_fprime.py -- closed-loop camera correction (F') harness for released Spatia (Wan2.2-TI2V-5B + VACE point-cloud
control + LoRA, MapAnything point-cloud memory). Run in venv_spatia; cwd is switched to @WORKSPACE@/sota_repos/spatia.
Spatia's repo is NOT modified: its own functions are imported from inference.py and its round loop (inference.py:473-598)
is re-stated here with the arm substitutions marked "F'".

  env -u LD_LIBRARY_PATH .../venv_spatia/bin/python spatia_fprime.py --batch <batch dir> --scene <name> --out <dir> \
      [--arms baseline,write_measured,replan_commanded,replan_measured] [--seed 20917] [--num_frames 161]

Inputs: a batch in the lyra2_plugins/batch_dev10 format (index.json, images/<name>.png, trajectories/<name>.npz with w2c
anchor-relative OpenCV in anchor-median-depth units + pixel intrinsics, captions/<name>.json).
Outputs (original npz layout, replaced here by the mp4 described above): <out>/<arm>/<scene>_s<seed>.npz  (frames uint8 [num_frames,704,1248,3]; frames[i] <-> planned[i]; frames[0] is
Spatia's own decode of the start image) and .json (planned / commanded / measured / history_label per frame, OpenCV c2w in
the batch gauge; output-resolution pixel K; events with per-round residuals, timing, peak GPU memory).

Arms (all measure every generated frame with DA3 after each round, for logging):
  baseline          Spatia as is, driven by the planned path.
  write_measured    history poses = DA3-measured poses (anchor gauge, x the same metric factor D as the commanded path) in
                    the three places Spatia uses history extrinsics: MapAnything point-cloud input (inference.py:496,511),
                    the render poses of the round's 9 history frames (inference.py:510) and frustum reference selection
                    (inference.py:543).
  replan_commanded  the next round's NEW-frame targets (inference.py:510 for frames >= the previous round end; they also
                    feed new_cam_w2c, inference.py:550) = rejoin_offset(measured end pose of the previous round, plan,
                    planned previous end): the start offset decays linearly to 0 at the round's last real frame.
  replan_measured   both.
Correction acts only at round boundaries (a round is generated in one call): with the official 121 + 81-frame rounds
(9 history frames) the round starts are 0, 112, 184, ... so num_frames 161 -> ONE correction point (before frame 121),
num_frames 241 -> two (before 121 and 193). --first_round_frames / --round_frames (4k+1, > hist) give more, e.g.
--round_frames 41 -> 161 frames in rounds [0,121) [112,153) [144,161) -> two correction points (non-official setting).

One process = one (scene, seed). Round 0 is arm-independent (nothing is measured before the first correction point;
inference.py:487-489 runs it without LoRA and without history), so it is generated ONCE and every arm branches from its
state. Every later round of every arm runs with the LoRA loaded (inference.py:490-493), so nothing is unloaded/reloaded
between arms; every pipe.call_latent_inference re-seeds its noise from --seed (inference.py:568). Branching is therefore
the same computation as separate runs (up to GPU nondeterminism) at ~60 % of the cost for 4 arms.

Gauge: Spatia hands every pose to MapAnything as METRIC (utils/map_anything_inference.py:126-130) and the round-0 cloud is
built from the start image alone, i.e. in MapAnything's metric depth. The plan is therefore fed with translations x D,
D = median per-pixel depth of the round-0 cloud seen from the anchor (fprime_poses.zbuffer_median_depth) = metric units per
anchor-median-depth unit (--scale_mode mapanything; 'fixed' takes --units_per_anchor_depth, 'none' feeds raw units).
DA3 poses come back in anchor-median-depth units (scale fixed per DA3 call on the anchor depth) and are written with x D.

Deviations from inference.py (also listed in every json 'deviations'):
  * the start image is resized to 1248x704 with PIL .resize((W,H)) -- the same stretch the pipeline applies to the first-
    frame latent (wan/wan_video_new.py:528) -- BEFORE it enters the history list; inference.py:469-470 puts the raw image
    there (official test images are already 1248x704; any other size breaks the MapAnything history video).
  * translation scale D (above); prompt = batch caption (largest caption key <= round start) instead of prompt.txt lines.
  * force_rebuild_intermediate=True and per-arm work dirs (official default reuses a cached cloud/render if present).
  * no final mp4; the frames npz holds the first num_frames frames of Spatia's own joint final decode (inference.py:600-607;
    the last round's padding frames are dropped). tqdm replaced by a step timer (one cuda synchronize per step).
"""
import argparse, json, os, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
import numpy as np

HERE = Path('@WORKSPACE@/analysis/plan_base_restart_20261008/spatia_plugins')         # fprime_poses + spatia_da3_worker.py live here
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fprime_poses as FP                                                     # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'fair_common_20261025'))
import fair_common as fc                                                      # noqa: E402  (mp4 writer, shared run format)

SPATIA = Path('@WORKSPACE@/sota_repos/spatia')
MODELS = Path('@WORKSPACE@/baselines/models/spatia')
SPATIA_PY = '@WORKSPACE@/baselines/venv_spatia/bin/python'
DA3_PY = '@WORKSPACE@/baselines/venv_vmem/bin/python'
COMP = Path('@WORKSPACE@/training_runs/base_restart_20261008/dataset_v3/components')
ARMS = {
    'baseline': dict(name='baseline', write='commanded', replan=False),
    'sham': dict(name='sham', write='commanded', replan=True),
    'write_measured': dict(name='write_measured', write='measured', replan=False),
    'replan_commanded': dict(name='replan_commanded', write='commanded', replan=True),
    'replan_measured': dict(name='replan_measured', write='measured', replan=True),
}
DEVIATIONS = [
    'deterministic PyTorch3D naive rasterization (bin_size=0), same point radius and compositor; avoids coarse-bin tie-order noise',
    'start image PIL-resized (stretch) to WxH before entering the history list (pipeline does the same for the first-frame latent)',
    'plan translations x spatia_units_per_anchor_depth (MapAnything metric depth of the anchor) -- Spatia expects metric poses',
    'prompt = batch caption (largest key <= round start) + Spatia suffix; no prompt.txt',
    'force_rebuild_intermediate=True, per-arm work dirs; round 0 generated once and shared by all arms of a (scene, seed)',
    'mp4 (crf 12) = processed start image + frames 1..num_frames-1 of the joint final decode (last-round padding dropped); tqdm -> step timer',
]


def jdefault(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def atomic_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, default=jdefault)
    os.replace(tmp, path)


def free_gib(path='@HOME_ROOT@'):
    return shutil.disk_usage(path).free / 2**30


# ---------------------------------------------------------------------------------------------------------- monitors
class SmiPeak(threading.Thread):
    """Device-wide nvidia-smi memory.used peak (covers the MapAnything / renderer subprocesses and the DA3 worker)."""
    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        vis = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
        self.idx = vis if vis.isdigit() else '0'; self.period = period; self.peak = 0; self.win = 0; self.stop = False

    def run(self):
        while not self.stop:
            try:
                r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', self.idx],
                                   capture_output=True, text=True, timeout=10)
                v = int(r.stdout.strip().splitlines()[0])
                self.peak = max(self.peak, v); self.win = max(self.win, v)
            except Exception:
                pass
            time.sleep(self.period)

    def window(self):
        v = self.win; self.win = 0; return v


class StepTimer:
    """progress_bar_cmd for pipe.call_latent_inference: per-step wall time with a cuda synchronize at each step."""
    def __init__(self, sync, label=''):
        self.sync = sync; self.times = []; self.label = label

    def __call__(self, it, disable=False):
        self.times = []; self.sync(); t = time.perf_counter(); n = len(it) if hasattr(it, '__len__') else None
        for i, x in enumerate(it):
            yield x
            self.sync(); now = time.perf_counter(); self.times.append(now - t); t = now
            if (i + 1) % 10 == 0 or (n is not None and i + 1 == n):
                print(f'[F\'] {self.label} step {i + 1}/{n}  {self.times[-1]:.2f} s/it', flush=True)


class DA3Client:
    """Persistent spatia_da3_worker.py process (venv_vmem); arrays travel through npz files in a per-PID /dev/shm dir."""
    def __init__(self, python, log_path, ipc_dir):
        env = dict(os.environ); env.pop('LD_LIBRARY_PATH', None); env['HF_HUB_OFFLINE'] = '1'
        self.log = open(log_path, 'a'); self.ipc = Path(ipc_dir); self.ready = None
        self.p = subprocess.Popen([python, str(HERE / 'spatia_da3_worker.py')], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.log, text=True, bufsize=1, env=env, cwd=str(HERE))

    def _line(self):
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f'DA3 worker exited (rc={self.p.poll()}), see {self.log.name}')
        return line.strip()

    def wait_ready(self):
        line = self._line()
        if not line.startswith('READY'):
            raise RuntimeError('DA3 worker: unexpected first line ' + line[:200])
        self.ready = json.loads(line[len('READY'):]); return self.ready

    def measure(self, anchor, frames, K, query, anchor_depth, max_views, tail):
        req, out = self.ipc / 'req.npz', self.ipc / 'out.npz'
        np.savez(req, anchor=anchor, frames=np.stack(frames), K=np.asarray(K, np.float32), query=np.asarray(query, np.int64),
                 anchor_depth=np.zeros((0,), np.float32) if anchor_depth is None else np.asarray(anchor_depth, np.float32),
                 max_views=np.int64(max_views), tail=np.int64(tail))
        self.p.stdin.write(json.dumps(dict(cmd='measure', req=str(req), out=str(out))) + '\n'); self.p.stdin.flush()
        rep = json.loads(self._line())
        if not rep.get('ok'):
            raise RuntimeError('DA3 worker: ' + str(rep.get('error')))
        poses = np.load(out)['poses']
        if np.isfinite(poses).all():req.unlink()
        return {int(q): (poses[i].copy() if np.isfinite(poses[i]).all() else None) for i, q in enumerate(query)}, rep

    def close(self):
        try:
            self.p.stdin.write(json.dumps(dict(cmd='quit')) + '\n'); self.p.stdin.flush(); self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self.log.close()


# ---------------------------------------------------------------------------------------------------------- inputs
def load_scene(batch, name, F):
    from PIL import Image
    batch = Path(batch); idx = json.load(open(batch / 'index.json'))
    ent = [s for s in idx['scenes'] if s['name'] == name]
    if not ent:
        raise KeyError(f'scene {name} not in {batch}/index.json')
    z = np.load(batch / 'trajectories' / f'{name}.npz')
    w2c = z['w2c'].astype(np.float64)
    if len(w2c) < F:
        raise ValueError(f'{name}: trajectory has {len(w2c)} poses < num_frames {F}')
    plan = np.linalg.inv(w2c[:F]); notes = []
    if np.abs(plan[0] - np.eye(4)).max() > 1e-4:                                 # the format is anchor-relative; enforce it
        plan = np.linalg.inv(plan[0]) @ plan; notes.append('re-anchored plan to frame 0')
    Wb, Hb = int(z['image_width']), int(z['image_height'])
    Ks = z['intrinsics'][:F].astype(np.float64); K = Ks[0]
    if np.abs(Ks - K).max() > 1e-3 * abs(K[0, 0]):
        notes.append('per-frame intrinsics vary; Spatia takes one K -> frame 0 used')
    img = Image.open(batch / 'images' / f'{name}.png').convert('RGB')
    if img.size != (Wb, Hb):
        notes.append(f'image {img.size} != trajectory image size {(Wb, Hb)}; normalized K from the trajectory size')
    cap_path = batch / 'captions' / f'{name}.json'
    captions = json.load(open(cap_path)) if cap_path.exists() else idx.get('prompt', '')
    return dict(name=name, entry=ent[0], plan=plan, K_batch=K, Wb=Wb, Hb=Hb, kn=FP.normalized_intrinsics(K, Wb, Hb), image=img,
                captions=captions, default_prompt=idx.get('prompt', ''), stream_id=ent[0].get('stream_id'), notes=notes,
                batch=str(batch))


def gauge_depth(mode, sc):
    """Anchor depth handed to DA3 for the per-call scale: 'comp' = stored component depth (only when it is exactly the
    batch image's resolution, i.e. the same crop: batch_dev10), else ones = 1/median(DA3 anchor depth) (the gauge's
    definition: translation in anchor-median-depth units)."""
    # The batch gauge is fixed from its own anchor, not a new all-ones gauge.
    from PIL import Image
    with np.load(Path(sc['batch'])/'trajectories'/f"{sc['name']}.npz") as z:
        d=z['anchor_depth'].astype(np.float32)
    assert np.isfinite(d).any()
    return d, 'frozen_batch_anchor'


def spatia_args(SI, a):
    """Spatia's own argparse defaults (inference.py:618-670) + our run values; derived strides come from its parser."""
    argv = ['inference.py', '--width', str(a.width), '--height', str(a.height), '--max_frames', str(a.num_frames),
            '--first_round_frames', str(a.first_round_frames), '--round_frames', str(a.round_frames),
            '--hist_frames', str(a.hist_frames), '--num_inference_steps', str(a.num_inference_steps),
            '--cfg_scale', str(a.cfg_scale), '--sampler', a.sampler, '--seed', str(a.seed),
            '--vace_path', str(MODELS / 'control_weight_8500.safetensors'), '--lora_path', str(MODELS / 'lora_weights_10000.safetensors'),
            '--map_python', SPATIA_PY, '--render_python', SPATIA_PY]
    old = sys.argv
    try:
        sys.argv = argv; s = SI.get_args()
    finally:
        sys.argv = old
    s.force_rebuild_intermediate = True
    return s


# ---------------------------------------------------------------------------------------------------------- state
class State:
    def __init__(self, F):
        self.frames = []; self.mframes = []; self.latents = []; self.cmd = None
        self.meas = [None] * F; self.labels = None; self.label_src = [None] * F
        self.events = []; self.rounds = []; self.seconds = 0.0; self.est = 0.0

    def branch(self):
        s = State(len(self.meas))
        s.frames = list(self.frames); s.mframes = list(self.mframes); s.latents = list(self.latents); s.cmd = self.cmd.copy()
        s.meas = list(self.meas); s.labels = self.labels.copy(); s.label_src = list(self.label_src)
        s.events = [dict(e, shared_round0=True) for e in self.events]; s.rounds = [dict(r) for r in self.rounds]
        return s


class Runner:
    def __init__(self, a, sc, SI, torch, pipe, lora_sd, sargs, da3, smi, work):
        from PIL import Image
        self.a, self.sc, self.SI, self.torch, self.pipe, self.lora_sd, self.s, self.da3, self.smi = a, sc, SI, torch, pipe, lora_sd, sargs, da3, smi
        self.F = a.num_frames; self.plan = sc['plan']; self.work = work; self.D = None; self.scale_info = None
        self.W, self.H = sargs.width, sargs.height
        self.K_out = FP.pixel_K(sc['kn'], self.W, self.H)
        self.mw = a.meas_width; self.mh = int(round(a.meas_width * self.H / self.W))
        self.K_meas = FP.pixel_K(sc['kn'], self.mw, self.mh)
        self.adepth, self.gauge = gauge_depth(a.gauge, sc)
        self.Image = Image
        self.intrinsic_path = work / 'camera_intrinsics_spatia.txt'
        FP.write_intrinsics_txt(self.intrinsic_path, sc['kn'])
        self.intrinsic = SI.read_intrinsics_from_txt(self.intrinsic_path)          # official reader, float32 [4]

    # -- helpers
    def to_t(self, c2w, units):
        return self.torch.from_numpy(FP.c2w_to_spatia_w2c(c2w, units).astype(np.float32))

    def small(self, pil):
        return np.asarray(pil.convert('RGB').resize((self.mw, self.mh), self.Image.BICUBIC))

    def prompt(self, start):
        frame = start if self.a.prompt_mode == 'per_round' else 0
        return FP.caption_for_frame(self.sc['captions'], frame, default=self.sc['default_prompt'])

    def set_scale(self, map_out):
        """Round 0: D from the anchor-only MapAnything cloud; the commanded path is then written in Spatia's own camera file
        format and read back with Spatia's reader (inference.py:75-85, 454) -- that tensor must equal what we feed."""
        a = self.a
        if a.scale_mode == 'mapanything':
            import open3d as o3d
            P = np.asarray(o3d.io.read_point_cloud(str(map_out['points_path'])).points)
            import cv2
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'continuous_eval_20261029'))
            from anchor_scale import point_cloud_depth, fit_scale, freeze_scale
            native = point_cloud_depth(P, FP.c2w_to_spatia_w2c(self.plan[0], 1.0), self.sc['kn'])
            reference = cv2.resize(self.adepth, (native.shape[1],native.shape[0]))
            self.D, self.scale_info = fit_scale(reference, native)
            self.D, self.scale_info = freeze_scale(self.D, self.scale_info, 'spatia', self.sc['name'])
            self.scale_info['native_depth_stage'] = 'anchor-only MapAnything cloud z-buffer'
            (self.work/'scale_calibration.json').write_text(json.dumps(self.scale_info, indent=2))
        elif a.scale_mode == 'fixed':
            self.D, self.scale_info = float(a.units_per_anchor_depth), dict(method='fixed')
        else:
            self.D, self.scale_info = 1.0, dict(method='none')
        if not (np.isfinite(self.D) and self.D > 0):
            raise RuntimeError(f'bad metric scale {self.D} {self.scale_info}')
        self.scale_info['mode'] = a.scale_mode
        txt = self.work / 'camera_w2c_spatia.txt'
        FP.write_w2c_txt(txt, FP.c2w_to_spatia_w2c(self.plan, self.D))
        ext = self.SI.read_w2cs_from_txt(txt)
        if not np.array_equal(ext, self.to_t(self.plan, self.D).numpy()):
            raise RuntimeError('camera txt round trip mismatch')
        print(f"[F'] spatia_units_per_anchor_depth D={self.D:.4f} {self.scale_info}", flush=True)

    # -- one Spatia round (inference.py:473-598) with the F' substitutions
    def do_round(self, st, r, arm, work_dir):
        SI, s, pipe, torch = self.SI, self.s, self.pipe, self.torch
        F, hist = self.F, s.hist_frames
        t_round = time.time(); tm = {}
        start, end, target = SI.get_round_range(r, F, s.first_round_frames, s.round_frames, hist)
        prev_end = len(st.frames); padding = target - (end - start)
        if r > 0 and start != prev_end - hist:
            raise RuntimeError(f'round range bookkeeping: start {start} != {prev_end} - {hist}')
        self.smi.window(); torch.cuda.reset_peak_memory_stats()
        prompt = self.prompt(start)
        print(f"[F'] {arm['name'] if r else 'shared'} round {r}: frames [{start},{end}) target={target} new=[{prev_end},{end}) pad={padding}", flush=True)
        if r == 0:                                                                # inference.py:487-493
            SI.set_pipe_units(pipe, hist_frames=hist, use_history=False); pipe.unload_lora(pipe.dit, lora_state_dict=self.lora_sd)
        else:
            SI.set_pipe_units(pipe, hist_frames=hist, use_history=True)
            if self.lora_sd is not None:
                pipe.load_lora(pipe.dit, lora_state_dict=self.lora_sd)
        # ---- F' replan: the new frames' targets start from the measured end pose of the previous round
        replan = None
        if r > 0 and arm['replan']:
            M = st.meas[prev_end - 1]; E = self.plan[prev_end - 1]
            if arm['name']=='sham':M=E.copy()
            if FP.valid_pose(M):
                planned = self.plan[prev_end:end]
                st.cmd[prev_end:end] = planned.copy() if np.array_equal(M,E) else (FP.rejoin_offset(M, planned, E) if self.a.replan_mode == 'offset' else FP.replan_se3(M, planned, E))
                if arm['name']!='sham' and arm['write']=='commanded':
                    st.cmd[prev_end:end]=(st.cmd[prev_end-1]@np.linalg.inv(M))@st.cmd[prev_end:end]
                replan = dict(kind='replan', round=r, frame=prev_end, mode=self.a.replan_mode, frames=end - prev_end,
                              position_error=float(np.linalg.norm(M[:3, 3] - E[:3, 3])), rotation_error_deg=FP.geodesic_deg(M, E))
            else:
                raise RuntimeError('Cannot apply planned feedback: invalid DA3 endpoint')
            st.events.append(replan)
        # ---- F' write: history pose labels (frames < prev_end)
        if r > 0 and arm['write'] == 'measured':
            labels, src = FP.history_labels(st.meas[:prev_end], st.cmd[:prev_end], self.plan[0])
        else:
            labels = st.cmd[:prev_end].copy(); labels[0] = self.plan[0]; src = ['anchor'] + ['commanded'] * (prev_end - 1)
        st.labels[:prev_end] = labels; st.label_src[:prev_end] = src
        # ---- MapAnything on the history (inference.py:495-504; map_extrinsics = history labels)
        t = time.time()
        map_frames = st.frames[::s.map_stride]
        map_ext = self.to_t(labels, self.D if r > 0 else 1.0)[::s.map_stride]   # round 0: frame 0 only (zero translation)
        map_out = SI.run_mapanything(args=s, round_idx=r, map_frames=map_frames, map_extrinsics=map_ext,
                                     intrinsic_path=self.intrinsic_path, work_dir=work_dir)
        tm['mapanything'] = time.time() - t
        if r == 0:
            self.set_scale(map_out)
        # ---- round extrinsics (inference.py:510-516): history part = labels, new part = commanded (replanned)
        seg = np.concatenate([labels[start:prev_end], st.cmd[prev_end:end]], 0)
        round_extrinsic = self.to_t(seg, self.D)
        if round_extrinsic.shape[0] < target:
            round_extrinsic = torch.cat([round_extrinsic, round_extrinsic[-1:].repeat(padding, 1, 1)], dim=0)
        t = time.time()
        point_video_path, point_score_path = SI.run_render(args=s, round_idx=r, points_path=map_out['points_path'],
                                                           map_video_path=map_out['map_video_path'], round_extrinsic=round_extrinsic,
                                                           intrinsic=torch.from_numpy(self.intrinsic), work_dir=work_dir)
        tm['render'] = time.time() - t
        # ---- inference.py:528-563
        round_hist_frames = st.frames[-hist:]; round_ref_frames = st.frames[:-hist]
        round_hist_latents_num = max(0, (len(round_hist_frames) - 1) // 4 + 1)
        input_point = SI.load_video_frames(point_video_path, s.height, s.width)
        input_point_score = SI.load_video_frames(point_score_path, s.height, s.width)
        input_point_score = [255 - np.array(frame) for frame in input_point_score]
        t = time.time()
        if r == 0:
            input_video, input_image, selected, ar_hist = None, round_hist_frames[0], None, 0
        else:
            input_video, input_image = round_hist_frames, None
            all_exist = self.to_t(labels[:start], self.D).numpy()[::s.ref_hist_stride]          # F' write: labels
            all_ref = round_ref_frames[::s.ref_hist_stride]
            if len(all_exist) != len(all_ref):
                raise ValueError(f'history cameras vs frames: {len(all_exist)} vs {len(all_ref)}')
            new_cam = round_extrinsic.cpu().numpy()[::s.ref_stride]
            selected = SI.select_reference_frames(pipe=pipe, points_path=map_out['points_path'], all_exist_cam_w2cs=all_exist,
                                                  new_cam_w2c=new_cam, intrinsic=self.intrinsic, all_ref_frames=all_ref, args=s,
                                                  save_dir=work_dir / f'round_{r:02d}' / 'selected_ref_frames')
            if s.ref_num == 0 and selected is not None:
                selected = [round_hist_frames[0]]
            ar_hist = round_hist_latents_num
        tm['ref_select'] = time.time() - t
        # ---- denoise (inference.py:565-585)
        timer = StepTimer(torch.cuda.synchronize, label=f"{arm['name'] if r else 'shared'} r{r}")
        t = time.time()
        round_video, round_latents = pipe.call_latent_inference(
            prompt=f'{prompt} 4K, realistic, high quality, HDR.', negative_prompt=SI.NEGATIVE_PROMPT, seed=s.seed, tiled=False,
            height=s.height, width=s.width, input_video=input_video, input_image=input_image, control_video=input_point,
            control_score=input_point_score, ref_images=selected, ar_hist_latents_num=ar_hist, vace_scale=s.vace_scale,
            num_frames=target, num_inference_steps=s.num_inference_steps, cfg_scale=s.cfg_scale, sigma_shift=s.sigma_shift,
            sampler=s.sampler, return_latents=True, progress_bar_cmd=timer)
        tm['call'] = time.time() - t; tm['steps_loop'] = float(sum(timer.times)); tm['steps'] = len(timer.times)
        SI.save_video(round_video, str(work_dir / f'round_{r:02d}' / 'generated_round.mp4'), fps=s.fps, quality=5)   # inference.py:587-588
        # ---- history update (inference.py:590-598)
        new_frames = round_video[len(round_hist_frames):(-padding if padding > 0 else None)]
        st.frames.extend(new_frames)
        st.latents.append((round_latents[:, :, round_hist_latents_num:] if input_video is not None else round_latents).detach().cpu())
        if len(st.frames) != end:
            raise RuntimeError(f'frame bookkeeping: {len(st.frames)} != {end}')
        # ---- F' measurement of this round's new frames (DA3, anchor gauge)
        t = time.time()
        if r == 0:
            st.mframes = [self.small(st.frames[0])]
        st.mframes.extend(self.small(f) for f in new_frames)
        query = sorted(set(range(prev_end, end, self.a.meas_stride)) | {end - 1})
        try:                                    # non-fatal: logging for the baseline; a replan arm then logs replan_skipped
            meas, rep = self.da3.measure(st.mframes[0], st.mframes[1:], self.K_meas, query, self.adepth, self.a.meas_max_views, self.a.meas_tail)
        except Exception as e:
            traceback.print_exc(); meas = {}; rep = dict(sec=None, calls=[], peak_gib=None, error=f'{type(e).__name__}: {e}')
            st.events.append(dict(kind='measure_failed', round=r, error=rep['error']))
        for q, p in meas.items():
            st.meas[q] = p
        tm['measure'] = time.time() - t
        res_t = [None if st.meas[j] is None else float(np.linalg.norm(st.meas[j][:3, 3] - st.cmd[j][:3, 3])) for j in range(prev_end, end)]
        res_r = [None if st.meas[j] is None else FP.geodesic_deg(st.meas[j], st.cmd[j]) for j in range(prev_end, end)]
        tm['total'] = time.time() - t_round
        est = tm['total'] - tm['steps_loop'] + self.a.estimate_steps * tm['steps_loop'] / max(tm['steps'], 1)
        ev = dict(kind='round', round=r, frames=[start, end], new=[prev_end, end], target=target, padding=padding, prompt=prompt,
                  refs=0 if selected is None else len(selected), residual=res_t, residual_deg=res_r,
                  residual_end=res_t[-1], residual_end_deg=res_r[-1],
                  residual_mean=None if not [x for x in res_t if x is not None] else float(np.mean([x for x in res_t if x is not None])),
                  label_sources={k: src.count(k) for k in set(src)}, time={k: round(v, 2) if isinstance(v, float) else v for k, v in tm.items()},
                  estimate_seconds_at_steps=round(est, 1), gpu_peak_mib=self.smi.window(),
                  torch_peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), da3=dict(sec=rep.get('sec'), calls=rep.get('calls'), peak_gib=rep.get('peak_gib'), error=rep.get('error')))
        st.events.append(ev)
        st.rounds.append(dict(round=r, start=start, end=end, target=target, padding=padding, new=[prev_end, end], prompt=prompt,
                              correction=replan, render_c2w=np.concatenate([seg, np.repeat(seg[-1:], padding, 0)]).tolist()))
        st.seconds += tm['total']; st.est += est
        print(f"[F'] round {r} done {tm['total']:.0f}s  residual_end={ev['residual_end']}  deg={ev['residual_end_deg']}", flush=True)

    def final_labels(self, st, arm):
        """history_label per frame: the label Spatia actually used for frames that served as history (< the last round's
        history end); for the last round's own frames (never history) the arm's rule, with source 'unused:<rule>'."""
        h = st.rounds[-1]['new'][0] if st.rounds else 1
        if arm['write'] == 'measured':
            lab, src = FP.history_labels(st.meas, st.cmd, self.plan[0])
        else:
            lab = st.cmd.copy(); lab[0] = self.plan[0]; src = ['anchor'] + ['commanded'] * (self.F - 1)
        lab[:h] = st.labels[:h]; src = list(st.label_src[:h]) + ['unused:' + x for x in src[h:]]
        return lab, src

    def finalize(self, st):
        """inference.py:600-607 (joint final decode of all kept latents)."""
        pipe, torch = self.pipe, self.torch
        t = time.time()
        pipe.load_models_to_device(['vae'])
        lat = torch.cat([l.to(pipe.device) for l in st.latents], dim=2)
        video = pipe.vae.decode(lat, device=pipe.device, tiled=False)
        video = pipe.vae_output_to_video(video)
        pipe.load_models_to_device([])
        frames = np.stack([np.asarray(f.convert('RGB')) for f in video[:self.F]])
        if frames.shape != (self.F, self.H, self.W, 3):
            raise RuntimeError(f'final video {frames.shape} != {(self.F, self.H, self.W, 3)}')
        return frames, time.time() - t


# ---------------------------------------------------------------------------------------------------------- main
def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--batch', type=Path, required=True); p.add_argument('--scene', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arms', default='baseline,write_measured,replan_commanded,replan_measured')
    p.add_argument('--seed', type=int, default=20917)                            # inference.py:650
    p.add_argument('--num_frames', type=int, default=161, help='poses used = output frames incl. frame 0 (Spatia max_frames)')
    p.add_argument('--first_round_frames', type=int, default=121); p.add_argument('--round_frames', type=int, default=81)
    p.add_argument('--hist_frames', type=int, default=9); p.add_argument('--num_inference_steps', type=int, default=40)
    p.add_argument('--cfg_scale', type=float, default=3.5); p.add_argument('--sampler', default='uni_pc')
    p.add_argument('--width', type=int, default=1248); p.add_argument('--height', type=int, default=704)
    p.add_argument('--replan_mode', default='offset', choices=['offset', 'se3'], help='offset = closed_loop.rejoin_offset; se3 = VMem/Memorizon harness')
    p.add_argument('--scale_mode', default='mapanything', choices=['mapanything', 'fixed', 'none'])
    p.add_argument('--units_per_anchor_depth', type=float, default=None)
    p.add_argument('--gauge', default='auto', choices=['auto', 'comp', 'median'])
    p.add_argument('--prompt_mode', default='per_round', choices=['per_round', 'first'])
    p.add_argument('--meas_width', type=int, default=504); p.add_argument('--meas_tail', type=int, default=12)
    p.add_argument('--meas_max_views', type=int, default=20); p.add_argument('--meas_stride', type=int, default=1)
    p.add_argument('--da3_python', default=DA3_PY); p.add_argument('--work_root', type=Path, default=None)
    p.add_argument('--min_free_gib', type=float, default=10.0); p.add_argument('--prune_work', action='store_true',
                   help="after an arm's npz+json are written, delete the point clouds (*.ply) in that arm's own work dir")
    p.add_argument('--estimate_steps', type=int, default=40, help='steps used for the time estimate written to the jsons')
    p.add_argument('--dry_run', action='store_true', help='CPU only: parse the batch, plan rounds, write + re-read the camera files')
    a = p.parse_args()
    a.arm_list = [ARMS[n.strip()] for n in a.arms.split(',') if n.strip()]
    for k in ('first_round_frames', 'round_frames', 'hist_frames'):
        if getattr(a, k) % 4 != 1:
            p.error(f'--{k} must be 4k+1 (Wan VAE)')
    if a.round_frames <= a.hist_frames:
        p.error('--round_frames must exceed --hist_frames')
    if a.scale_mode == 'fixed' and not a.units_per_anchor_depth:
        p.error('--scale_mode fixed needs --units_per_anchor_depth')
    if a.meas_tail < 1 or a.meas_max_views < a.meas_tail + 1:
        p.error('--meas_max_views must be >= --meas_tail + 1')
    a.out = a.out.resolve(); a.batch = a.batch.resolve()
    a.work_root = (a.work_root or a.out / '_work').resolve()
    a.official = dict(width=1248, height=704, first_round_frames=121, round_frames=81, hist_frames=9, num_inference_steps=40,
                      cfg_scale=3.5, sampler='uni_pc')
    a.non_official = {k: getattr(a, k) for k, v in a.official.items() if getattr(a, k) != v}
    return a


def main():
    a = parse()
    os.environ["SPATIA_DETERMINISTIC_RASTER"] = "1"
    sc = load_scene(a.batch, a.scene, a.num_frames)
    tag = f'{a.scene}_s{a.seed}'
    done = lambda arm: (a.out / arm['name'] / f'{tag}.json').exists() and (a.out / arm['name'] / f'{tag}.mp4').exists()
    todo = [arm for arm in a.arm_list if not done(arm)]
    os.chdir(SPATIA); sys.path.insert(0, str(SPATIA))
    import torch
    import inference as SI
    sargs = spatia_args(SI, a)
    sargs.verbose_subprocess = True  # Preserve MapAnything/render errors in the queue log.
    R = SI.compute_total_rounds(a.num_frames, sargs.first_round_frames, sargs.round_frames, sargs.hist_frames)
    ranges = [SI.get_round_range(r, a.num_frames, sargs.first_round_frames, sargs.round_frames, sargs.hist_frames) for r in range(R)]
    stamp = time.strftime('%Y%m%d-%H%M%S')
    if a.dry_run:
        work = a.work_root / f'{tag}_dryrun_{stamp}'; work.mkdir(parents=True, exist_ok=False)
        D = a.units_per_anchor_depth or 1.0
        FP.write_w2c_txt(work / 'camera_w2c_spatia.txt', FP.c2w_to_spatia_w2c(sc['plan'], D))
        ext = SI.read_w2cs_from_txt(work / 'camera_w2c_spatia.txt')
        back = FP.spatia_w2c_to_c2w(ext.astype(np.float64), D)
        FP.write_intrinsics_txt(work / 'camera_intrinsics_spatia.txt', sc['kn'])
        kn_back = SI.read_intrinsics_from_txt(work / 'camera_intrinsics_spatia.txt')
        _, gmode = gauge_depth(a.gauge, sc)
        print(json.dumps(dict(scene=a.scene, todo=[x['name'] for x in todo], rounds=ranges, correction_points=R - 1,
                              correction_before_frames=[rg[0] + sargs.hist_frames for rg in ranges[1:]],
                              prompts=[FP.caption_for_frame(sc['captions'], rg[0] if a.prompt_mode == 'per_round' else 0, sc['default_prompt'])[:80] for rg in ranges],
                              kn=sc['kn'], kn_readback=kn_back, K_out=FP.pixel_K(sc['kn'], sargs.width, sargs.height),
                              txt_roundtrip_max_err=float(np.abs(back - sc['plan']).max()), D_assumed=D, gauge=gmode, notes=sc['notes'],
                              strides=dict(map=sargs.map_stride, ref=sargs.ref_stride, ref_hist=sargs.ref_hist_stride),
                              plan_path_length=float(np.linalg.norm(np.diff(sc['plan'][:, :3, 3], axis=0), axis=1).sum()),
                              non_official=a.non_official, work=work), default=jdefault, indent=1))
        return 0
    if not todo:
        print(f"[F'] {tag}: all arms done, nothing to do", flush=True); return 0
    if free_gib() < a.min_free_gib:
        print(f"[F'] disk {free_gib():.1f} GiB < {a.min_free_gib} -> stop", flush=True); return 3
    work = a.work_root / f'{tag}_{stamp}'; work.mkdir(parents=True, exist_ok=False)
    ipc = Path('/dev/shm') / f'spatia_fprime_{os.getpid()}' if Path('/dev/shm').is_dir() else work / 'ipc'
    ipc.mkdir(parents=True, exist_ok=True)
    smi = SmiPeak(); smi.start()
    proc = dict(scene=a.scene, seed=a.seed, batch=str(a.batch), work=str(work), arms={}, started=stamp, non_official=a.non_official)
    (a.out / '_proc').mkdir(parents=True, exist_ok=True)
    da3 = None; rc = 0
    try:
        t0 = time.time()
        da3 = DA3Client(a.da3_python, work / 'da3_worker.log', ipc)                 # loads on CPU while the pipe loads
        pipe, lora_sd = SI.build_pipe(vace_path=Path(sargs.vace_path), lora_path=Path(sargs.lora_path), hist_frames=sargs.hist_frames)
        proc['da3_ready'] = da3.wait_ready(); proc['startup_seconds'] = round(time.time() - t0, 1)
        if proc['da3_ready'].get('fake'):
            raise RuntimeError('GV_DA3_FAKE=1 is set: the DA3 worker returns fake poses (protocol test only) -> refusing a real run')
        run = Runner(a, sc, SI, torch, pipe, lora_sd, sargs, da3, smi, work)
        # ---- shared round 0
        st0 = State(a.num_frames); st0.cmd = sc['plan'].copy(); st0.labels = sc['plan'].copy()
        st0.frames = [sc['image'].resize((sargs.width, sargs.height)) if sc['image'].size != (sargs.width, sargs.height) else sc['image']]
        shared = work / 'shared'; shared.mkdir()
        run.do_round(st0, 0, ARMS['baseline'], shared)
        proc.update(shared_round0_seconds=round(st0.seconds, 1), shared_round0_estimate=round(st0.est, 1), D=run.D, scale=run.scale_info, gauge=run.gauge)
        atomic_json(a.out / '_proc' / f'{tag}.json', proc)
        for arm in todo:
            if free_gib() < a.min_free_gib:
                proc['arms'][arm['name']] = dict(status='skipped_disk', free_gib=round(free_gib(), 1)); rc = 3; break
            ta = time.time(); st = st0.branch(); arm_work = work / arm['name']; arm_work.mkdir()
            try:
                for r in range(1, R):
                    run.do_round(st, r, arm, arm_work)
                frames, t_dec = run.finalize(st)
                dst = a.out / arm['name'] / f'{tag}.mp4'; dst.parent.mkdir(parents=True, exist_ok=True)
                frame0 = np.asarray(st0.frames[0].convert('RGB'))                          # processed start image (the model's input)
                t = time.time(); tmp = dst.with_name(dst.stem + '.tmp.mp4')
                fc.write_mp4(tmp, [frame0] + list(frames[1:])); t_save = time.time() - t
                arm_sec = time.time() - ta; arm_est = st.est + t_dec + t_save           # a branch starts at est = seconds = 0
                labels_final, src_final = run.final_labels(st, arm)
                st.events.append(dict(kind='final', decode_seconds=round(t_dec, 1), save_seconds=round(t_save, 1),
                                      mp4_mib=round(tmp.stat().st_size / 2**20, 1), gpu_peak_process_mib=smi.peak))
                ends = [e for e in st.events if e['kind'] == 'round']
                log = dict(arm=arm, scene=a.scene, seed=a.seed, seconds=round(st0.seconds + arm_sec, 1),
                           seconds_shared_round0=round(st0.seconds, 1), seconds_arm=round(arm_sec, 1), seconds_startup=proc['startup_seconds'],
                           estimate_seconds_standalone=round(st0.est + arm_est + proc['startup_seconds'], 1),
                           estimate_steps=a.estimate_steps,
                           planned=sc['plan'].tolist(), commanded=st.cmd.tolist(),
                           measured=[None if m is None else np.asarray(m).tolist() for m in st.meas],
                           history_label=labels_final.tolist(), label_source=src_final,
                           residual=[None if st.meas[j] is None else float(np.linalg.norm(st.meas[j][:3, 3] - st.cmd[j][:3, 3])) for j in range(a.num_frames)],
                           residual_deg=[None if st.meas[j] is None else FP.geodesic_deg(st.meas[j], st.cmd[j]) for j in range(a.num_frames)],
                           pose_convention='OpenCV c2w, anchor-relative (frame 0 = identity), translation in anchor-median-depth units (batch gauge); '
                                           'Spatia w2c = inv(c2w with translation x spatia_units_per_anchor_depth)',
                           spatia_units_per_anchor_depth=run.D, scale=run.scale_info, gauge=run.gauge,
                           K=run.K_out.tolist(), K_norm=list(map(float, sc['kn'])), width=run.W, height=run.H,
                           frames_shape=list(frames.shape), frame0='mp4 frame 0 = the processed start image (PIL stretch to WxH), not Spatia decode',
                           model='spatia', preprocess='spatia', K_out=run.K_out.tolist(), out_hw=[run.H, run.W], frames_in_mp4=int(len(frames)),
                           mp4=dict(codec='libx264', crf=12, preset='slow', pix_fmt='yuv420p', fps=16),
                           rounds=st.rounds, correction_points=R - 1, correction_before_frames=[rg[0] + sargs.hist_frames for rg in ranges[1:]],
                           events=st.events, gpu_peak_mib=max([e.get('gpu_peak_mib', 0) for e in ends] + [0]),
                           torch_peak_gib=max([e.get('torch_peak_gib', 0) for e in ends] + [0]),
                           settings=dict(batch=str(a.batch), image_size_batch=list(sc['image'].size), num_frames=a.num_frames,
                                         spatia={k: getattr(sargs, k) for k in ('width', 'height', 'max_frames', 'first_round_frames', 'round_frames',
                                                 'hist_frames', 'fps', 'map_fps', 'ref_fps', 'ref_hist_fps', 'ref_num', 'num_inference_steps',
                                                 'cfg_scale', 'sigma_shift', 'sampler', 'vace_scale', 'seed', 'map_conf_percentile', 'map_voxel_size',
                                                 'render_voxel_size', 'render_voxel_size_step', 'map_stride', 'ref_stride', 'ref_hist_stride',
                                                 'vace_path', 'lora_path')},
                                         replan_mode=a.replan_mode, scale_mode=a.scale_mode, gauge=a.gauge, prompt_mode=a.prompt_mode,
                                         meas=dict(width=run.mw, height=run.mh, tail=a.meas_tail, max_views=a.meas_max_views, stride=a.meas_stride,
                                                   K=run.K_meas.tolist(), worker_python=a.da3_python),
                                         non_official=a.non_official, work=str(work), scene_notes=sc['notes']),
                           deviations=DEVIATIONS)
                atomic_json(dst.with_suffix('.json'), log); os.replace(tmp, dst)          # mp4 last: marks the run finished
                proc['arms'][arm['name']] = dict(status='ok', seconds=log['seconds'], seconds_arm=log['seconds_arm'],
                                                 estimate_arm=round(arm_est, 1), residual_end=[e['residual_end'] for e in ends])
                print('DONE ' + json.dumps(dict(arm=arm['name'], scene=a.scene, seed=a.seed, sec=log['seconds'], sec_arm=log['seconds_arm'],
                                                residual_end=[None if e['residual_end'] is None else round(e['residual_end'], 3) for e in ends],
                                                gpu_peak_mib=log['gpu_peak_mib'], mp4_mib=st.events[-1]['mp4_mib'])), flush=True)
                if a.prune_work:
                    for ply in arm_work.rglob('*.ply'):
                        ply.unlink()
            except Exception as e:
                rc = 1; traceback.print_exc()
                err = dict(arm=arm, scene=a.scene, seed=a.seed, error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc(), events=st.events)
                (a.out / arm['name']).mkdir(parents=True, exist_ok=True)
                atomic_json(a.out / arm['name'] / f'{tag}.error.json', err)
                proc['arms'][arm['name']] = dict(status='error', error=err['error'])
                torch.cuda.empty_cache()
            atomic_json(a.out / '_proc' / f'{tag}.json', proc)
        if a.prune_work:
            for ply in shared.rglob('*.ply'):
                ply.unlink()
        est = dict(startup=proc['startup_seconds'], shared_round0=proc['shared_round0_estimate'],
                   arms={k: v.get('estimate_arm') for k, v in proc['arms'].items() if v.get('status') == 'ok'}, steps=a.estimate_steps)
        proc['estimate'] = est; proc['gpu_peak_process_mib'] = smi.peak
        atomic_json(a.out / '_proc' / f'{tag}.json', proc)
        print('ESTIMATE ' + json.dumps(est), flush=True)
    finally:
        smi.stop = True
        if da3 is not None:
            da3.close()
        shutil.rmtree(ipc, ignore_errors=True)                                    # this process's own IPC files only
    return rc


if __name__ == '__main__':
    sys.exit(main())
