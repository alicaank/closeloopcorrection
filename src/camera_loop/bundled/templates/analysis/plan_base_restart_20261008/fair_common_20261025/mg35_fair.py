"""Matrix-Game-3.5 (first-person base model) on the fair common set: 10 videos, 160 frames, seed 7, official settings.

Inputs follow the official infer.py interface exactly (anchor image + camera.npz + prompt); only these choices are ours:
- anchor = the batch 864x480 frame (what every model gets); MG3.5's dataset stretches it to 1280x704 (bilinear antialias,
  no crop because 864x480 is smaller) -> fair_common.preprocess('matrixgame').
- camera: our planned c2w per frame (anchor-relative OpenCV) with translation x metres_per_anchor_unit (MG3.5 expects metric
  translation; the same dataset metric scale Astronex uses), intrinsics = the true 864x480 pixel K (MG3.5 rescales to 1280x704).
  Poses 0..168 (2 blocks x 84 frames) come from the batch's overrun path (frames 161..168 are the real continuation).
- frame/pose mapping (read from code): history latents = anchor latent + 21 latents per block, decoded to 1 + 84*blocks frames;
  read_camera_params gives section k the frames G..G+83 with G = 1 + 84k -> output frame f <-> pose f. Frames 1..160 are scored.
- output: the raw decoded history frames (mg35_launch.py hooks: no burned-in frame counter, lossless npy) -> fair mp4 crf 12.
- VAE decode tiled (official flag --vae_decode_tiled) so the 169-frame 1280x704 decode fits 32 GB; generation unchanged.
usage (venv_mg35 python): mg35_fair.py --out RUNS/matrixgame [--videos a b] [--seed 7] [--extra ...]"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import fair_common as fc
REPO = Path('@WORKSPACE@/sota_repos/matrix_game/Matrix-Game-3.5'); sys.path.insert(0, str(REPO))
import infer as MG                                                                  # official helpers (workspace, index, outputs)
N_POSES = 1 + 84 * 2


def run_generation(args, index_path, tmp_root, env):
    """= infer.run_generation with the script swapped for mg35_launch.py (same argv to run_pipeline.py)."""
    ckpt = MG.resolve_weight('ckpt_first', args.ckpt); wan = MG.resolve_weight('wan', args.wan_dir)
    cmd = [sys.executable, '-m', 'accelerate.commands.launch', '--num_processes', '1', '--num_machines', '1', '--machine_rank', '0',
           '--mixed_precision', 'bf16', '--main_process_port', str(MG.free_port()), str(HERE / 'mg35_launch.py'),
           '--config', str(REPO / 'configs/infer_first_person.yaml'),
           '--model_id_with_origin_paths', f'{wan}:diffusion_pytorch_model*.safetensors,{wan}:models_t5_umt5-xxl-enc-bf16.pth,{wan}:Wan2.2_VAE.pth',
           '--tokenizer_path', MG.resolve_weight('tokenizer', args.tokenizer_dir), '--trained_dit', ckpt,
           '--dataset_index_path', index_path, '--val_dataset_index_path', index_path, '--num_val_batches', '1',
           '--num_validation_blocks', str(args.num_blocks), '--validation_sample_offset', '0',
           '--output_path', os.path.join(tmp_root, 'raw'), '--log_dir_name', 'run', '--validation_seed', str(args.seed)] + args.extra
    t0 = time.time(); subprocess.run(cmd, cwd=REPO, env=env, check=True)
    return os.path.join(tmp_root, 'raw', 'run', 'validation'), time.time() - t0


def one(name, out, seed, extra):
    dst = out / 'baseline' / f'{name}_s{seed}.mp4'
    if dst.exists():
        print('skip', dst, flush=True); return
    sc = fc.scene(name); mpu = float(sc['metres_per_anchor_unit'])
    c2w = np.array(sc['c2w_ext'][:N_POSES], np.float64); assert len(c2w) == N_POSES, len(c2w)
    c2w[:, :3, 3] *= mpu
    K = sc['K']; intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32)
    work = out / '_work' / name; shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    np.savez(work / 'camera.npz', extrinsics_c2w=c2w.astype(np.float32), intrinsics=intr)
    args = SimpleNamespace(person='first', image=sc['image_path'], camera=str(work / 'camera.npz'), prompt=sc['caption'], prompt_file='',
                           caption='', refs='', num_blocks=2, steps=0, cfg_scale=0.0, seed=seed, output=str(work / 'out'), name=name,
                           camera_convention='c2w', keep_workspace=False, ckpt='', wan_dir='', tokenizer_dir='', da3_dir='', extra=list(extra))
    env = dict(os.environ); env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', ''); env['MG35_REPO'] = str(REPO)
    env['DA3_MODEL_PATH'] = MG.resolve_weight('da3', ''); env.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'true')
    os.chdir(REPO)
    out_root, tmp_root, ws = MG.build_workspace(args)
    index_path = MG.build_index(ws, sys.executable, env)
    val_dir, sec = run_generation(args, index_path, tmp_root, env)
    raw = sorted(Path(val_dir).glob('*_history_raw.npy')); assert len(raw) == 1, raw
    fr = np.load(raw[0]); assert fr.shape[0] >= fc.N_FRAMES + 1 and fr.shape[1:3] == (704, 1280), fr.shape
    frame0 = fc.preprocess('matrixgame', sc['anchor'])
    K_out = fc.scale_crop_K(K, 1280 / 864, 704 / 480)
    log = dict(model='matrixgame', arm=dict(name='baseline'), scene=name, seed=seed, seconds=sec, n_decoded=int(fr.shape[0]),
               metres_per_anchor_unit=mpu, prompt=sc['caption'], checkpoint='RiemannDynamics/Matrix-Game-3.5-Base first-person.safetensors',
               settings=dict(config='configs/infer_first_person.yaml (25 steps, cfg 5.0, 704x1280)', num_blocks=2, extra=list(extra)),
               planned=fc.planned_c2w(name)[:fc.N_FRAMES + 1].tolist(), pose_frame_ids=list(range(fc.N_FRAMES + 1)),
               frame_pose_mapping='output frame f <-> pose f (anchor latent + 21 latents/block, section k frames 1+84k..84(k+1))',
               deviations=['anchor = batch 864x480 frame (stretched to 1280x704 by MG3.5)', 'translation x metres_per_anchor_unit',
                           'raw decoded frames captured before the frame-counter overlay (mg35_launch.py)', 'tiled VAE decode'])
    fc.save_run(dst, [fr[f] for f in range(1, fc.N_FRAMES + 1)], frame0, K_out, log, preprocess='matrixgame')
    shutil.rmtree(tmp_root, ignore_errors=True); shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(dict(scene=name, sec=round(sec, 1), frames=int(fr.shape[0]), out=str(dst))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--videos', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--min_free_gib', type=float, default=10.)
    ap.add_argument('--extra', nargs='*', default=['--vae_decode_tiled'])
    a = ap.parse_args(); out = a.out.resolve()
    names = a.videos or [s['name'] for s in fc.index()['scenes']]
    for n in names:
        if fc.free_gib() < a.min_free_gib:
            sys.exit(f'disk under {a.min_free_gib} GiB, stop before {n}')
        one(n, out, a.seed, a.extra)
