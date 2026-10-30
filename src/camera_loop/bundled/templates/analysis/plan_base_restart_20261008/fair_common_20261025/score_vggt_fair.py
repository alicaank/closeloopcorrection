"""Camera (VGGT-1B) + fidelity (DINO / LPIPS vs the recording) on the fair external runs, scored like our c10_* runs.

Frame indices (identical for every model and for our runs):
  VGGT      : frame 0 + the 40 latent-end frames 4, 8, ..., 160, exactly the views closed_loop_864.py --vggt 1 uses for ours
              (functions of check_metrics_864.py copied verbatim below): poses relative to frame 0; ONE scale from VGGT's frame-0
              depth vs the gauge depth of frame 0 (= DA3 metric depth of frame 0 with the run's K / metres_per_anchor_unit,
              renderer864.da3_gauge_depths); cam_errors per 16-frame call vs the planned path (batch_ours10 = the recorded path):
              pos_err (mean over the call's 4 latent ends, anchor units), exec_ratio, rot_err_deg (call end). Headline = call 10:
              rot_err at frame 160, pos_err_call10, and pos_err_160 = |t_VGGT(160) - t_plan(160)| (not available for c10: eval.json
              keeps only the per-call values).
  DINO/LPIPS: frames 1, 5, 9, ..., 157 (the only generated frames our c10 runs store), 4 per call (1+16c, 5+16c, 9+16c, 13+16c),
              DINOv2-S cosine and LPIPS-alex (evaluate_recorded.metrics_models, the c10 functions) against the RECORDED frame of the
              same index passed through the model's own first-frame preprocessing (fair_common.preprocess: crop / resize /
              stretch), so generated and reference frames share one geometry. For our runs (--c10) the reference is the raw
              864x480 recording (our model's input geometry). NB: closed_loop_864's own per-call dino/lpips use frames 1, 3, ..., 15
              of each call; the --c10 rows recompute them at the shared indices.
Inputs: fair runs <root>/<arm>/<name>_s<seed>.mp4 + .json (fair_common.save_run, frame 0 = processed input), Lyra-2 runs
<root>/<arm>/<name>_frames/frame_%05d.png (KIND lyra; fair_common.load_lyra). VGGT input keeps each model's aspect: 518 wide,
height round(518 h / w / 14) * 14 = 294 for every 16:9-like output (identical to c10's fixed 294x518), 518 for VMem's 576x576.
Usage (cd fresh_teacher_20261021; source env.sh):
  score_vggt_fair.py OUT.json KIND ROOT [ARM ...]       KIND = astronex | memorizon | worldplay | vmem | spatia | lyra
  score_vggt_fair.py OUT.json --c10 RUN_DIR [...]       DINO/LPIPS of our c10 runs at the shared indices (VGGT from their eval.json)
  score_vggt_fair.py OUT.json --table [C10_RUN_DIR ...] summary (+ the c10 runs' eval.json VGGT numbers)"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fair_common as fc

CALLS = 10
FID_IDS = fc.WS_IDS                                                                   # 1, 5, ..., 157


def cam_errors(M, P, mpu, calls):
    """verbatim check_metrics_864.cam_errors."""
    per = []
    for c in range(calls):
        ends = list(range(4 * c + 1, 4 * c + 5)); pe = [float(np.linalg.norm(M[j][:3, 3] / mpu - P[4 * j][:3, 3])) for j in ends]
        a_ = M[4 * c] if c else np.eye(4); b_ = M[4 * c + 4]
        ratio = float(np.linalg.norm((b_[:3, 3] - a_[:3, 3]) / mpu) / max(np.linalg.norm(P[16 * c + 16][:3, 3] - P[16 * c][:3, 3]), 1e-6))
        rot = float(np.degrees(Rotation.from_matrix(b_[:3, :3].T @ P[16 * c + 16][:3, :3]).magnitude()))
        per.append(dict(call=c + 1, pos_err=float(np.mean(pe)), exec_ratio=ratio, rot_err_deg=rot))
    return per


@torch.no_grad()
def vggt_relposes(model, views, gauge_depth0, size=(294, 518)):
    """check_metrics_864.vggt_relposes with the input size as a parameter (c10: fixed (294, 518))."""
    sys.path.insert(0, '@WORKSPACE@/VGGT-World')
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = torch.from_numpy(np.stack(views)).float().permute(0, 3, 1, 2).cuda() / 255.
    x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model.forward_vggt(x[None])
    ext, _ = pose_encoding_to_extri_intri(pred['pose_enc'], x.shape[-2:])
    w2c = torch.eye(4, dtype=torch.float64).repeat(len(views), 1, 1); w2c[:, :3, :] = ext[0].double().cpu()
    rel = (w2c[0] @ torch.linalg.inv(w2c)).numpy()
    d0 = pred['depth'][0, 0, :, :, 0].float()
    ours = F.interpolate(torch.as_tensor(gauge_depth0).float()[None, None].cuda(), size=d0.shape, mode='bilinear', align_corners=False)[0, 0]
    ok = (d0 > 0) & (ours > 0) & torch.isfinite(d0) & torch.isfinite(ours); s = float(torch.median(ours[ok] / d0[ok]))
    rel[:, :3, 3] *= s
    return rel, s


def da3_gauge_depth0(da3, frame0, K, mpu):
    """renderer864.da3_gauge_depths for one view (single view -> DA3's own metric depth, no extrinsics), / metres per unit."""
    pred = da3.inference(image=[frame0], intrinsics=np.asarray(K, np.float32)[None], extrinsics=None, align_to_input_extrinsics=False,
                         align_to_input_ext_scale=False, infer_gs=False, process_res=max(frame0.shape[:2]), export_dir=None)
    d = torch.as_tensor(np.asarray(pred.depth[0], np.float32))
    if d.shape != frame0.shape[:2]:
        d = F.interpolate(d[None, None], frame0.shape[:2], mode='bilinear', align_corners=False)[0, 0]
    return (d / mpu).numpy()


def vggt_size(hw):
    return (int(round(518 * hw[0] / hw[1] / 14)) * 14, 518)


REC = {}


def recorded(name, j):
    if (name, j) not in REC:
        s = fc.index_scene(name)
        REC[(name, j)] = np.array(Image.open(Path(s['frames_dir']) / f"frame_{s['original_start'] + j + 1:05d}.jpg").convert('RGB'))
    return REC[(name, j)]


def fidelity(emb, lp, gen, ref):
    """gen / ref: {frame index: uint8} at the same size -> per-call DINO / LPIPS over the call's 4 shared indices."""
    out = dict(dino_call=[], lpips_call=[])
    for c in range(CALLS):
        ids = [j for j in FID_IDS if 16 * c < j <= 16 * c + 16]
        g = np.stack([gen[j] for j in ids]); r = np.stack([ref[j] for j in ids])
        out['dino_call'].append(float((emb(g) * emb(r)).sum(-1).mean())); out['lpips_call'].append(float(lp(g, r).mean()))
    return out


def summarize(res, out, c10_dirs=()):
    rows = {k: {s: dict(x) for s, x in v.items()} for k, v in res.items()}
    for d in map(Path, c10_dirs):                                                       # our runs: closed_loop_864's own VGGT block
        ev = json.load(open(d / 'eval.json')); tag = 'ours:' + d.name; rows.setdefault(tag, {})
        for s in ev['scenes']:
            if 'vggt' in s:
                rows[tag].setdefault(s['stream'], {}).update(per_call=s['vggt'], vggt_scale=s.get('vggt_scale'), pos_err_160=None)
    summary = {}
    for tag, v in rows.items():
        e = dict(n=len(v)); vv = [x for x in v.values() if 'per_call' in x]
        if vv:
            for k in ('pos_err', 'exec_ratio', 'rot_err_deg'):
                e['vggt_' + k] = [round(float(np.mean([x['per_call'][c][k] for x in vv])), 4) for c in range(CALLS)]
            e['pos_err_call10'] = e['vggt_pos_err'][-1]; e['rot_err_160'] = e['vggt_rot_err_deg'][-1]
            p160 = [x['pos_err_160'] for x in vv if x.get('pos_err_160') is not None]
            e['pos_err_160'] = round(float(np.mean(p160)), 4) if p160 else None
        ff = [x for x in v.values() if 'dino_call' in x]
        if ff:
            for k in ('dino', 'lpips'):
                e[k + '_call'] = [round(float(np.mean([x[k + '_call'][c] for x in ff])), 4) for c in range(CALLS)]
                e[k + '_mean'] = round(float(np.mean(e[k + '_call'])), 4); e[k + '_late'] = round(float(np.mean(e[k + '_call'][5:])), 4)
        summary[tag] = e
    paired = {}
    M = {'pos_err_call10': lambda x: x['per_call'][-1]['pos_err'], 'rot_err_160': lambda x: x['per_call'][-1]['rot_err_deg'],
         'pos_err_160': lambda x: x['pos_err_160'], 'pos_err_mean': lambda x: float(np.mean([c['pos_err'] for c in x['per_call']])),
         'dino_mean': lambda x: float(np.mean(x['dino_call'])), 'lpips_mean': lambda x: float(np.mean(x['lpips_call']))}
    for tag, v in res.items():
        kind, arm = tag.split(':', 1); b = res.get(f'{kind}:baseline')
        if arm != 'baseline' and b and kind != 'ours':
            for k, f in M.items():
                d = []
                for s in v:
                    try:
                        d.append(f(v[s]) - f(b[s]))
                    except (KeyError, TypeError):
                        pass
                if d:
                    paired.setdefault(tag, {})['d_' + k] = round(float(np.mean(d)), 4); paired[tag]['d_' + k + '_sign'] = f'{sum(x > 0 for x in d)}+/{sum(x < 0 for x in d)}-'
    json.dump(dict(per_video=res, summary=summary, paired_vs_baseline=paired), open(out.with_name(out.stem + '_summary.json'), 'w'), indent=1)
    print('SUMMARY', json.dumps(summary)); print('PAIRED', json.dumps(paired))


def main():
    out = Path(sys.argv[1]); args = sys.argv[2:]; res = json.load(open(out)) if out.exists() else {}
    if args and args[0] == '--table':
        return summarize(res, out, args[1:])
    from evaluate_recorded import metrics_models                                       # joint_samplerfix (env.sh), the c10 DINO/LPIPS
    emb, lp = metrics_models()
    if args[0] == '--c10':
        for rd in map(Path, args[1:]):
            tag = 'ours:' + rd.name
            for npzf in sorted(rd.glob('frames_*.npz')):
                name = npzf.stem.replace('frames_', '')
                if 'dino_call' in res.get(tag, {}).get(name, {}):
                    continue
                gen = np.load(npzf)['gen']; g = {1 + 4 * i: x for i, x in enumerate(gen)}
                res.setdefault(tag, {}).setdefault(name, {}).update(fidelity(emb, lp, g, {j: recorded(name, j) for j in FID_IDS}))
                json.dump(res, open(out, 'w')); print(json.dumps(dict(run=tag, scene=name, dino=round(float(np.mean(res[tag][name]['dino_call'])), 3))), flush=True)
        return summarize(res, out)
    kind, root = args[0], Path(args[1]); arms = args[2:]
    from realized_pose import load_da3
    from score_vggt import load_vggt                                                   # memorizon_plugins (env.sh PYTHONPATH), VGGT-1B weights
    da3 = load_da3('cuda'); vggt = load_vggt()
    for arm, scene, loader in fc.iter_runs(kind, root):
        tag = f'{kind}:{arm}'
        if (arms and arm not in arms) or scene in res.get(tag, {}):
            continue
        t0 = time.time(); f0, fr, K, hw, log = loader(); P = fc.planned_c2w(scene); mpu = float(fc.index_scene(scene)['metres_per_anchor_unit'])
        pre = log.get('preprocess') or kind
        gd0 = da3_gauge_depth0(da3, f0, K, mpu)
        size = vggt_size(hw); rel, sv = vggt_relposes(vggt, [f0] + [fr[4 * j] for j in range(1, 4 * CALLS + 1)], gd0, size)
        per = cam_errors({j: rel[j] for j in range(1, 4 * CALLS + 1)}, P, 1.0, CALLS)
        ref = {j: fc.preprocess(pre, recorded(scene, j), (hw[1], hw[0])) for j in FID_IDS}
        r = dict(per_call=per, vggt_scale=sv, vggt_size=list(size), pos_err_160=float(np.linalg.norm(rel[40][:3, 3] - P[160][:3, 3])),
                 rot_err_160=per[-1]['rot_err_deg'], vggt_c2w={4 * j: rel[j].tolist() for j in range(0, 4 * CALLS + 1)}, preprocess=pre, hw=list(hw),
                 **fidelity(emb, lp, fr, ref))
        r['sec'] = round(time.time() - t0, 1); res.setdefault(tag, {})[scene] = r; json.dump(res, open(out, 'w'))
        print(json.dumps(dict(run=tag, scene=scene, pos=[round(p['pos_err'], 3) for p in per], ratio=[round(p['exec_ratio'], 2) for p in per],
                              rot160=round(per[-1]['rot_err_deg'], 2), pos160=round(r['pos_err_160'], 3), dino=round(float(np.mean(r['dino_call'])), 3),
                              lpips=round(float(np.mean(r['lpips_call'])), 3), size=size, sec=r['sec'])), flush=True)
    summarize(res, out)


if __name__ == '__main__':
    main()
