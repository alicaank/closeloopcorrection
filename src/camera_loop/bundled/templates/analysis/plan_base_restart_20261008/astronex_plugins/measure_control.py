"""Astronex official-vs-harness control (README audit 10-23): does the model turn by the commanded yaw through its OWN
generate.py, and does our harness get the same motion from the same image/prompt/trajectory?
VGGT-1B on the latent-end frames (RGB 4j), rotation only (scale-free), vs the commanded rotation of the trajectory string
(parse_trajectory: 3 deg per latent frame for 'l'/'j').
Usage (recorded_eval/run_eval_python.sh): measure_control.py <out.json> name:path:traj [...]   (path = .mp4 or harness .npz)"""
import json, sys
import numpy as np, torch, torch.nn.functional as F, imageio.v3 as iio
from scipy.spatial.transform import Rotation as Rr
sys.path.insert(0, '@WORKSPACE@/analysis/plan_base_restart_20261008/memorizon_plugins/calib_20261016'); sys.path.insert(0, '@WORKSPACE@/sota_repos/astronex_world')
from calib_probe import load_vggt
from utils.camera_trajectory import parse_trajectory


@torch.no_grad()
def vggt_rel(model, views):
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = torch.from_numpy(np.stack(views)).float().permute(0, 3, 1, 2).cuda() / 255.
    x = F.interpolate(x, size=(294, 518), mode='bilinear', align_corners=False)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model.forward_vggt(x[None])
    ext, _ = pose_encoding_to_extri_intri(pred['pose_enc'], x.shape[-2:])
    w2c = torch.eye(4, dtype=torch.float64).repeat(len(views), 1, 1); w2c[:, :3, :] = ext[0].double().cpu()
    return (w2c[0] @ torch.linalg.inv(w2c)).numpy()                                         # c2w relative to view 0 (OpenCV)


def main():
    out = sys.argv[1]; model = load_vggt().cuda(); res = {}
    for spec in sys.argv[2:]:
        name, path, traj = spec.split(':', 2)
        if path.endswith('.npz'):
            fr = np.load(path)['frames']; step = 2                                          # harness stores every 2nd RGB frame
        else:
            fr = np.stack(list(iio.imiter(path))); step = 4
        C = np.linalg.inv(parse_trajectory(traj).astype(np.float64))                        # commanded c2w per latent frame
        n = min(len(C), (len(fr) - 1) // step + 1); views = [fr[step * j] for j in range(n)]
        M = vggt_rel(model, views)
        cmd = [Rr.from_matrix(C[j][:3, :3] @ C[0][:3, :3].T) for j in range(n)]; meas = [Rr.from_matrix(M[j][:3, :3]) for j in range(n)]
        ang_c = [float(np.degrees(r.magnitude())) for r in cmd]; ang_m = [float(np.degrees(r.magnitude())) for r in meas]
        rv_c, rv_m = cmd[-1].as_rotvec(), meas[-1].as_rotvec()
        cos = float(rv_c @ rv_m / max(np.linalg.norm(rv_c) * np.linalg.norm(rv_m), 1e-9))
        res[name] = dict(path=path, traj=traj, n_rgb=len(fr), n_lat=n, cmd_deg=ang_c, meas_deg=ang_m, end_ratio=ang_m[-1] / max(ang_c[-1], 1e-9), axis_cos=cos)
        print(json.dumps(dict(name=name, n_rgb=len(fr), cmd_end=round(ang_c[-1], 1), meas_end=round(ang_m[-1], 1), ratio=round(ang_m[-1] / max(ang_c[-1], 1e-9), 2),
                              axis_cos=round(cos, 2), meas_every4=[round(a, 1) for a in ang_m[::4]])), flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
