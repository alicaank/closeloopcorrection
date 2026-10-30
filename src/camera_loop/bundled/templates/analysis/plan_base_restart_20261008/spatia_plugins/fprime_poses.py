"""Pose / gauge helpers for the Spatia F' harness (numpy + scipy only, CPU-testable in any venv).

Conventions
  ours   : OpenCV c2w, anchor-relative (frame 0 = identity), translation in anchor-median-depth units (the batch gauge:
           lyra2_plugins/batch_dev10 index.json "gauge").  Every pose stored in the run json is in this convention.
  spatia : OpenCV w2c [3,4] float32 as read by inference.py:75-85 (read_w2cs_from_txt), translation in MapAnything's
           METRIC units: utils/map_anything_inference.py:126-130 hands every pose to MapAnything with is_metric_scale=True,
           and the first round's cloud comes from ONE view (zero baseline), so its scale is MapAnything's metric depth.
           ours -> spatia multiplies translations by D = spatia_units_per_anchor_depth (median metric depth of the anchor
           in the round-0 MapAnything cloud), so the commanded motion has the planned size relative to the scene.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pad44(m):
    m = np.asarray(m, dtype=np.float64)
    if m.shape[-2:] == (4, 4):
        return m.copy()
    out = np.broadcast_to(np.eye(4), m.shape[:-2] + (4, 4)).copy()
    out[..., :3, :4] = m
    return out


# ------------------------------------------------------------------ ours <-> spatia
def c2w_to_spatia_w2c(c2w, units):
    """ours c2w (...,4,4) -> spatia w2c (...,3,4) float64 (translation scaled by `units` before inversion)."""
    c = pad44(c2w)
    c[..., :3, 3] *= float(units)
    return np.linalg.inv(c)[..., :3, :4]


def spatia_w2c_to_c2w(w2c, units):
    """spatia w2c (...,3,4)|(...,4,4) -> ours c2w (...,4,4)."""
    c = np.linalg.inv(pad44(w2c))
    c[..., :3, 3] /= float(units)
    return c


def write_w2c_txt(path, w2c):
    """Spatia's camera file format (test_cases/*/w2c.txt): one JSON 4x4 list per line (float64 repr round-trips exactly)."""
    w = pad44(w2c)
    with open(path, 'w', encoding='utf-8') as f:
        for m in w:
            f.write(json.dumps(m.tolist()) + '\n')


def normalized_intrinsics(K, width, height):
    """pixel K at (width,height) -> Spatia's normalized [fx/W, fy/H, cx/W, cy/H] (test_cases/*/intrinsics.txt)."""
    K = np.asarray(K, dtype=np.float64)
    return np.array([K[0, 0] / width, K[1, 1] / height, K[0, 2] / width, K[1, 2] / height])


def pixel_K(kn, width, height):
    fx, fy, cx, cy = [float(v) for v in kn]
    return np.array([[fx * width, 0., cx * width], [0., fy * height, cy * height], [0., 0., 1.]])


def write_intrinsics_txt(path, kn):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('[' + ' '.join(repr(float(v)) for v in kn) + ']')


# ------------------------------------------------------------------ replanning
def rejoin_offset(M, planned, prev_end):
    """VERBATIM copy of fresh_teacher_20261021/closed_loop.py:25-34 (test_fprime_poses.py checks equality against the
    original source). The plan is kept and only the start offset (measured M vs the plan's previous end) is removed
    linearly over the call: M == prev_end returns the plan unchanged, the last pose is always the planned end."""
    M = np.asarray(M, np.float64); E = np.asarray(prev_end, np.float64); planned = np.asarray(planned, np.float64); n = len(planned)
    dt = M[:3, 3] - E[:3, 3]; dR = Rotation.from_matrix(M[:3, :3] @ E[:3, :3].T)
    sl = Slerp([0, 1], Rotation.concatenate([dR, Rotation.identity()])); out = planned.copy()
    for k in range(n):
        a = (k + 1) / n; out[k, :3, 3] = planned[k, :3, 3] + (1 - a) * dt; out[k, :3, :3] = sl(a).as_matrix() @ planned[k, :3, :3]
    return out


def se3_interp(delta, w):
    """memorizon_plugins/gv_memorizon.py:39-43 (fraction w of a world-frame rigid correction)."""
    rot = Rotation.from_matrix(delta[:3, :3])
    r = Slerp([0., 1.], Rotation.concatenate([Rotation.identity(), rot]))(w).as_matrix()
    out = np.eye(4); out[:3, :3] = r; out[:3, 3] = w * delta[:3, 3]
    return out


def replan_se3(M, planned, prev_end):
    """The VMem / Memorizon / Lyra harness variant: cmd[k] = se3_interp(M prev_end^-1, 1-(k+1)/n) @ planned[k]."""
    planned = np.asarray(planned, np.float64); n = len(planned)
    delta = np.asarray(M, np.float64) @ np.linalg.inv(np.asarray(prev_end, np.float64))
    return np.stack([se3_interp(delta, 1. - (k + 1) / n) @ planned[k] for k in range(n)])


def geodesic_deg(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return float(np.degrees(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()))


def interp_pose(A, B, t):
    """translation lerp + rotation slerp from A (t=0) to B (t=1)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    sl = Slerp([0, 1], Rotation.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
    out = np.eye(4); out[:3, :3] = sl(t).as_matrix(); out[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
    return out


def valid_pose(p):
    return p is not None and np.all(np.isfinite(np.asarray(p, np.float64)))


def history_labels(measured, commanded, anchor):
    """Pose labels of generated history frames for write=measured: DA3 pose where measured, SE(3) interpolation between
    the nearest measured neighbours (frame 0 = anchor counts as measured) for unmeasured frames inside the measured range,
    commanded pose otherwise. Returns (labels (n,4,4), sources list)."""
    n = len(commanded); labels = np.asarray(commanded, np.float64).copy(); src = ['commanded'] * n
    labels[0] = np.asarray(anchor, np.float64); src[0] = 'anchor'
    known = [0] + [i for i in range(1, n) if valid_pose(measured[i])]
    for i in known[1:]:
        labels[i] = np.asarray(measured[i], np.float64); src[i] = 'measured'
    for a, b in zip(known[:-1], known[1:]):
        for i in range(a + 1, b):
            labels[i] = interp_pose(labels[a], labels[b], (i - a) / (b - a)); src[i] = 'interp'
    return labels, src


# ------------------------------------------------------------------ scale: MapAnything metric depth of the anchor
def zbuffer_median_depth(points_world, w2c, kn, size=(156, 88), min_fill=0.2):
    """Median per-pixel depth of a point cloud seen from camera w2c (normalized intrinsics kn) via a nearest-point z-buffer
    at `size` (W,H). The voxelised MapAnything cloud over-samples far surfaces relative to near ones, so the median over
    POINTS is biased; the z-buffer median approximates the median over PIXELS, i.e. the batch gauge's anchor median depth.
    Falls back to the point median when fewer than min_fill of the pixels are covered. Returns (depth, info)."""
    P = np.asarray(points_world, np.float64)
    w = pad44(w2c)
    pc = P @ w[:3, :3].T + w[:3, 3]
    z = pc[:, 2]; ok = z > 1e-6
    Wz, Hz = size; fx, fy, cx, cy = kn[0] * Wz, kn[1] * Hz, kn[2] * Wz, kn[3] * Hz
    u = np.floor(fx * pc[ok, 0] / z[ok] + cx).astype(np.int64); v = np.floor(fy * pc[ok, 1] / z[ok] + cy).astype(np.int64)
    zin = z[ok]; inside = (u >= 0) & (u < Wz) & (v >= 0) & (v < Hz)
    zb = np.full(Wz * Hz, np.inf); np.minimum.at(zb, v[inside] * Wz + u[inside], zin[inside])
    filled = np.isfinite(zb); fill = float(filled.mean())
    if fill >= min_fill:
        return float(np.median(zb[filled])), dict(method='zbuffer', fill=fill, size=[Wz, Hz], points=int(len(P)))
    return float(np.median(zin)) if zin.size else float('nan'), dict(method='point_median', fill=fill, size=[Wz, Hz], points=int(len(P)))


# ------------------------------------------------------------------ captions
def caption_for_frame(captions, frame, default=''):
    """Batch captions json: {"<frame>": text, ...} (lyra2 batch format) -> text of the largest key <= frame;
    also accepts a plain string or {"caption": text}."""
    if isinstance(captions, str):
        return captions
    if isinstance(captions, dict):
        if 'caption' in captions and isinstance(captions['caption'], str):
            return captions['caption']
        keyed = sorted((int(k), v) for k, v in captions.items() if str(k).lstrip('-').isdigit())
        best = None
        for k, v in keyed:
            if k <= frame:
                best = v
        if best is None and keyed:
            best = keyed[0][1]
        if best is not None:
            return best
    return default
