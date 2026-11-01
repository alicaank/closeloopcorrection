"""Command-independent support for repeated causal geometry windows."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import poses
from .quality import depth_quality, pose_quality


@dataclass(frozen=True)
class EvidencePolicy:
    min_shared_frames: int = 3
    max_shared_position_depth_fraction: float = .03
    max_shared_rotation_degrees: float = 3.
    min_supported_pixels: int = 64
    min_supported_fraction: float = .15
    max_relative_depth_residual: float = .15
    max_rgb_residual: float = .18
    min_texture: float = .01


def select_window(last, previous_ids=(), max_views=16):
    if last < 1 or max_views < 8:
        raise ValueError("Insufficient causal window")
    shared = sorted({int(i) for i in previous_ids if 0 < i < last})[-4:]
    selected = {0, last, *shared}
    # Include a recent neighbouring view and then fill the causal time span.
    selected.add(max(1, last - max(1, last // 16)))
    for i in np.linspace(1, last - 1, max_views * 2).round().astype(int):
        if len(selected) >= max_views:
            break
        selected.add(int(i))
    return np.array(sorted(selected), dtype=np.int64)


def projected_support(depth, cameras, intrinsics, images, policy=EvidencePolicy()):
    """Depth/photometric support of the latest view from its nearest context.

    Uses estimated geometry only; this is evidence, not independent ground truth.
    Images and K must be in the estimator's returned depth pixel grid.
    """
    depth = np.asarray(depth, float)
    images = np.asarray(images, float) / 255.
    cameras = poses(cameras)
    intrinsics = np.asarray(intrinsics, float)
    if depth.ndim != 3 or len(depth) < 2 or images.shape != (*depth.shape, 3):
        raise ValueError("Depth and RGB pixel grids differ")
    h, w = depth.shape[-2:]
    yy, xx = np.mgrid[3:h - 3:4, 3:w - 3:4]
    xy = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    z = depth[-2, yy, xx].ravel()
    xyz = np.linalg.inv(intrinsics[-2]) @ xy * z
    relative = np.linalg.inv(cameras[-1]) @ cameras[-2]
    projected = relative[:3, :3] @ xyz + relative[:3, 3:4]
    pix = intrinsics[-1] @ projected
    positive = np.isfinite(projected).all(axis=0) & (projected[2] > 1e-6) & np.isfinite(z) & (z > 0)
    uv = np.zeros((2, len(z)), dtype=int)
    uv[:, positive] = np.rint(pix[:2, positive] / pix[2:3, positive]).astype(int)
    valid = positive & (uv[0] >= 1) & (uv[0] < w - 1) & (uv[1] >= 1) & (uv[1] < h - 1)
    source = images[-2, yy, xx].reshape(-1, 3)
    gray = images[-2].mean(-1)
    texture = np.hypot(gray[yy + 1, xx] - gray[yy - 1, xx],
                       gray[yy, xx + 1] - gray[yy, xx - 1]).ravel()
    valid &= texture >= policy.min_texture
    candidates = np.flatnonzero(valid)
    supported = np.zeros(len(z), dtype=bool)
    if len(candidates):
        u, v = uv[:, candidates]
        dest_z = depth[-1, v, u]
        dz = np.abs(dest_z - projected[2, candidates]) / np.maximum(projected[2, candidates], 1e-6)
        rgb = np.mean(np.abs(images[-1, v, u] - source[candidates]), axis=1)
        supported[candidates] = np.isfinite(dest_z) & (dest_z > 0) & (dz <= policy.max_relative_depth_residual) & (rgb <= policy.max_rgb_residual)
    count = int(supported.sum())
    fraction = count / max(1, len(z))
    cells = ((xy[0] > w / 2).astype(int) + 2 * (xy[1] > h / 2).astype(int))[supported]
    coverage = len(np.unique(cells))
    return dict(passed=count >= policy.min_supported_pixels and fraction >= policy.min_supported_fraction and coverage >= 3,
                supported_pixels=count, sampled_pixels=len(z), supported_fraction=fraction,
                occupied_quadrants=coverage, in_view_textured_pixels=len(candidates))


def assess_window(frame_ids, cameras, native_depth, reference_depth, intrinsics,
                  images, previous=None, policy=EvidencePolicy()):
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or not len(ids) or ids[0] != 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Window timestamps must be unique and anchor first")
    cameras = poses(cameras)
    qdepth = depth_quality(reference_depth, native_depth[0])
    qpose = pose_quality(cameras)
    # Registration uses image/depth evidence only, never a target camera path.
    scale = qdepth.get("reference_units_per_native_unit", 1.)
    registered = np.linalg.inv(cameras[0]) @ cameras
    registered[:, :3, 3] *= scale
    support = projected_support(native_depth, cameras, intrinsics, images, policy)
    valid_reference = np.asarray(reference_depth)[np.isfinite(reference_depth) & (np.asarray(reference_depth) > 0)]
    depth_unit = float(np.median(valid_reference)) if len(valid_reference) else 1.
    shared = [int(i) for i in ids if i != 0 and previous is not None and int(i) in previous]
    reasons = []
    if not qdepth["passed"]: reasons.append("anchor_depth_fit")
    if not qpose["passed"]: reasons.append("invalid_pose")
    if not support["passed"]: reasons.append("insufficient_projected_image_support")
    position_noise, rotation_noise = 0., 0.
    if previous is not None:
        if len(shared) < policy.min_shared_frames:
            reasons.append("insufficient_shared_frames")
        else:
            current = {int(i): p for i, p in zip(ids, registered)}
            ep = [np.linalg.norm(current[i][:3, 3] - previous[i][:3, 3]) / depth_unit for i in shared]
            er = [np.degrees(Rotation.from_matrix(current[i][:3, :3].T @ previous[i][:3, :3]).magnitude()) for i in shared]
            position_noise, rotation_noise = float(np.quantile(ep, .9)), float(np.quantile(er, .9))
            if position_noise > policy.max_shared_position_depth_fraction: reasons.append("shared_position_disagreement")
            if rotation_noise > policy.max_shared_rotation_degrees: reasons.append("shared_rotation_disagreement")
    confidence = min(1., support["supported_fraction"] / .5)
    confidence /= 1 + position_noise / policy.max_shared_position_depth_fraction + rotation_noise / policy.max_shared_rotation_degrees
    if previous is None: confidence *= .5
    report = dict(passed=not reasons, confidence=float(confidence) if not reasons else 0.,
                  reasons=reasons, frame_id=int(ids[-1]), selected_frame_ids=ids.tolist(),
                  shared_frame_ids=shared, bootstrap=previous is None,
                  shared_position_p90_depth_fraction=position_noise, shared_rotation_p90_degrees=rotation_noise,
                  depth=qdepth, pose=qpose, support=support, thresholds=asdict(policy),
                  policy="anchor_registered_overlap_and_projected_image_support_v1",
                  uses_commanded_poses=False)
    return registered, report
