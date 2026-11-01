"""Quality diagnostics in fixed anchor units. No generated command or eval score enters fitting."""

import numpy as np

POLICY = dict(min_pixels=256, min_coverage=0.20, max_p90_over_p10=4.0)


class GeometryQualityError(ValueError):
    def __init__(self, report):
        self.quality = report
        super().__init__("Unreliable geometry: " + ", ".join(report["reasons"]))


def depth_quality(reference, native, mask=None):
    ref = np.asarray(reference, dtype=np.float64)
    nat = np.asarray(native, dtype=np.float64)
    if ref.ndim != 2 or ref.shape != nat.shape:
        raise ValueError("Depths must share corresponding image pixels")
    valid = np.isfinite(ref) & np.isfinite(nat) & (ref > 1e-6) & (nat > 1e-6)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != ref.shape:
            raise ValueError("Mask shape differs from depth")
        valid &= mask
    reasons = []
    n = int(valid.sum())
    coverage = float(valid.mean())
    if n < POLICY["min_pixels"]:
        reasons.append("insufficient_valid_pixels")
    if coverage < POLICY["min_coverage"]:
        reasons.append("insufficient_coverage")
    out = dict(
        valid_pixels=n,
        valid_fraction=coverage,
        thresholds=POLICY.copy(),
        reasons=reasons,
    )
    if n:
        ratio = ref[valid] / nat[valid]
        scale = float(np.median(ratio))
        p10, p90 = np.quantile(ratio, [0.1, 0.9])
        if not np.isfinite(scale) or scale <= 0:
            reasons.append("invalid_scale")
        dispersion = float(p90 / p10)
        if not np.isfinite(dispersion) or dispersion > POLICY["max_p90_over_p10"]:
            reasons.append("nonuniform_depth_ratio")
        out.update(
            reference_units_per_native_unit=scale,
            p90_over_p10=dispersion,
            median_relative_residual=float(np.median(np.abs(ratio / scale - 1))),
        )
        regions = []
        h, w = ref.shape
        for ys, xs in [
            (slice(0, h // 2), slice(0, w // 2)),
            (slice(0, h // 2), slice(w // 2, w)),
            (slice(h // 2, h), slice(0, w // 2)),
            (slice(h // 2, h), slice(w // 2, w)),
        ]:
            ok = valid[ys, xs]
            regions.append(
                float(np.median(ref[ys, xs][ok] / nat[ys, xs][ok]) / scale)
                if ok.sum() >= 32
                else None
            )
        out["quadrant_scale_relative_to_global"] = regions
    out["passed"] = not reasons
    return out


def pose_quality(poses):
    bad = []
    observed = 0
    for i, p in enumerate(poses):
        if p is None:
            continue  # unrequested context is deliberately omitted by the native observer
        observed += 1
        a = np.asarray(p, dtype=float)
        if a.shape != (4, 4) or not np.isfinite(a).all():
            bad.append(i)
            continue
        R = a[:3, :3]
        if (
            not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6)
            or np.max(np.abs(R.T @ R - np.eye(3))) > 1e-3
            or abs(np.linalg.det(R) - 1) > 1e-3
        ):
            bad.append(i)
    reasons = ["invalid_SE3_pose"] if bad else []
    if not observed:
        reasons.append("no_observed_poses")
    return dict(passed=not reasons, invalid_indices=bad, observed_count=observed,
                valid_count=observed - len(bad), reasons=reasons)
