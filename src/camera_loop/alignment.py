"""Calibrate native units from observed anchor geometry or documented native metadata."""

from dataclasses import dataclass, asdict
import hashlib, json, os
from pathlib import Path
import numpy as np
from .geometry import poses, basis


class AlignmentError(ValueError):
    pass


def depth_fit(reference, native, mask=None):
    ref = np.asarray(reference, dtype=float)
    nat = np.asarray(native, dtype=float)
    if ref.ndim != 2 or ref.shape != nat.shape:
        raise AlignmentError("Depths must be registered to identical anchor pixels")
    good = np.isfinite(ref) & np.isfinite(nat) & (ref > 1e-6) & (nat > 1e-6)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != ref.shape:
            raise AlignmentError("Mask shape mismatch")
        good &= m
    if good.sum() < 256 or good.mean() < 0.2:
        raise AlignmentError("Insufficient shared anchor-depth coverage")
    ratio = nat[good] / ref[good]
    k = float(np.median(ratio))
    p10, p90 = np.quantile(ratio, [0.1, 0.9])
    if not np.isfinite(k) or k <= 0 or p90 / p10 > 4:
        raise AlignmentError(
            "Depth mismatch cannot be represented reliably by one scale"
        )
    return k, dict(
        valid_pixels=int(good.sum()),
        valid_fraction=float(good.mean()),
        p90_over_p10=float(p90 / p10),
        median_relative_residual=float(np.median(np.abs(ratio / k - 1))),
        reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
    )


@dataclass(frozen=True)
class CameraAlignment:
    native_units_per_reference_unit: float
    direction: str = "c2w"
    axes: str = "opencv"
    model: str = "custom"
    method: str = "explicit"
    evidence: str = ""

    def __post_init__(self):
        if (
            not np.isfinite(self.native_units_per_reference_unit)
            or self.native_units_per_reference_unit <= 0
        ):
            raise AlignmentError("Native unit scale must be finite and positive")
        if self.direction not in ["c2w", "w2c"]:
            raise AlignmentError("Unknown matrix direction")
        basis(self.axes)
        if not self.evidence:
            raise AlignmentError(
                "Record the geometry/metadata used to establish native units"
            )

    def encode(self, reference_c2w):
        a = poses(reference_c2w)
        a[..., :3, 3] *= self.native_units_per_reference_unit
        a = a @ basis(self.axes)
        return np.linalg.inv(a) if self.direction == "w2c" else a

    def decode(self, native_pose):
        a = poses(native_pose)
        if self.direction == "w2c":
            a = np.linalg.inv(a)
        a = a @ basis(self.axes)
        a[..., :3, 3] /= self.native_units_per_reference_unit
        return a

    def native_depth(self, reference_depth):
        return np.asarray(reference_depth) * self.native_units_per_reference_unit

    def reference_depth(self, native_depth):
        return np.asarray(native_depth) / self.native_units_per_reference_unit

    def save(self, path, *, anchor_fingerprint, checkpoint, context="initial_anchor"):
        """Bind once per checkpoint/anchor/native-normalization context; never overwrite drift."""
        p = Path(path)
        receipt = dict(
            alignment=asdict(self),
            anchor_fingerprint=anchor_fingerprint,
            checkpoint=checkpoint,
            context=context,
        )
        if p.exists():
            old = json.loads(p.read_text())
            for key in ["anchor_fingerprint", "checkpoint", "context"]:
                if old[key] != receipt[key]:
                    raise AlignmentError("Calibration belongs to a different " + key)
            other = CameraAlignment(**old["alignment"])
            for key in ["model", "direction", "axes", "method"]:
                if getattr(other, key) != getattr(self, key):
                    raise AlignmentError("Calibration contract changed")
            if (
                abs(
                    self.native_units_per_reference_unit
                    / other.native_units_per_reference_unit
                    - 1
                )
                > 0.03
            ):
                raise AlignmentError("Calibration drift across matched arms exceeds 3%")
            return other
        p.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation refuses accidental overwrite. Call again to read a racing writer.
        with p.open("x") as f:
            json.dump(receipt, f, indent=2)
        return self


def align_model(
    model,
    *,
    reference_depth=None,
    native_depth=None,
    mask=None,
    metres_per_reference_unit=None,
    native_units_per_reference_unit=None,
    evidence="",
):
    from .registry import profile

    p = profile(model)
    strategy = p["scale_strategy"]
    stats = {}
    if strategy == "anchor_depth":
        if reference_depth is None or native_depth is None:
            raise AlignmentError(
                f"{model} requires registered reference and native anchor depth"
            )
        k, stats = depth_fit(reference_depth, native_depth, mask)
        method = "anchor_depth"
        evidence = evidence or "registered initial observed anchor depths"
    elif strategy in ["metric", "metric_div4"]:
        if metres_per_reference_unit is None:
            raise AlignmentError(f"{model} requires the recorded metric conversion")
        k = float(metres_per_reference_unit) / (4 if strategy == "metric_div4" else 1)
        method = strategy
        if not evidence:
            raise AlignmentError("Identify metric sidecar or calibration provenance")
    else:
        if native_units_per_reference_unit is None or not evidence:
            raise AlignmentError(
                f"{model}: native normalization is model-specific; supply its measured/released factor and provenance, never assume 1"
            )
        k = float(native_units_per_reference_unit)
        method = "native_callback"
    return CameraAlignment(k, p["direction"], p["axes"], model, method, evidence), stats
