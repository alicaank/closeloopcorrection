"""Causal future-command correction in a fixed reference frame."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import poses


def _frame_id(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class Measurement:
    c2w: np.ndarray | None  # None is allowed only when correction is bypassed.
    frame_id: int
    quality: dict

    def validate(self):
        _frame_id(self.frame_id, "Measurement frame")
        if not self.quality.get("passed", False):
            raise ValueError("Observer quality gate did not pass")
        a = poses(self.c2w)
        if a.shape != (4, 4):
            raise ValueError("Expected one measured endpoint")
        return a


def rejoin(measured, planned_endpoint, future, translation_gain=1.0, rotation_gain=1.0,
           *, progress=None):
    """Fade a boundary residual over elapsed time; omitted progress assumes uniform samples."""
    m = poses(measured)
    e = poses(planned_endpoint)
    p = poses(future)
    if p.ndim != 3 or not len(p):
        raise ValueError("Expected a nonempty future chunk")
    if m.shape != (4, 4) or e.shape != (4, 4):
        raise ValueError("Endpoints must be single poses")
    if not 0 <= translation_gain <= 1 or not 0 <= rotation_gain <= 1:
        raise ValueError("Gains must lie in [0,1]")
    progress = (
        np.arange(1, len(p) + 1, dtype=float) / len(p)
        if progress is None else np.asarray(progress, dtype=float)
    )
    if (
        progress.shape != (len(p),)
        or not np.isfinite(progress).all()
        or np.any(progress < 0)
        or np.any(progress > 1)
        or np.any(progress[1:] < progress[:-1])
    ):
        raise ValueError("Supply nondecreasing rejoin progress in [0,1] per future pose")
    if np.array_equal(m, e):
        return p
    dt = m[:3, 3] - e[:3, 3]
    dr = Rotation.from_matrix(m[:3, :3] @ e[:3, :3].T)
    s = Slerp([0, 1], Rotation.concatenate([dr, Rotation.identity()]))
    for i, a in enumerate(progress):
        p[i, :3, 3] += (1 - a) * translation_gain * dt
        p[i, :3, :3] = s(1 - (1 - a) * rotation_gain).as_matrix() @ p[i, :3, :3]
    return p


class CorrectionSession:
    def __init__(
        self,
        alignment,
        *,
        history_labels="commanded",
        layout="absolute",
        translation_gain=1.0,
        rotation_gain=1.0
    ):
        if history_labels not in ["commanded", "measured"]:
            raise ValueError("Declare the model memory label policy")
        if layout not in ["absolute", "relative_chain"]:
            raise ValueError("Unsupported command layout")
        if not 0 <= translation_gain <= 1 or not 0 <= rotation_gain <= 1:
            raise ValueError("Gains must lie in [0,1]")
        self.alignment = alignment
        self.history_labels = history_labels
        self.layout = layout
        self.translation_gain = translation_gain
        self.rotation_gain = rotation_gain

    def correct(
        self,
        measurement,
        *,
        planned_endpoint,
        future_reference_c2w,
        future_frame_ids,
        available_last_frame,
        previous_native_command,
        sham=False
    ):
        last = _frame_id(available_last_frame, "Last available frame")
        observed = _frame_id(measurement.frame_id, "Measurement frame")
        future = poses(future_reference_c2w)
        if future.ndim != 3:
            raise ValueError("Expected a sequence of future poses")
        ids = np.asarray(future_frame_ids)
        if (
            ids.ndim != 1
            or len(ids) != len(future)
            or not len(ids)
            or not np.issubdtype(ids.dtype, np.integer)
        ):
            raise ValueError("Supply one integer output-frame ID per future pose")
        # Direct comparisons avoid unsigned subtraction wrapping a decreasing sequence.
        if np.any(ids[1:] <= ids[:-1]) or int(ids[0]) <= last:
            raise ValueError("Correction cannot overwrite generated history")
        if observed != last:
            raise ValueError(
                "Measure the actual last generated frame; stale/future estimates are not interchangeable"
            )
        previous = self.alignment.decode(previous_native_command)
        endpoint = poses(planned_endpoint)
        if previous.shape != (4, 4):
            raise ValueError("Previous command must be one absolute native pose")
        if endpoint.shape != (4, 4):
            raise ValueError("Planned endpoint must be one pose")
        # Subtract as integers before float conversion, including for uint64 frame IDs.
        elapsed = np.asarray([int(frame) - last for frame in ids], dtype=float)
        progress = elapsed / elapsed[-1]
        # Sham is an exact controller bypass; baseline must have supplied these planned commands.
        if sham:
            commands = future.copy()
            desired = future.copy()
            anchor = previous
        else:
            measured = measurement.validate()
            # Apply gains to the measured residual BEFORE bridging frames. A zero
            # gain must not leave a full-strength measured-frame compensation behind.
            effective = endpoint.copy()
            effective[:3, 3] += self.translation_gain * (
                measured[:3, 3] - endpoint[:3, 3]
            )
            delta = Rotation.from_matrix(measured[:3, :3] @ endpoint[:3, :3].T)
            effective[:3, :3] = (
                Slerp([0, 1], Rotation.concatenate([Rotation.identity(), delta]))(
                    self.rotation_gain
                ).as_matrix()
                @ endpoint[:3, :3]
            )
            desired = rejoin(effective, endpoint, future, progress=progress)
            commands = (
                (previous @ np.linalg.inv(effective)) @ desired
                if self.history_labels == "commanded"
                else desired
            )
            anchor = previous if self.history_labels == "commanded" else effective
        encoded = self.alignment.encode(commands)
        if self.layout == "relative_chain":
            # Convert both ends to the native camera basis before composing relatives.
            native_abs = (
                encoded if self.alignment.direction == "c2w" else np.linalg.inv(encoded)
            )
            start = self.alignment.encode(anchor)
            start = start if self.alignment.direction == "c2w" else np.linalg.inv(start)
            prev = np.concatenate([start[None], native_abs[:-1]])
            encoded = np.linalg.inv(prev) @ native_abs
            if self.alignment.direction == "w2c":
                encoded = np.linalg.inv(encoded)
        return encoded, dict(
            observed_frame=int(measurement.frame_id),
            future_frame_ids=ids.tolist(),
            rejoin_progress=progress.tolist() if not sham else None,
            scale=self.alignment.native_units_per_reference_unit,
            history_labels=self.history_labels,
            layout=self.layout,
            sham=sham,
            observer_quality=measurement.quality,
            commanded_reference_c2w=commands.tolist(),
            desired_reference_c2w=desired.tolist(),
        )
