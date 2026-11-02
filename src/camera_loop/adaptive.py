"""Stateful bounded recovery in a declared native camera contract.

B and C share evidence gating, timing, memory and motion limits. C additionally
fits a conservative diagonal response in supported native motion directions.
Bounds are experiment declarations, not universal model safety thresholds.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import poses


def vector(p):
    return np.r_[p[:3, 3], Rotation.from_matrix(p[:3, :3]).as_rotvec()]


def transform(v):
    p = np.eye(4)
    p[:3, 3] = v[:3]
    p[:3, :3] = Rotation.from_rotvec(v[3:]).as_matrix()
    return p


def bounded(raw, previous, magnitude, change):
    """Project into a speed ball, then interpolate from a feasible prior speed."""
    target = raw * min(1., magnitude / max(np.linalg.norm(raw), 1e-15))
    delta = target - previous
    result = previous + delta * min(1., change / max(np.linalg.norm(delta), 1e-15))
    return result, bool(np.linalg.norm(result - raw) > 1e-9)


@dataclass(frozen=True)
class RecoveryConfig:
    return_frames: float = 64.
    feedback_gain: float = .5
    acceleration_frames: float = 12.
    response_min: float = .75
    response_max: float = 1.5
    response_prior: float = 4.
    response_min_samples: int = 3
    response_min_cosine: float = .8


class DirectionalResponse:
    def __init__(self, config):
        self.config = config
        self.xx = np.zeros(6)
        self.xy = np.zeros(6)
        self.count = np.zeros(6, dtype=int)

    def gains(self):
        c = self.config
        fitted = np.clip((self.xy + c.response_prior) / (self.xx + c.response_prior),
                         c.response_min, c.response_max)
        return np.where(self.count >= c.response_min_samples, fitted, 1.)

    def update(self, issued, observed, scale, confidence):
        x, y = vector(issued) / scale, vector(observed) / scale
        used = np.zeros(6, dtype=bool)
        for sl in [slice(0, 3), slice(3, 6)]:
            nx, ny = np.linalg.norm(x[sl]), np.linalg.norm(y[sl])
            cosine = float(x[sl] @ y[sl] / max(nx * ny, 1e-12))
            # Do not turn wrong-axis/near-zero measurements into inverse gains.
            if nx < .1 or ny < .1 or cosine < self.config.response_min_cosine:
                continue
            local = (np.abs(x[sl]) >= max(.1, .25 * nx)) & (x[sl] * y[sl] > 0)
            # A diagonal model must explain the block before it may adapt it.
            prediction = self.gains()[sl] * x[sl]
            if np.linalg.norm(y[sl] - prediction) > .75 * max(nx, ny):
                continue
            used[sl] = local
        self.xx[used] = .9 * self.xx[used] + confidence * x[used] ** 2
        self.xy[used] = .9 * self.xy[used] + confidence * x[used] * y[used]
        self.count[used] += 1
        return used.tolist()


class AdaptiveController:
    def __init__(self, plan, frame_ids, alignment, *, arm="B", initial_count=4,
                 initial_native_relative=None, limits=None, config=RecoveryConfig()):
        self.plan = poses(plan)
        self.ids = np.asarray(frame_ids)
        if self.plan.ndim != 3 or self.ids.shape != (len(self.plan),):
            raise ValueError("One timestamp per planned pose is required")
        if (not np.issubdtype(self.ids.dtype, np.integer) or self.ids[0] != 0
                or np.any(self.ids[1:] <= self.ids[:-1])):
            raise ValueError("Expected increasing integer output frames starting at zero")
        if arm not in ("B", "C") or not 1 < initial_count < len(self.plan):
            raise ValueError("Invalid arm or initial generated prefix")
        if (config.return_frames <= 0 or config.acceleration_frames <= 0
                or not 0 < config.feedback_gain <= 1
                or not 0 < config.response_min <= 1 <= config.response_max):
            raise ValueError("Invalid recovery configuration")
        self.alignment, self.arm, self.config = alignment, arm, config
        self.dt = np.r_[1., np.diff(self.ids).astype(float)]
        absolute = self._native(self.plan)
        self.issued = np.repeat(np.eye(4)[None], len(self.plan), axis=0)
        self.issued[1:] = np.linalg.inv(absolute[:-1]) @ absolute[1:]
        if initial_native_relative is not None:
            native = poses(initial_native_relative)
            if native.shape != self.issued.shape:
                raise ValueError("Native initialization must cover the full plan")
            self.issued = native.copy()
        v = np.stack([vector(x) for x in self.issued]) / self.dt[:, None]
        derived = np.array([np.linalg.norm(v[1:, :3], axis=1).max(),
                            np.linalg.norm(v[1:, 3:], axis=1).max()])
        self.limits = derived if limits is None else np.asarray(limits, float)
        if self.limits.shape != (2,) or not np.isfinite(self.limits).all() or np.any(self.limits < derived - 1e-6):
            raise ValueError("Declared speed limits must cover the original native plan")
        self.predicted = self.plan[initial_count - 1].copy()
        self.commands = self.plan.copy()
        if initial_native_relative is not None:
            stored = absolute[0].copy()
            for j in range(1, initial_count):
                stored = stored @ self.issued[j]
                self.commands[j] = self._reference(stored)
        self.next_index = initial_count
        self.response = DirectionalResponse(config)
        self.last_observation = None
        self.last_observation_index = None
        self.last_clipped = False
        self.events = []

    def _native(self, reference):
        out = self.alignment.encode(reference)
        return out if self.alignment.direction == "c2w" else np.linalg.inv(out)

    def _reference(self, native):
        return self.alignment.decode(native if self.alignment.direction == "c2w" else np.linalg.inv(native))

    def next_chunk(self, lo, hi, measured, evidence, *, stored_boundary=None,
                   available_last_frame=None):
        """Issue future cameras against the host's actual stored boundary label.

        ``stored_boundary`` is canonical c2w, supplied only after a host has
        rebuilt its memory under that label. It never rewrites command history.
        A delayed camera anchor may precede the latest decoded RGB frame, but
        every new conditioning timestamp must still be in the ungenerated future.
        """
        if lo != self.next_index or not lo < hi <= len(self.plan):
            raise ValueError("Stale update or history rewrite")
        if int(evidence.get("frame_id", -1)) != int(self.ids[lo - 1]):
            raise ValueError("Observation is not at the actual generated boundary")
        available = self.ids[lo - 1] if available_last_frame is None else available_last_frame
        if (not isinstance(available, (int, np.integer))
                or not self.ids[lo - 1] <= available < self.ids[lo]):
            raise ValueError("Future conditioning overlaps already generated frames")
        boundary = self.commands[lo - 1] if stored_boundary is None else poses(stored_boundary)
        if boundary.shape != (4, 4):
            raise ValueError("Expected one actual stored boundary label")
        accepted = bool(evidence.get("passed", False))
        confidence = float(evidence.get("confidence", 0.)) if accepted else 0.
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid observation confidence")
        fit = [False] * 6
        if accepted:
            m = poses(measured)
            if m.shape != (4, 4):
                raise ValueError("Expected one measured endpoint")
            if self.arm == "C" and self.last_observation_index == lo - 1 - evidence.get("previous_chunk_size", -1) and not self.last_clipped:
                actual = np.eye(4)
                for command in self.issued[self.last_observation_index + 1:lo]:
                    actual = actual @ command
                seen = np.linalg.inv(self._native(self.last_observation)) @ self._native(m)
                duration = self.ids[lo - 1] - self.ids[self.last_observation_index]
                scale = np.maximum(np.repeat(self.limits * duration, 3), 1e-9)
                fit = self.response.update(actual, seen, scale, confidence)
            self.predicted = m.copy()
            self.last_observation = m.copy()
            self.last_observation_index = lo - 1
        else:
            # Prediction is a separate belief, never a measured-memory write.
            self.last_observation_index = None
        gains = self.response.gains() if self.arm == "C" else np.ones(6)
        original_future = self.commands[lo:hi].copy()
        fresh_feedback = accepted and confidence > 0 and not np.allclose(self.predicted, self.plan[lo-1], atol=1e-12, rtol=0)
        state = self.predicted.copy()
        native_command = self._native(boundary)
        previous = vector(self.issued[lo - 1]) / self.dt[lo - 1]
        previous_R = self.issued[lo - 1, :3, :3]
        clipped, horizons, raw_inputs, bridge_residuals = [], [], [], []
        for j in range(lo, hi):
            dt = self.dt[j]
            # Transport prior body-frame velocity into the next command frame.
            previous = np.r_[previous_R.T @ previous[:3], previous_R.T @ previous[3:]]
            residual = state[:3, 3] - self.plan[j - 1, :3, 3]
            r = Rotation.from_matrix(state[:3, :3] @ self.plan[j - 1, :3, :3].T).as_rotvec()
            native_error = np.array([np.linalg.norm(residual) * self.alignment.native_units_per_reference_unit,
                                     np.linalg.norm(r)])
            recovery_time = np.zeros(2)
            movable = self.limits > 1e-12
            recovery_time[movable] = 2 * native_error[movable] / self.limits[movable]
            horizon = max(self.config.return_frames, float(recovery_time.max()))
            alpha = confidence * self.config.feedback_gain * (-np.expm1(-dt / horizon))
            desired = self.plan[j].copy()
            desired[:3, 3] += (1 - alpha) * residual
            desired[:3, :3] = Rotation.from_rotvec((1 - alpha) * r).as_matrix() @ desired[:3, :3]
            physical = np.linalg.inv(self._native(state)) @ self._native(desired)
            requested = vector(physical) / gains / dt
            t, tc = bounded(requested[:3], previous[:3], self.limits[0], self.limits[0] * dt / self.config.acceleration_frames)
            rvel, rc = bounded(requested[3:], previous[3:], self.limits[1], self.limits[1] * dt / self.config.acceleration_frames)
            velocity = np.r_[t, rvel]
            command = transform(velocity * dt)
            old_native_command = native_command.copy()
            native_command = native_command @ command
            self.commands[j] = self._reference(native_command)
            self.issued[j] = command
            # Check the commanded-memory bridge in the actual native basis.
            bridge_residuals.append(float(np.max(np.abs(np.linalg.inv(old_native_command) @ native_command - command))))
            state = self._reference(self._native(state) @ transform(velocity * dt * gains))
            previous, previous_R = velocity, command[:3, :3]
            clipped.append([tc, rc]);horizons.append(horizon);raw_inputs.append(requested.tolist())
        self.predicted = state
        self.next_index = hi
        self.last_clipped = bool(np.any(clipped))
        event = dict(frame_id=int(self.ids[lo - 1]), future_frame_ids=self.ids[lo:hi].tolist(),
                     available_last_frame=int(available),
                     observation_delay_frames=int(available - self.ids[lo - 1]),
                     stored_boundary=boundary.tolist(),
                     boundary_source="issued_command" if stored_boundary is None else "host_rebuilt_memory",
                     accepted=accepted, confidence=confidence, response_gains=gains.tolist(),
                     controller_called=True, fresh_residual_feedback=bool(fresh_feedback),
                     commands_changed=not np.allclose(original_future,self.commands[lo:hi],atol=1e-12,rtol=0),
                     response_update_axes=fit, return_horizon_frames=horizons, clipped=clipped,
                     requested_native_velocity=raw_inputs, bridge_max_error=max(bridge_residuals),
                     predicted_endpoint=state.tolist(), target_endpoint=self.plan[hi - 1].tolist(),
                     endpoint_tracking_error=float(np.linalg.norm(state[:3, 3] - self.plan[hi - 1, :3, 3])),
                     evidence=evidence)
        self.events.append(event)
        relative = self.issued[lo:hi].copy()
        return relative if self.alignment.direction == "c2w" else np.linalg.inv(relative), event

    def metadata(self):
        return dict(config=asdict(self.config), native_speed_limits=self.limits.tolist(),
                    arm=self.arm, response="identity" if self.arm == "B" else "supported_diagonal",
                    native_relative_inputs=self.issued.tolist(), controller_events=self.events,
                    history_policy="immutable issued commands; explicit host-rebuilt memory boundary supported; future writes only",
                    target_policy="original output timestamps; no endpoint snap or time warp")
