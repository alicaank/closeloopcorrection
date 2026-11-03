"""Experimental shared camera normalization; no native cache mutation or unit fit."""
from dataclasses import dataclass
import numpy as np
from .geometry import poses


@dataclass(frozen=True)
class SharedCameraGauge:
    commanded_control: np.ndarray
    measured_history: np.ndarray
    receipt: dict


def shared_translation_gauge(plan_relative, history_relative):
    """Prepare matched controls from a plan and an accepted emitted prefix.

    Inputs are adapter-native *framewise relative* transforms, at identical
    native sample indices. The prefix starts at sample zero and must already
    have passed observation/registration checks. Future plan poses are known
    commands, never future observed geometry. Both outputs use the same maximum
    translation-step norm and have identical future tensors.

    This is a candidate arithmetic primitive. The caller must rebuild every
    camera-dependent retained state before adopting a changed denominator;
    this function neither performs that rebuild nor certifies training support.
    """
    plan = np.asarray(plan_relative)
    history = np.asarray(history_relative)
    if plan.ndim != 3 or history.ndim != 3 or not 1 <= len(history) < len(plan):
        raise ValueError('A nonempty emitted prefix and at least one future sample are required')
    plan = poses(plan).copy()
    history = poses(history).copy()
    if not np.array_equal(plan[0], np.eye(4)) or not np.array_equal(history[0], np.eye(4)):
        raise ValueError('Framewise camera sequences must start with exact identity')
    planned_max = float(np.linalg.norm(plan[:, :3, 3], axis=1).max())
    history_max = float(np.linalg.norm(history[:, :3, 3], axis=1).max())
    denominator = max(planned_max, history_max)
    if not np.isfinite(denominator):
        raise ValueError('Translation norm overflow')
    control = plan.copy()
    measured = plan.copy()
    measured[:len(history)] = history
    if denominator > 0:
        control[:, :3, 3] /= denominator
        measured[:, :3, 3] /= denominator
    # Exact zero motion has no identifiable scale; do not invent a unit value.
    for x in (control, measured):
        x.setflags(write=False)
    receipt = dict(denominator=denominator, planned_step_max=planned_max,
        observed_step_max=history_max, emitted_native_samples=len(history),
        total_native_samples=len(plan), scale_identified=False,
        normalization_applied=denominator > 0,
        future_tensors_equal=bool(np.array_equal(control[len(history):], measured[len(history):])),
        changed_from_plan_normalizer=denominator != planned_max,
        cache_rebuild_qualified=False, training_support_qualified=False,
        deployment_qualified=False)
    return SharedCameraGauge(control, measured, receipt)
