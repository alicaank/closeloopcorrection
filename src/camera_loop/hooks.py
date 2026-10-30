"""Minimal integration hooks for a new model; no dependency on a generation framework."""

from dataclasses import dataclass
from typing import Callable
from .alignment import align_model
from .controller import CorrectionSession


@dataclass
class ModelHooks:
    model: str
    calibration: Callable  # returns keyword arguments for align_model, from native geometry/metadata
    previous_command: Callable  # absolute native pose at last generated frame, including past compensation
    write_future: (
        Callable  # (output_frame_ids, native_pose_array), must not rewrite past memory
    )
    history_labels: str = "commanded"
    layout: str = "absolute"


class AlignedLoop:
    def __init__(
        self, hooks, *, receipt_path, checkpoint, anchor_id, context="initial_anchor"
    ):
        self.hooks = hooks
        alignment, self.calibration_diagnostics = align_model(
            hooks.model, **hooks.calibration()
        )
        alignment = alignment.save(
            receipt_path,
            anchor_fingerprint=self.calibration_diagnostics.get(
                "reference_sha256", anchor_id
            ),
            checkpoint=checkpoint,
            context=f"{anchor_id}:{context}",
        )
        self.session = CorrectionSession(
            alignment, history_labels=hooks.history_labels, layout=hooks.layout
        )

    def update(
        self,
        measurement,
        *,
        planned_endpoint,
        future_reference_c2w,
        future_frame_ids,
        available_last_frame,
        sham=False,
    ):
        commands, event = self.session.correct(
            measurement,
            planned_endpoint=planned_endpoint,
            future_reference_c2w=future_reference_c2w,
            future_frame_ids=future_frame_ids,
            available_last_frame=available_last_frame,
            previous_native_command=self.hooks.previous_command(),
            sham=sham,
        )
        self.hooks.write_future(future_frame_ids, commands)
        return event
