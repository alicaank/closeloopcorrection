"""Controller invariants independent of generator, scene and sampling schedule."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from camera_loop import CameraAlignment, CorrectionSession, Measurement
from camera_loop.controller import rejoin
from camera_loop.quality import pose_quality


class ControllerContract(unittest.TestCase):
    def run_chunk(self, ids, *, direction="c2w", axes="opencv", sham=False,
                  measurement=None, last=0):
        ids = np.asarray(ids)
        future = np.repeat(np.eye(4)[None], len(ids), axis=0)
        future[:, 0, 3] = ids
        if measurement is None:
            measured = np.eye(4)
            measured[1, 3] = 2
            measured[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
            measurement = Measurement(measured, last, {"passed": True})
        alignment = CameraAlignment(3, direction, axes, evidence="test")
        output, event = CorrectionSession(alignment).correct(
            measurement,
            planned_endpoint=np.eye(4),
            future_reference_c2w=future,
            future_frame_ids=ids,
            available_last_frame=last,
            previous_native_command=alignment.encode(np.eye(4)),
            sham=sham,
        )
        return alignment.decode(output), event

    def test_rejoin_uses_elapsed_frames_for_translation_and_rotation(self):
        _, event = self.run_chunk([11, 12, 20], last=10)
        desired = np.asarray(event["desired_reference_c2w"])
        np.testing.assert_allclose(desired[:, 1, 3], [1.8, 1.6, 0], atol=1e-12)
        angles = Rotation.from_matrix(desired[:, :3, :3]).as_euler("xyz", degrees=True)
        np.testing.assert_allclose(angles[:, 2], [27, 24, 0], atol=1e-12)

    def test_sampling_density_does_not_change_shared_commands(self):
        sparse, _ = self.run_chunk([1, 2, 10])
        dense, _ = self.run_chunk(np.arange(1, 11))
        np.testing.assert_allclose(sparse, dense[[0, 1, 9]], atol=1e-12)

    def test_uniform_schedule_retains_linear_fade(self):
        _, event = self.run_chunk([4, 8, 12])
        desired = np.asarray(event["desired_reference_c2w"])
        np.testing.assert_allclose(desired[:, 1, 3], [4 / 3, 2 / 3, 0], atol=1e-12)

    def test_sham_bypasses_rejected_or_missing_pose_in_all_conventions(self):
        for direction in ["c2w", "w2c"]:
            for axes in ["opencv", "opengl"]:
                with self.subTest(direction=direction, axes=axes):
                    measured = Measurement(None, 0, {"passed": False, "reasons": ["no_overlap"]})
                    output, event = self.run_chunk(
                        [1, 2, 10], direction=direction, axes=axes,
                        sham=True, measurement=measured,
                    )
                    expected = np.repeat(np.eye(4)[None], 3, axis=0)
                    expected[:, 0, 3] = [1, 2, 10]
                    np.testing.assert_allclose(output, expected, atol=1e-12)
                    self.assertFalse(event["observer_quality"]["passed"])

    def test_sham_still_enforces_timestamp_and_history_constraints(self):
        for frame in [True, 0.0, -1, 1]:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                self.run_chunk([1, 2], sham=True,
                               measurement=Measurement(None, frame, {"passed": False}))
        with self.assertRaises(ValueError):
            self.run_chunk([0, 1], sham=True)

    def test_unsigned_decreasing_frames_cannot_bypass_order_check(self):
        with self.assertRaises(ValueError):
            self.run_chunk(np.array([3, 2, 1], dtype=np.uint64))

    def test_available_frame_must_be_integer_not_bool_or_float(self):
        for last in [False, 0.0, -1]:
            with self.subTest(last=last), self.assertRaises(ValueError):
                self.run_chunk([1, 2], last=last,
                               measurement=Measurement(np.eye(4), 0, {"passed": True}))

    def test_commanded_memory_relative_motion_in_every_native_convention(self):
        measured = np.eye(4)
        measured[:3, :3] = Rotation.from_euler("xyz", [.2, -.4, .1]).as_matrix()
        measured[:3, 3] = [1, 2, -3]
        stored = np.eye(4)
        stored[:3, :3] = Rotation.from_euler("xyz", [-.1, .3, -.2]).as_matrix()
        stored[:3, 3] = [-2, 1, 4]
        future = np.repeat(np.eye(4)[None], 3, axis=0)
        future[:, 0, 3] = [1, 2, 10]
        for direction in ["c2w", "w2c"]:
            for axes in ["opencv", "opengl"]:
                for k in [.01, 7, 100]:
                    with self.subTest(direction=direction, axes=axes, scale=k):
                        alignment = CameraAlignment(k, direction, axes, evidence="test")
                        encoded, event = CorrectionSession(alignment, layout="relative_chain").correct(
                            Measurement(measured, 0, {"passed": True}),
                            planned_endpoint=np.eye(4), future_reference_c2w=future,
                            future_frame_ids=[1, 2, 10], available_last_frame=0,
                            previous_native_command=alignment.encode(stored),
                        )
                        absolute = alignment.encode(stored)
                        if direction == "w2c":
                            absolute = np.linalg.inv(absolute)
                        recovered = []
                        for delta in encoded:
                            absolute = absolute @ (delta if direction == "c2w" else np.linalg.inv(delta))
                            recovered.append(absolute.copy() if direction == "c2w" else np.linalg.inv(absolute))
                        commands = alignment.decode(recovered)
                        desired = np.asarray(event["desired_reference_c2w"])
                        np.testing.assert_allclose(np.linalg.inv(stored) @ commands,
                                                   np.linalg.inv(measured) @ desired, atol=1e-11)

    def test_rejoin_rejects_invalid_progress(self):
        for progress in [[.7, .3], [-.1, 1], [.1, 1.1], [np.nan, 1], [.5]]:
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                rejoin(np.eye(4), np.eye(4), np.repeat(np.eye(4)[None], 2, axis=0),
                       progress=progress)


class PoseQualityContract(unittest.TestCase):
    def test_empty_observation_cannot_pass(self):
        for observations in [[], [None], [None, None]]:
            with self.subTest(observations=observations):
                self.assertFalse(pose_quality(observations)["passed"])

    def test_omitted_context_is_allowed_when_an_actual_pose_exists(self):
        self.assertTrue(pose_quality([None, np.eye(4)])["passed"])

    def test_bundled_observer_has_same_empty_observation_policy(self):
        root = Path(__file__).resolve().parents[1] / "src/camera_loop/bundled/templates"
        path = root / "analysis/plan_base_restart_20261008/continuous_eval_20261029/scale_20261030/correction_v3_20261030/geometry_quality.py"
        spec = importlib.util.spec_from_file_location("bundled_quality", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for observations in [[], [None, None]]:
            self.assertFalse(module.pose_quality(observations)["passed"])
        self.assertTrue(module.pose_quality([None, np.eye(4)])["passed"])


if __name__ == "__main__":
    unittest.main()
