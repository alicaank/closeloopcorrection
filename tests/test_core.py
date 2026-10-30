import importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from camera_loop import (
    CameraAlignment,
    AlignmentError,
    align_model,
    CorrectionSession,
    Measurement,
    PixelTransform,
    reanchor,
)
from camera_loop.hooks import ModelHooks, AlignedLoop
from camera_loop.registry import profiles, register_model, _EXTRA
from camera_loop.quality import depth_quality, pose_quality


def trajectory(n=9):
    p = np.repeat(np.eye(4)[None], n, axis=0)
    p[:, :3, :3] = Rotation.from_euler(
        "xyz", np.linspace([0.1, 0.2, 0.3], [0.2, 0.4, -0.2], n)
    ).as_matrix()
    p[:, :3, 3] = np.linspace([1, 2, 3], [2, 4, 5], n)
    return p


class Core(unittest.TestCase):
    def test_all_camera_conventions_roundtrip(self):
        p = trajectory()
        for k in [0.01, 1, 7, 100]:
            for direction in ["c2w", "w2c"]:
                for axes in ["opencv", "opengl"]:
                    a = CameraAlignment(k, direction, axes, evidence="fixture")
                    np.testing.assert_allclose(a.decode(a.encode(p)), p, atol=1e-11)

    def test_auto_native_depth_scale(self):
        ref = np.linspace(1, 3, 1024).reshape(32, 32)
        for model in ["lyra", "spatia", "worldwarp"]:
            a, q = align_model(model, reference_depth=ref, native_depth=ref * 3.7)
            self.assertAlmostEqual(a.native_units_per_reference_unit, 3.7)
            self.assertLess(q["median_relative_residual"], 1e-14)

    def test_metric_and_memorizon_divisor(self):
        for model, k in [("sana", 8), ("matrixgame", 8), ("memorizon", 2)]:
            a, _ = align_model(
                model, metres_per_reference_unit=8, evidence="metric sidecar"
            )
            self.assertEqual(a.native_units_per_reference_unit, k)

    def test_unknown_native_scale_not_assumed_one(self):
        for model in ["lingbot", "dreamx", "worldplay", "vmem", "astronex"]:
            with self.assertRaises(AlignmentError):
                align_model(model)
            a, _ = align_model(
                model, native_units_per_reference_unit=2, evidence="native callback"
            )
            self.assertEqual(a.native_units_per_reference_unit, 2)

    def test_reject_poor_geometry(self):
        ref = np.ones((32, 32))
        bad = np.full((32, 32), np.nan)
        bad[0, 0] = 1
        with self.assertRaises(AlignmentError):
            align_model("lyra", reference_depth=ref, native_depth=bad)
        self.assertFalse(depth_quality(ref, bad)["passed"])
        self.assertFalse(pose_quality([np.diag([2, 1, 1, 1])])["passed"])

    def test_calibration_binding_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scale.json"
            a = CameraAlignment(2, evidence="fixture")
            a.save(p, anchor_fingerprint="a", checkpoint="c")
            old = p.read_bytes()
            near = CameraAlignment(2.01, evidence="fixture").save(
                p, anchor_fingerprint="a", checkpoint="c"
            )
            self.assertEqual(near.native_units_per_reference_unit, 2)
            for kw in [
                dict(anchor_fingerprint="b", checkpoint="c"),
                dict(anchor_fingerprint="a", checkpoint="other"),
            ]:
                with self.assertRaises(AlignmentError):
                    a.save(p, **kw)
            with self.assertRaises(AlignmentError):
                CameraAlignment(3, evidence="fixture").save(
                    p, anchor_fingerprint="a", checkpoint="c"
                )
            self.assertEqual(old, p.read_bytes())

    def test_pixel_transform_rays_and_crop_edges(self):
        tr = PixelTransform((100, 80), (80, 60), (10, 20, 40, 30))
        K = np.array([[70, 0, 49], [0, 65, 39], [0, 0, 1.0]])
        pixels = np.array([[20, 30, 1], [40, 40, 1]], float).T
        np.testing.assert_allclose(
            np.linalg.inv(tr.intrinsics(K)) @ tr.matrix @ pixels,
            np.linalg.inv(K) @ pixels,
            atol=1e-14,
        )
        depth = np.ones((80, 100)) * 100
        depth[20:50, 10:50] = 2
        np.testing.assert_array_equal(tr.depth(depth), np.ones((60, 80)) * 2)

    def test_commanded_memory_bridge(self):
        p = trajectory()
        m = p[0].copy()
        m[:3, 3] += [0.2, -0.1, 0.4]
        m[:3, :3] = Rotation.from_euler("y", 0.2).as_matrix() @ m[:3, :3]
        previous = p[0].copy()
        previous[:3, 3] += [-0.4, 0.2, 0.1]
        for direction in ["c2w", "w2c"]:
            a = CameraAlignment(3, direction, "opengl", evidence="fixture")
            c = CorrectionSession(a)
            out, event = c.correct(
                Measurement(m, 0, {"passed": True}),
                planned_endpoint=p[0],
                future_reference_c2w=p[1:],
                future_frame_ids=np.arange(1, 9),
                available_last_frame=0,
                previous_native_command=a.encode(previous),
            )
            recovered = m @ np.linalg.inv(previous) @ a.decode(out)
            np.testing.assert_allclose(
                recovered, event["desired_reference_c2w"], atol=1e-12
            )
            np.testing.assert_allclose(recovered[-1], p[-1], atol=1e-12)

    def test_relative_chain_in_native_basis(self):
        p = trajectory()
        a = CameraAlignment(2, "c2w", "opengl", evidence="fixture")
        c = CorrectionSession(a, layout="relative_chain")
        out, _ = c.correct(
            Measurement(p[0], 0, {"passed": True}),
            planned_endpoint=p[0],
            future_reference_c2w=p[1:],
            future_frame_ids=np.arange(1, 9),
            available_last_frame=0,
            previous_native_command=a.encode(p[0]),
        )
        absolute = a.encode(p[0])
        recovered = []
        for delta in out:
            absolute = absolute @ delta
            recovered.append(absolute.copy())
        np.testing.assert_allclose(a.decode(recovered), p[1:], atol=1e-12)

    def test_causality_and_quality(self):
        p = trajectory()
        a = CameraAlignment(1, evidence="fixture")
        c = CorrectionSession(a)
        kw = dict(
            planned_endpoint=p[0],
            future_reference_c2w=p[1:],
            future_frame_ids=np.arange(1, 9),
            available_last_frame=0,
            previous_native_command=p[0],
        )
        for m in [
            Measurement(p[0], 1, {"passed": True}),
            Measurement(p[0], 0, {"passed": False}),
        ]:
            with self.assertRaises(ValueError):
                c.correct(m, **kw)
        kw["future_frame_ids"] = np.arange(0, 8)
        with self.assertRaises(ValueError):
            c.correct(Measurement(p[0], 0, {"passed": True}), **kw)

    def test_sham_and_inputs_unchanged(self):
        p = trajectory()
        old = p.copy()
        a = CameraAlignment(1, evidence="fixture")
        c = CorrectionSession(a)
        m = p[0].copy()
        m[:3, 3] += 1
        out, _ = c.correct(
            Measurement(m, 0, {"passed": True}),
            planned_endpoint=p[0],
            future_reference_c2w=p[1:],
            future_frame_ids=np.arange(1, 9),
            available_last_frame=0,
            previous_native_command=p[0],
            sham=True,
        )
        np.testing.assert_array_equal(out, p[1:])
        np.testing.assert_array_equal(p, old)

    def test_custom_hooks_calibrate_and_write_only_future(self):
        register_model("toy", scale_strategy="anchor_depth")
        try:
            p = trajectory()
            written = []
            ref = np.ones((32, 32))
            a = CameraAlignment(2, evidence="fixture")
            hooks = ModelHooks(
                "toy",
                lambda: dict(reference_depth=ref, native_depth=ref * 2),
                lambda: a.encode(p[0]),
                lambda ids, cmd: written.append((ids, cmd)),
            )
            with tempfile.TemporaryDirectory() as td:
                loop = AlignedLoop(
                    hooks,
                    receipt_path=Path(td) / "a.json",
                    checkpoint="toy1",
                    anchor_id="scene",
                )
                loop.update(
                    Measurement(p[0], 0, {"passed": True}),
                    planned_endpoint=p[0],
                    future_reference_c2w=p[1:],
                    future_frame_ids=np.arange(1, 9),
                    available_last_frame=0,
                )
                self.assertEqual(len(written), 1)
                np.testing.assert_allclose(written[0][1], a.encode(p[1:]), atol=1e-12)
                # Identical depth arrays do not make two different anchors interchangeable.
                with self.assertRaises(AlignmentError):
                    AlignedLoop(
                        hooks,
                        receipt_path=Path(td) / "a.json",
                        checkpoint="toy1",
                        anchor_id="different-scene",
                    )
        finally:
            _EXTRA.pop("toy")

    def test_simulated_underexecution(self):
        path = Path(__file__).resolve().parents[1] / "examples/synthetic_loop.py"
        spec = importlib.util.spec_from_file_location("demo", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.assertLess(m.run(True), m.run(False))


if __name__ == "__main__":
    unittest.main()
