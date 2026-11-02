import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image

from camera_loop.datasets import import_batch, validate_batch, fingerprint, read_arrays
from camera_loop.experiments import plan_experiment, execute_plan
from camera_loop.integrations import materialize, command
from camera_loop.experiments import check_lyra_cache


class Datasets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        p = np.repeat(np.eye(4)[None], 505, axis=0)
        p[:, 0, 3] = np.linspace(0, -1, 505)
        self.k = np.array([[90.0, 0, 64.0], [0, 90.0, 36.0], [0, 0, 1.0]])
        Image.new("RGB", (128, 72), (30, 60, 90)).save(self.root / "source.png")
        np.savez(
            self.root / "geometry.npz",
            w2c=p,
            intrinsics=self.k,
            anchor_depth=np.ones((72, 128)),
        )
        self.spec = dict(
            dataset="fixture",
            split="test",
            pose_convention="opencv_w2c_anchor_relative",
            depth_convention="z_depth_same_units_as_translation",
            scenes=[
                dict(
                    name=name,
                    image="source.png",
                    geometry="geometry.npz",
                    caption="A room",
                    metres_per_anchor_unit=2,
                )
                for name in ["scene_a", "scene_b"]
            ],
        )
        self.manifest = self.root / "source.json"
        self.manifest.write_text(json.dumps(self.spec))
        self.batch = self.root / "batch"

    def tearDown(self):
        self.temp.cleanup()

    def make(self):
        return import_batch(self.manifest, self.batch, resize=True)

    def test_import_resizes_registered_geometry_and_retains_scene_ids(self):
        self.make()
        report = validate_batch(self.batch, "lyra")
        self.assertTrue(report["passed"])
        self.assertEqual([x["name"] for x in report["scenes"]], ["scene_a", "scene_b"])
        a = read_arrays(self.batch / "trajectories/scene_a.npz")
        self.assertEqual(a["anchor_depth"].shape, (480, 864))
        np.testing.assert_allclose(a["anchor_depth"], 1)
        ray = np.linalg.inv(self.k) @ np.array([20.0, 30.0, 1.0])
        scaled = np.array(
            [(20 + 0.5) * 864 / 128 - 0.5, (30 + 0.5) * 480 / 72 - 0.5, 1.0]
        )
        np.testing.assert_allclose(
            np.linalg.inv(a["intrinsics"][0]) @ scaled, ray, atol=1e-12
        )
        with self.assertRaises(ValueError):
            self.make()

    def test_resize_and_conventions_require_explicit_choice(self):
        with self.assertRaises(ValueError):
            import_batch(self.manifest, self.batch)
        self.assertFalse(self.batch.exists())
        self.spec["pose_convention"] = "opencv_c2w"
        self.manifest.write_text(json.dumps(self.spec))
        with self.assertRaises(ValueError):
            self.make()

    def test_missing_metric_and_short_trajectory_are_reported_before_launch(self):
        self.make()
        p = self.batch / "index.json"
        d = json.loads(p.read_text())
        del d["scenes"][0]["metres_per_anchor_unit"]
        p.write_text(json.dumps(d))
        r = validate_batch(self.batch, "lyra")
        self.assertFalse(r["passed"])
        self.assertFalse(r["scenes"][0]["passed"])
        q = self.batch / "trajectories/scene_b.npz"
        a = read_arrays(q)
        a["w2c"] = a["w2c"][:10]
        a["intrinsics"] = a["intrinsics"][:10]
        a["w2c_overrun"] = a["w2c_overrun"][:10]
        np.savez(q, **a)
        r = validate_batch(self.batch, "spatia", ["scene_b"])
        self.assertFalse(r["passed"])
        self.assertIn("481", str(r))

    def test_malformed_prefix_and_image_grid_rejected(self):
        self.make()
        q = self.batch / "trajectories/scene_a.npz"
        a = read_arrays(q)
        a["w2c_overrun"][4, 0, 3] += 2
        np.savez(q, **a)
        self.assertFalse(validate_batch(self.batch, "spatia", ["scene_a"])["passed"])
        Image.new("RGB", (128, 72)).save(self.batch / "images/scene_b.png")
        self.assertFalse(validate_batch(self.batch, "spatia", ["scene_b"])["passed"])

    def test_duplicate_or_unsafe_ids_rejected(self):
        for name in ["scene_a", "../escape"]:
            self.spec["scenes"][1]["name"] = name
            self.manifest.write_text(json.dumps(self.spec))
            with self.assertRaises(ValueError):
                self.make()

    def test_multi_scene_plan_is_serial_isolated_and_stops_on_failure(self):
        self.make()
        w = self.root / "workspace"
        materialize(w, {})
        py = self.root / "python"
        py.write_text("placeholder")
        plan = plan_experiment(
            w,
            "spatia",
            str(py),
            self.batch,
            self.root / "results",
            scenes=["scene_b", "scene_a"],
        )
        self.assertEqual(
            [(j["scene"], j["arm"]) for j in plan["jobs"]],
            [
                ("scene_b", "baseline"),
                ("scene_b", "correction"),
                ("scene_a", "baseline"),
                ("scene_a", "correction"),
            ],
        )
        self.assertFalse(Path(plan["out"]).exists())
        with patch(
            "camera_loop.experiments.subprocess.run",
            side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)],
        ) as run:
            with self.assertRaises(RuntimeError):
                execute_plan(plan)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(
                run.call_args.kwargs["env"]["GV_SCALE_CACHE"].endswith(
                    fingerprint(self.batch)
                )
            )
        status = json.loads((Path(plan["out"]) / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(len(status["jobs"]), 2)
        with self.assertRaises(ValueError):
            execute_plan(plan)

    def test_changed_batch_invalidates_plan(self):
        self.make()
        plan = plan_experiment(
            "/w", "spatia", "/python", self.batch, self.root / "results"
        )
        p = self.batch / "captions/scene_a.json"
        p.write_text(json.dumps({"0": "Changed caption"}))
        with self.assertRaisesRegex(ValueError, "changed"):
            execute_plan(plan)

    def test_python_override_relocates_spatia_child_interpreters(self):
        w = self.root / "workspace"
        materialize(
            w, {"models": ["spatia"], "python": {"spatia": "/opt/custom/bin/python"}}
        )
        text = (
            w / "analysis/plan_base_restart_20261008/long_domain_20261028/spatia_run.py"
        ).read_text()
        self.assertIn("/opt/custom/bin/python", text)
        self.assertNotIn(str(w / "baselines/venv_spatia/bin/python"), text)

    def test_clone_initializes_lyra_runtime_submodules_at_pinned_checkout(self):
        with patch("camera_loop.integrations.subprocess.run") as run:
            materialize(self.root / "cloned", {"models": ["spatia"]}, clone=True)
        calls = [c.args[0] for c in run.call_args_list]
        self.assertTrue(any("--filter=blob:none" in c for c in calls))
        self.assertTrue(
            any(
                "submodule" in c
                and "Lyra-2/lyra_2/_src/inference/depth_anything_3" in c
                for c in calls
            )
        )

    def test_lyra_missing_cache_rejected_before_interpreter_launch(self):
        self.make()
        with patch("camera_loop.experiments.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "Missing Lyra caption cache"):
                check_lyra_cache(
                    "/w", self.batch, ["scene_a"], self.root / "missing.pt"
                )
            run.assert_not_called()

    def test_astronex_requires_no_recorded_native_scale_or_trajectory(self):
        self.make()
        p = self.batch / "index.json"
        d = json.loads(p.read_text())
        for row in d["scenes"]:
            row["prompts"] = {"astronex": "A room"}
            row.pop("metres_per_anchor_unit", None)
            q = self.batch / "trajectories" / f'{row["name"]}.npz'
            arrays = read_arrays(q)
            arrays.pop("metres_per_anchor_unit", None)
            self.assertNotIn("native_w2c_latent", arrays)
            np.savez(q, **arrays)
        p.write_text(json.dumps(d))
        self.assertTrue(validate_batch(self.batch, "astronex")["passed"])
        argv, _ = command("/w", "astronex", "/python", self.batch, "scene_a", "/out")
        self.assertNotIn("--scale", argv)

    def test_worldplay_offline_scale_is_not_forwarded(self):
        self.make()
        p = self.batch / "index.json"
        d = json.loads(p.read_text())
        for row in d["scenes"]:
            row.update(
                prompts={"worldplay": "A room"},
                native_units_per_anchor_unit=3,
                native_scale_evidence="fixture calibration",
            )
            q = self.batch / "trajectories" / f'{row["name"]}.npz'
            a = read_arrays(q)
            a["intrinsics"][:, :2, 2] = [432, 240]
            np.savez(q, **a)
        p.write_text(json.dumps(d))
        self.assertTrue(validate_batch(self.batch, "worldplay")["passed"])
        argv, _ = command("/w", "worldplay", "/python", self.batch, "scene_a", "/out")
        self.assertNotIn("--unit", argv)
        for row in d["scenes"]:
            del row["native_units_per_anchor_unit"]
            del row["native_scale_evidence"]
        p.write_text(json.dumps(d))
        self.assertTrue(validate_batch(self.batch, "worldplay")["passed"])


if __name__ == "__main__":
    unittest.main()
