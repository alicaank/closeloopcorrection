import compileall, hashlib, json, os, tempfile, unittest, subprocess
from pathlib import Path
from camera_loop.integrations import materialize, command, BUNDLE, V
from camera_loop.registry import profiles


class Integrations(unittest.TestCase):
    def test_patch_integrity_and_syntax(self):
        for path, expected in json.loads(
            (BUNDLE / "patch_manifest.json").read_text()
        ).items():
            data = (BUNDLE / path).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
            if data.strip():
                result = subprocess.run(
                    ["git", "apply", "--numstat", "-"], input=data, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_relocation_compile_and_v3_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "work"
            materialize(
                w,
                {
                    "home_root": "/home/teammate",
                    "conda_root": "/opt/conda",
                    "cache_root": "/data/cache",
                },
            )
            self.assertTrue(compileall.compile_dir(str(w), quiet=1))
            for p in w.rglob("*.py"):
                s = p.read_text()
                for token in ["@WORKSPACE@", "/home/hj", "@CONDA_ROOT@"]:
                    self.assertFalse(token in s, f"{token} remains in {p}")
            m = json.loads((w / V / "source_manifest.json").read_text())
            for name, record in m.items():
                self.assertEqual(
                    hashlib.sha256(
                        (w / V / "snapshots" / name).read_bytes()
                    ).hexdigest(),
                    record["sha256"],
                )
            with self.assertRaises(ValueError):
                materialize(w, {})

    def test_all_eleven_launchers_materialize(self):
        self.assertEqual(len(profiles()), 11)
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "work"
            materialize(w, {})
            for model in profiles():
                argv, env = command(
                    w, model, "/opt/python", "/data/batch", "scene", "/data/out"
                )
                self.assertTrue(Path(argv[1]).is_file(), model)
                self.assertEqual(env["LONG_BATCH"], "/data/batch")
                if model == "vmem":
                    self.assertEqual(argv[argv.index("--arm") + 1], "replan_commanded")
                if model in ["lyra", "spatia", "sana", "worldwarp"]:
                    self.assertIn("--observer-quality", argv)

    def test_vmem_measured_arm_is_explicit(self):
        argv, _ = command(
            "/w", "vmem", "/python", "/batch", "scene", "/out", "replan_measured"
        )
        self.assertEqual(argv[argv.index("--arm") + 1], "replan_measured")
        with self.assertRaises(ValueError):
            command(
                "/w", "spatia", "/python", "/batch", "scene", "/out", "replan_measured"
            )

    def test_historical_control_limits(self):
        for model in ["memorizon", "worldplay"]:
            with self.assertRaises(ValueError):
                command("/w", model, "/python", "/batch", "scene", "/out", "sham")

    def test_bundle_has_no_model_assets_or_machine_paths(self):
        forbidden = {".pt", ".pth", ".ckpt", ".safetensors", ".mp4", ".npz", ".npy"}
        for p in BUNDLE.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                self.assertNotIn(p.suffix, forbidden)
                self.assertNotIn("/home/hj", p.read_text(), str(p))


if __name__ == "__main__":
    unittest.main()
