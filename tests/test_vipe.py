"""Exercise the packaged evaluator after relocation without models or CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from camera_loop.integrations import A, BUNDLE, materialize


class VipeEvaluation(unittest.TestCase):
    def test_relocated_registration_and_entrypoints(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / 'work'
            materialize(w, {'models': ['spatia']})
            evaluator = w / A / 'vipe_eval_20261030'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
            result = subprocess.run(
                [sys.executable, str(evaluator / 'test_camera.py')],
                env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Ran 9 tests', result.stderr)
            for relative in ['long_domain_20261028/score_long.py',
                             'new_models_loop_20261028/score_pair.py',
                             'continuous_eval_20261029/scale_20261030/measure80.py']:
                result = subprocess.run(
                    [sys.executable, str(w / A / relative), '--help'],
                    env=env, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--horizon', result.stdout)

    def test_prepare_clones_pinned_vipe_instead_of_vggt(self):
        with tempfile.TemporaryDirectory() as td, patch(
            'camera_loop.integrations.subprocess.run'
        ) as run:
            materialize(Path(td) / 'work', {'models': ['spatia']}, clone=True)
            commands = [call.args[0] for call in run.call_args_list]
            clones = [cmd for cmd in commands if cmd[:2] == ['git', 'clone']]
            self.assertTrue(any('https://github.com/nv-tlabs/vipe.git' in cmd for cmd in clones))
            self.assertFalse(any('https://github.com/SimonSun0810/VGGT-World.git' in cmd for cmd in clones))
            pin = json.loads((BUNDLE / 'repos.json').read_text())['vipe']
            self.assertTrue(any(cmd[-3:] == ['checkout', '--detach', pin['commit']] for cmd in commands))


if __name__ == '__main__':
    unittest.main()
