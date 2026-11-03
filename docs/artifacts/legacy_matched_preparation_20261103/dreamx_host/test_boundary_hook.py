"""Exercise actual released DreamX camera encoding on CPU, without weights."""
import ast
import sys
import types
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

R = Path(__file__).resolve().parent
W = Path('/home/hj/Alican')
sys.path[:0] = [str(R), str(R/'source')]
from boundary_hook import BoundaryWriter, encode, query_frames, stage_boundary

repo = W/'analysis/plan_base_restart_20261008/model_readiness_20261027/repos/dreamx'
ns = dict(np=np, torch=torch)
for path, names in [(repo/'utils/trajectory_processor.py', ['Camera']),
                    (repo/'inference_ar_forcing.py', ['_invert_SE3', 'get_relative_pose', 'cam_params_to_prope_dict'])]:
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
entry = types.SimpleNamespace(**ns)
legacy_path = W/'camera-closed-loop/src/camera_loop/bundled/templates/analysis/plan_base_restart_20261008/new_models_loop_20261028/common.py'
node = next(n for n in ast.parse(legacy_path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'rejoin')
legacy = dict(np=np, Rotation=Rotation, Slerp=Slerp)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(legacy_path), 'exec'), legacy)


class MatchedBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.plan = np.repeat(np.eye(4)[None], 69, axis=0)
        self.plan[:, 0, 3] = np.arange(69)*.002
        self.camera = encode(entry, self.plan, 'cpu', torch.bfloat16)
        self.context = {k: v[:, 3*880:6*880] for k, v in self.camera.items()}
        self.ids = query_frames(3, 6, 18)
        self.measured = self.plan[self.ids].copy()
        self.measured[1:, 0, 3] -= .02
        self.measured[1:, :3, :3] = Rotation.from_euler('y', 7, degrees=True).as_matrix()

    def call(self, arm, **kwargs):
        defaults = dict(start=3, end=6, total=18, arm=arm, quality={'passed': True},
                        camera=self.camera, context=self.context, frame_seq_length=880)
        defaults.update(kwargs)
        return stage_boundary(entry, self.plan, self.plan, self.plan, self.measured, **defaults)

    def test_native_and_observer_controls_leave_both_actuators_exact(self):
        for arm in ['A', 'S']:
            commands, labels, context, camera, receipt = self.call(arm)
            np.testing.assert_array_equal(commands, self.plan)
            np.testing.assert_array_equal(labels, self.plan)
            self.assertIs(context, self.context)
            self.assertIs(camera, self.camera)
            self.assertFalse(receipt['applied'])

    def test_legacy_matches_historical_curve_and_measured_native_anchor(self):
        commands, labels, context, camera, receipt = self.call('L')
        future = [21, 25, 29]
        m = self.measured[-1]
        desired = legacy['rejoin'](m, self.plan[17], self.plan[future])
        np.testing.assert_array_equal(commands[future], desired)
        expected = np.linalg.inv(m) @ desired
        actual = np.linalg.inv(camera['viewmats'][0, 6*880:9*880:880].double().numpy())
        np.testing.assert_allclose(actual, expected, atol=.002, rtol=.002)
        self.assertTrue(receipt['future_changed'])
        self.assertFalse(receipt['context_changed'])
        self.assertIs(context, self.context)
        np.testing.assert_array_equal(labels, self.plan)
        np.testing.assert_array_equal(commands[:21], self.plan[:21])
        np.testing.assert_array_equal(commands[30:], self.plan[30:])
        self.assertTrue(torch.equal(camera['viewmats'][:, :6*880], self.camera['viewmats'][:, :6*880]))
        self.assertTrue(torch.equal(camera['viewmats'][:, 9*880:], self.camera['viewmats'][:, 9*880:]))
        self.assertTrue(torch.equal(camera['K'], self.camera['K']))

    def test_memory_changes_only_current_context_and_retains_future(self):
        commands, labels, context, camera, receipt = self.call('M')
        np.testing.assert_array_equal(commands, self.plan)
        np.testing.assert_array_equal(labels[self.ids], self.measured)
        keep = np.setdiff1d(np.arange(69), self.ids)
        np.testing.assert_array_equal(labels[keep], self.plan[keep])
        self.assertIs(camera, self.camera)
        expected = encode(entry, labels, 'cpu', torch.bfloat16)
        self.assertTrue(torch.equal(context['viewmats'], expected['viewmats'][:, 3*880:6*880]))
        self.assertTrue(torch.equal(context['K'], self.context['K']))
        self.assertTrue(receipt['context_changed'])
        self.assertFalse(receipt['future_changed'])

    def test_identity_roundtrip_is_exact_in_both_intervention_arms(self):
        self.measured = self.plan[self.ids].copy()
        for arm in ['L', 'M']:
            commands, labels, context, camera, receipt = self.call(arm)
            np.testing.assert_array_equal(commands, self.plan)
            np.testing.assert_array_equal(labels, self.plan)
            self.assertTrue(torch.equal(camera['viewmats'], self.camera['viewmats']))
            self.assertTrue(torch.equal(context['viewmats'], self.context['viewmats']))
            self.assertFalse(receipt['applied'])

    def test_failed_quality_invalid_geometry_and_anchor_change_abstain(self):
        cases = [(self.measured, {'passed': False})]
        for bad in ['nonfinite', 'improper_rotation', 'changed_anchor']:
            m = self.measured.copy()
            if bad == 'nonfinite': m[-1, 0, 3] = np.nan
            elif bad == 'improper_rotation': m[-1, 0, 0] = 2.
            else: m[0, 0, 3] = 1.
            cases.append((m, {'passed': True}))
        for measured, quality in cases:
            self.measured = measured
            for arm in ['L', 'M']:
                commands, labels, context, camera, receipt = self.call(arm, quality=quality)
                self.assertFalse(receipt['accepted_observation'])
                self.assertFalse(receipt['applied'])
                np.testing.assert_array_equal(commands, self.plan)
                np.testing.assert_array_equal(labels, self.plan)
                self.assertIs(context, self.context)
                self.assertIs(camera, self.camera)

    def test_first_boundary_corrects_lag_and_can_return_after_overshoot(self):
        for residual, expected in [(-.024, .016), (.048, -.008)]:
            self.measured = self.plan[self.ids].copy()
            self.measured[1:, 0, 3] += residual
            commands, _, _, _, receipt = self.call('L')
            steps = np.diff(np.r_[self.measured[-1, 0, 3], commands[[21,25,29], 0, 3]])
            np.testing.assert_allclose(steps, expected, atol=1e-12)
            self.assertTrue(receipt['applied'])
            self.assertEqual(receipt['first_potentially_affected_rgb'], 21)

    def test_timestamps_context_and_intrinsics_fail_closed(self):
        for start, end, total in [(4,6,18), (3,7,18), (15,18,18)]:
            with self.assertRaises(ValueError): query_frames(start, end, total)
        bad = {k: v.clone() for k, v in self.context.items()}
        bad['viewmats'][0,0,0,3] += 1.
        with self.assertRaisesRegex(ValueError, 'context'):
            self.call('L', context=bad)
        camera = {k: v.clone() for k, v in self.camera.items()}
        camera['K'] *= 2
        context = {k: v[:,3*880:6*880] for k, v in camera.items()}
        with self.assertRaisesRegex(ValueError, 'intrinsics'):
            self.call('L', camera=camera, context=context)

    def test_future_commit_requires_the_matching_completed_cache_boundary(self):
        writer = object.__new__(BoundaryWriter)
        changed = {'sentinel': object()}
        writer.pending = (6, changed)
        with self.assertRaises(ValueError): writer.after_cache(9, self.camera)
        self.assertIs(writer.after_cache(6, self.camera), changed)
        self.assertIsNone(writer.pending)
        self.assertIs(writer.after_cache(9, self.camera), self.camera)


if __name__ == '__main__':
    unittest.main(verbosity=2)
