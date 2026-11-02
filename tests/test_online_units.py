"""Causality, changing native gauges, and native-interface regression tests."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from camera_loop.online_units import (
    CalibrationUnavailable, OnlineDepthCalibration, capture_native_normalizer,
    metric_prediction_scale, runtime_anchor_scale,
)
from camera_loop.integrations import BUNDLE

TEMPLATES = BUNDLE / 'templates'
BASE = TEMPLATES / 'analysis/plan_base_restart_20261008'


def extract_function(path, name, scope):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class OnlineUnits(unittest.TestCase):
    def setUp(self):
        self.depth = np.linspace(2, 6, 32*32).reshape(32, 32)
        self.rgb = np.full((32, 32, 3), 80, np.uint8)
        self.K = np.array([[20., 0, 15.5], [0, 20., 15.5], [0, 0, 1.]])

    def window(self, units, k, first=0, last=0, **overrides):
        values = dict(first_depth=self.depth*k, last_depth=self.depth*k,
                      first_rgb=self.rgb, last_rgb=self.rgb, first_frame=first,
                      last_frame=last, available_frame=last, context=f'w{last}',
                      source='initial_input' if last == 0 else 'generated_prefix')
        values.update(overrides)
        return units.update(**values)

    def test_four_windows_change_gauge_without_changing_reference(self):
        units = OnlineDepthCalibration(self.depth, self.rgb, model='worldwarp')
        for k, first, last in [(3.7, 0, 0), (.5, 0, 80), (8., 80, 160), (1.2, 160, 240)]:
            self.window(units, k, first, last)
            self.assertAlmostEqual(units.scale_for(f'w{last}'), k)
            np.testing.assert_allclose(units.anchors[last][0], self.depth)
        self.assertEqual(set(units.anchors), {0, 240})

    def test_invalid_windows_invalidate_previous_scale(self):
        cases = [dict(available_frame=79), dict(first_frame=40),
                 dict(first_frame=-1), dict(first_frame=0.), dict(first_frame=False),
                 dict(source='recorded_reference'), dict(source='initial_input'),
                 dict(first_rgb=255-self.rgb), dict(first_depth=np.zeros((32, 32))),
                 dict(last_depth=np.full((32, 32), np.nan)), dict(context='w0')]
        for change in cases:
            with self.subTest(change=list(change)):
                units = OnlineDepthCalibration(self.depth, self.rgb, model='worldwarp')
                self.window(units, 2.)
                with self.assertRaises(CalibrationUnavailable):
                    self.window(units, 3., 0, 80, **change)
                with self.assertRaises(CalibrationUnavailable):
                    units.scale_for('w0')
                self.assertEqual(units.events[-1]['status'], 'unavailable')

    def test_nonuniform_depth_does_not_establish_scalar(self):
        units = OnlineDepthCalibration(self.depth, self.rgb, model='worldwarp')
        native = self.depth.copy(); native[:16] *= 10
        with self.assertRaises(CalibrationUnavailable):
            self.window(units, 1., first_depth=native)

    def test_receipts_are_not_reused_as_calibration(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'units.json'
            units = OnlineDepthCalibration(self.depth, self.rgb, model='worldwarp', receipt=path)
            self.window(units, 2.)
            with self.assertRaises(CalibrationUnavailable):
                OnlineDepthCalibration(self.depth, self.rgb, model='worldwarp', receipt=path)
            self.assertEqual(json.loads(path.read_text())['events'][0]['status'], 'identified')

    def test_metric_depth_correspondence_and_native_divisor(self):
        # Actual preprocessing crops four pixels on each side; estimated network
        # intrinsics deliberately differ and must not determine pixel alignment.
        processed = self.K.copy(); processed[:2, 2] -= 4
        calls = []
        def processor(**kwargs):
            calls.append(kwargs)
            return None, None, processed[None]
        observer = SimpleNamespace(input_processor=processor)
        prediction = SimpleNamespace(is_metric=True, depth=(self.depth[4:-4, 4:-4]*7)[None],
                                     intrinsics=(self.K*9)[None])
        k, record = metric_prediction_scale(observer, prediction, self.rgb, self.depth, self.K)
        self.assertAlmostEqual(k, 7/4)
        self.assertEqual(record['native_metric_divisor'], 4)
        self.assertEqual(calls[0]['process_res'], 504)
        # Old median-depth shortcut is wrong when reference median is not one.
        self.assertNotAlmostEqual(k, np.median(prediction.depth)/4)

    def test_missing_metric_depth_has_no_one_fallback(self):
        observer = SimpleNamespace(input_processor=lambda **kw: (None, None, self.K[None]))
        for flag, depth in [(False, self.depth), (None, self.depth), (float('nan'), self.depth), (True, self.depth*np.nan), (True, self.depth*0)]:
            with self.subTest(flag=flag), self.assertRaises(CalibrationUnavailable):
                metric_prediction_scale(observer, SimpleNamespace(is_metric=flag, depth=depth[None]),
                                        self.rgb, self.depth, self.K)

    def test_runtime_provider_receives_only_current_anchor(self):
        calls = []
        def provider(**kwargs):
            calls.append(kwargs)
            self.assertEqual(set(kwargs), {'model', 'anchor', 'intrinsics', 'frame_id', 'available_frame', 'anchor_sha256'})
            return dict(native_depth=self.depth*len(calls)*3, source='initial_input', frame_id=0,
                        anchor_sha256=kwargs['anchor_sha256'], native_unit_evidence='synthetic native geometry')
        with tempfile.TemporaryDirectory() as td:
            for i in range(2):
                k, _ = runtime_anchor_scale('astronex', self.rgb, self.depth, self.K,
                                             receipt=Path(td)/f'{i}.json', provider=provider)
                self.assertAlmostEqual(k, (i+1)*3)
            with self.assertRaises(CalibrationUnavailable):
                runtime_anchor_scale('astronex', self.rgb, self.depth, self.K,
                                     receipt=Path(td)/'0.json', provider=provider)
        self.assertEqual(len(calls), 2)

    def test_unavailable_provider_fails_and_records_reason(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', {'CAMERA_LOOP_UNIT_PROVIDER': ''}):
            path = Path(td)/'units.json'
            with self.assertRaisesRegex(CalibrationUnavailable, 'native units are unidentified'):
                runtime_anchor_scale('worldplay', self.rgb, self.depth, self.K, receipt=path)
            self.assertEqual(json.loads(path.read_text())['status'], 'unavailable')

    def test_scalar_or_reference_video_provider_rejected(self):
        for bad in [dict(scale=2.), dict(source='recorded_reference'), dict(frame_id=80), dict(anchor_sha256='old')]:
            def provider(**kwargs):
                return dict(dict(native_depth=self.depth, source='initial_input', frame_id=0,
                                 anchor_sha256=kwargs['anchor_sha256'], native_unit_evidence='test'), **bad)
            with tempfile.TemporaryDirectory() as td, self.assertRaises(CalibrationUnavailable):
                runtime_anchor_scale('worldplay', self.rgb, self.depth, self.K,
                                     receipt=Path(td)/'units.json', provider=provider)

    def test_normalizer_preserves_outputs_and_logs_each_window(self):
        calls = []
        # Non-mutating stub deliberately returns an object distinct from the input.
        output = np.ones((2, 4, 4)); factor = np.array(2.5)
        def native(cameras):
            return factor, output
        wrapped = capture_native_normalizer(native, calls.append, lambda: 12)
        for _ in range(2):
            f, c = wrapped(np.zeros((2, 4, 4)))
            self.assertIs(f, factor); self.assertIs(c, output)
        self.assertEqual([c['context'] for c in calls], [0, 1])
        self.assertEqual(calls[0]['available_frame'], 12)

    def test_released_vmem_normalizer_cpu_parity(self):
        native = Path(__file__).resolve().parents[2]/'baselines/vmem/modeling/pipeline.py'
        if not native.exists():
            self.skipTest('Optional released VMem checkout unavailable')
        try:
            import torch
        except ImportError:
            self.skipTest('Released VMem CPU parity requires optional torch')
        fn = extract_function(native, 'get_translation_scaling_factor', dict(torch=torch))
        pipe = SimpleNamespace(camera_scale=.4, device='cpu', dtype=torch.float32)
        records = []
        wrapped = capture_native_normalizer(lambda x: fn(pipe, x), records.append, lambda: 80)
        for step in [0., .1, 2.]:
            cameras = torch.eye(4).repeat(9, 1, 1)
            cameras[:, 0, 3] = torch.arange(9)*step
            expected_factor, expected = fn(pipe, cameras.clone())
            actual_factor, actual = wrapped(cameras.clone())
            self.assertEqual(float(actual_factor), float(expected_factor))
            self.assertTrue(torch.equal(actual, expected))
        self.assertNotEqual(records[1]['factor'], records[2]['factor'])

    def test_worldwarp_native_callback_rescales_each_window_once(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Native tensor boundary test requires optional torch')
        units = OnlineDepthCalibration(self.depth, self.rgb, model='worldwarp')
        state = {}; window = {}; prefix = []
        scope = dict(units=units, scale_state=state, source_window=window, prefix=prefix,
                     array=lambda x: x.detach().numpy(), a=SimpleNamespace(camera_window='fixed'))
        callback = extract_function(BASE/'new_models_loop_20261028/worldwarp_run.py', 'calibrate_native', scope)
        poses = torch.eye(4).repeat(1, 161, 1, 1); poses[..., 0, 3] = 2
        video = torch.tensor(self.rgb/255).permute(2, 0, 1)[None, None].repeat(1, 81, 1, 1, 1)
        for i, k in enumerate([2., .5, 7.]):
            window['chunk'] = i
            prefix[:] = [None]*(i*80+1) if i else []
            depth = torch.tensor(self.depth*k)[None, None].repeat(1, 81, 1, 1)
            result = callback(poses, depth, i == 0, video)
            torch.testing.assert_close(result[..., 0, 3], torch.full_like(result[..., 0, 3], 2*k))
            torch.testing.assert_close(poses[..., 0, 3], torch.full_like(poses[..., 0, 3], 2))
            self.assertAlmostEqual(state['applied_native_units_per_reference_unit'], k)
        with self.assertRaises(CalibrationUnavailable):
            callback(poses, depth, False, video)  # repeated context cannot reapply scale
        self.assertEqual(state, {})

    def test_worldwarp_memory_converts_reference_to_current_native_units(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Native memory boundary test requires optional torch')
        plan = np.tile(np.eye(4), (3, 1, 1)); plan[:, 0, 3] = [0, 1, 2]
        world = dict(prefix=np.zeros((3, 2, 2, 3)), plan=plan, out=None,
                     scale_state=dict(status='identified', applied_native_units_per_reference_unit=3.))
        scope = dict(np=np, WORLD=world, policy=lambda: 'identity',
                     choose=lambda *args: SimpleNamespace(applied=True, poses=plan, reason='test'), append=lambda *args: None)
        fn = extract_function(TEMPLATES/'analysis/measured_memory_20261101/memory_hooks.py', 'worldwarp_memory', scope)
        old = torch.eye(4).repeat(1, 3, 1, 1)
        result = fn(dict(rel_cache_poses=old, is_first_chunk=False, T_video=3))
        np.testing.assert_allclose(result[0, :, 0, 3].numpy(), [-6, -3, 0])
        np.testing.assert_allclose(plan[:, 0, 3], [0, 1, 2])
        world['scale_state'].clear()
        with self.assertRaisesRegex(ValueError, 'units unavailable'):
            fn(dict(rel_cache_poses=old, is_first_chunk=False, T_video=3))

    def test_bundle_helper_matches_portable_source(self):
        import camera_loop.online_units as module
        self.assertEqual(Path(module.__file__).read_bytes(),
                         (BASE/'fair_common_20261025/causal_units.py').read_bytes())

    def test_supported_launchers_remove_offline_defaults(self):
        astro = (BASE/'long_domain_20261028/astronex_run.py').read_text()
        self.assertNotIn("sc['native_units_per_anchor_unit']", astro)
        self.assertNotIn("z['native_w2c_latent']", astro)
        for relative in ['fair_common_20261025/vmem_fair.py', 'long_domain_20261028/vmem_run.py']:
            source = (BASE/relative).read_text()
            self.assertNotIn('22.867', source)
            self.assertIn("choices=['window']", source)
        for relative in ['fair_common_20261025/wp_fair.py', 'worldplay_plugins/wp_fprime.py']:
            self.assertNotIn("add_argument('--unit'", (BASE/relative).read_text())


if __name__ == '__main__':
    unittest.main()
