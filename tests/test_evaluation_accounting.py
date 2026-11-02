import json,tempfile,unittest
from pathlib import Path
import numpy as np
from camera_loop.evaluation_cache import evaluation_key,cached_estimate
from camera_loop.telemetry import summarize_controller

class EvaluationAccountingTests(unittest.TestCase):
    def test_cache_covers_all_observer_inputs_and_contract(self):
        rgb=np.zeros((2,4,4,3),np.uint8);K=np.eye(3);d=np.ones((4,4))
        key=evaluation_key(rgb,K,d,evaluator_id='frozen-1')
        for a,b,c,v in [(rgb+1,K,d,'frozen-1'),(rgb,K*2,d,'frozen-1'),(rgb,K,d*2,'frozen-1'),(rgb,K,d,'frozen-2')]:
            self.assertNotEqual(key,evaluation_key(a,b,c,evaluator_id=v))
        with self.assertRaises(ValueError):evaluation_key(rgb,K,d,evaluator_id='')
    def test_identical_inputs_run_estimator_once_and_reuse_trajectory(self):
        calls=[]
        def estimate(f,k,d,out):
            calls.append(1);poses=np.tile(np.eye(4),(len(f),1,1));poses[:,0,3]=len(calls)
            np.savez(out,c2w=poses);out.with_suffix('.json').write_text('{"valid":true}')
            return poses,dict(registration=dict(valid=True))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);args=(np.zeros((2,4,4,3),np.uint8),np.eye(3),np.ones((4,4)))
            a,ma=cached_estimate(*args,p/'A.npz',cache_dir=p/'cache',evaluator_id='frozen',estimate=estimate)
            b,mb=cached_estimate(*args,p/'B.npz',cache_dir=p/'cache',evaluator_id='frozen',estimate=estimate)
            self.assertEqual(len(calls),1);np.testing.assert_array_equal(a,b)
            self.assertFalse(ma['evaluation_cache']['hit']);self.assertTrue(mb['evaluation_cache']['hit'])
            self.assertTrue((p/'B.json').exists())
            with self.assertRaises(FileExistsError):cached_estimate(*args,p/'B.npz',cache_dir=p/'cache',evaluator_id='frozen',estimate=estimate)
    def test_rejected_measurements_are_not_counted_as_feedback(self):
        events=[dict(accepted=i==0,confidence=.5 if i==0 else 0,response_update_axes=[False]*6,response_gains=[1]*6,clipped=[]) for i in range(6)]
        r=summarize_controller(dict(controller_events=events),[dict(correction_applied=True)]*6)
        self.assertEqual(r['scheduled_calls'],6);self.assertEqual(r['applied_feedback'],1)
        self.assertEqual(r['legacy_reported_applied'],6);self.assertIsNone(r['changed_command_boundaries'])

class LosslessArchiveTests(unittest.TestCase):
    def test_rgb_roundtrip_preserves_all_pixels(self):
        import shutil
        from camera_loop.lossless_archive import archive_rgb,read_rgb
        ff=shutil.which('ffmpeg') or '/home/hj/Alican/analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'
        if not Path(ff).exists():self.skipTest('FFmpeg not installed')
        rgb=np.random.default_rng(3).integers(0,256,(5,24,32,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'frames.mkv';receipt=archive_rgb(rgb,p,ff)
            self.assertTrue(receipt['pixel_exact']);np.testing.assert_array_equal(read_rgb(p),rgb)
