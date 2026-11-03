import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from camera_loop.conditioning_gauge import shared_translation_gauge


class SharedGaugeTests(unittest.TestCase):
    def example(self):
        p=np.repeat(np.eye(4)[None],8,axis=0)
        p[1:,:3,3]=np.arange(1,8)[:,None]*np.array([.01,.02,-.03])
        h=p[:4].copy();h[1:,:3,3]*=50
        h[1:,:3,:3]=Rotation.from_euler('z',[[1],[2],[3]],degrees=True).as_matrix()
        return p,h

    def test_shared_envelope_future_equality_and_no_mutation(self):
        p,h=self.example();before=(p.copy(),h.copy());r=shared_translation_gauge(p,h)
        for x in (r.commanded_control,r.measured_history):
            self.assertLessEqual(np.linalg.norm(x[:,:3,3],axis=1).max(),1+1e-15)
            self.assertFalse(x.flags.writeable)
        np.testing.assert_array_equal(r.commanded_control[4:],r.measured_history[4:])
        np.testing.assert_array_equal(r.commanded_control[:,:3,:3],p[:,:3,:3])
        np.testing.assert_array_equal(r.measured_history[:4,:3,:3],h[:,:3,:3])
        np.testing.assert_array_equal(p,before[0]);np.testing.assert_array_equal(h,before[1])
        self.assertTrue(r.receipt['changed_from_plan_normalizer'])
        self.assertFalse(r.receipt['deployment_qualified'])

    def test_common_unit_scale_invariance_and_ratio_preservation(self):
        p,h=self.example();r=shared_translation_gauge(p,h)
        for scale in (1e-8,.01,100,1e8):
            a,b=p.copy(),h.copy();a[:,:3,3]*=scale;b[:,:3,3]*=scale
            s=shared_translation_gauge(a,b)
            np.testing.assert_allclose(s.commanded_control,r.commanded_control,atol=1e-15)
            np.testing.assert_allclose(s.measured_history,r.measured_history,atol=1e-15)
        observed_ratio=np.linalg.norm(h[3,:3,3])/np.linalg.norm(p[7,:3,3])
        encoded_ratio=np.linalg.norm(r.measured_history[3,:3,3])/np.linalg.norm(r.measured_history[7,:3,3])
        self.assertAlmostEqual(observed_ratio,encoded_ratio)

    def test_identity_reproduces_plan_normalization_and_zero_motion_is_explicit(self):
        p,_=self.example();r=shared_translation_gauge(p,p[:4])
        expected=p.copy();expected[:,:3,3]/=np.linalg.norm(p[:,:3,3],axis=1).max()
        np.testing.assert_array_equal(r.commanded_control,expected)
        np.testing.assert_array_equal(r.measured_history,expected)
        self.assertFalse(r.receipt['changed_from_plan_normalizer'])
        p[:,:3,3]=0;r=shared_translation_gauge(p,p[:4])
        self.assertEqual(r.receipt['denominator'],0)
        self.assertFalse(r.receipt['normalization_applied'])
        self.assertFalse(r.receipt['scale_identified'])
        np.testing.assert_array_equal(r.measured_history,p)

    def test_invalid_or_nonprefix_contracts_fail(self):
        p,h=self.example()
        for bad in (h[:0],p,np.zeros((4,3,3)),np.full_like(h,np.nan)):
            with self.assertRaises(ValueError):shared_translation_gauge(p,bad)
        h[0,0,3]=1
        with self.assertRaises(ValueError):shared_translation_gauge(p,h)
