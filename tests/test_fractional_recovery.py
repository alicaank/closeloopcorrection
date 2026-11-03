import unittest
import numpy as np
from camera_loop.fractional_recovery import fractional_memory_continuation, issue_fractional_future


class FractionalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.plan=np.repeat(np.eye(4)[None],49,axis=0)
        self.plan[:,0,3]=np.arange(49)
        self.ids=np.arange(25)

    def history(self, offset):
        m=self.plan[:25].copy();m[1:,0,3]+=offset
        return m

    def run_policy(self, offset, **kw):
        config=dict(native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
        config.update(kw)
        return fractional_memory_continuation(self.plan,self.history(offset),self.ids,**config)

    def test_lag_has_headroom_and_recovers_requested_fraction(self):
        r=self.run_policy(-6)
        self.assertAlmostEqual(r.receipt['realized_fraction'],.5)
        self.assertAlmostEqual(r.commands[-1,0,3],45.)
        self.assertGreater(r.receipt['steps'][0]['speed'],1.)

    def test_overshoot_can_return_and_recover_requested_fraction(self):
        r=self.run_policy(96,native_stride=1)
        self.assertLess(r.commands[25,0,3],self.history(96)[-1,0,3])
        self.assertAlmostEqual(r.receipt['realized_fraction'],.5)
        self.assertAlmostEqual(r.commands[-1,0,3]-self.plan[-1,0,3],48.)

    def test_saturation_reports_unserved_recovery_and_bounds(self):
        r=self.run_policy(-240)
        self.assertLess(r.receipt['realized_fraction'],.5)
        self.assertGreater(r.receipt['unserved_recovery'][0],0.)
        for row in r.receipt['steps']:
            self.assertLessEqual(row['speed'],r.receipt['speed_ceiling']+1e-12)
            self.assertLessEqual(row['acceleration'],r.receipt['acceleration_ceiling']+1e-12)
        e=np.asarray(r.receipt['initial_residual']);left=np.asarray(r.receipt['endpoint_residual'])
        np.testing.assert_allclose(e-left,r.receipt['realized_command_recovery'])

    def test_overspeed_decelerates_with_bounded_slew(self):
        m=self.history(0);m[:,0,3]*=6
        r=fractional_memory_continuation(self.plan,m,self.ids,native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
        self.assertEqual(r.receipt['speed_ceiling'],6.)
        self.assertEqual(r.receipt['nominal_speed_budget'],2.)
        self.assertLess(r.receipt['steps'][0]['speed'],6.)
        self.assertGreater(r.receipt['steps'][0]['speed'],2.)
        self.assertTrue(all(x['acceleration']<=.75+1e-12 for x in r.receipt['steps']))

    def test_common_unit_change_preserves_policy(self):
        a=self.run_policy(-6)
        for scale in [.01,100.]:
            p=self.plan.copy();m=self.history(-6);p[:,:3,3]*=scale;m[:,:3,3]*=scale
            b=fractional_memory_continuation(p,m,self.ids,native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
            np.testing.assert_allclose(b.commands[:,:3,3]/scale,a.commands[:,:3,3],atol=1e-10)

    def test_identity_rotations_and_immutable_past(self):
        identity=self.run_policy(0)
        self.assertFalse(identity.receipt['applied'])
        np.testing.assert_array_equal(identity.commands,self.plan)
        r=self.run_policy(-6)
        np.testing.assert_array_equal(r.commands[:25],self.plan[:25])
        np.testing.assert_array_equal(r.commands[:,:3,:3],self.plan[:,:3,:3])

    def test_static_plan_does_not_invent_a_speed_budget(self):
        p=self.plan.copy();p[:,:3,3]=0;m=p[:25].copy();m[:,0,3]=2
        r=fractional_memory_continuation(p,m,self.ids,native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
        self.assertEqual(r.receipt['reason'],'no_plan_speed_budget')
        np.testing.assert_array_equal(r.commands[25:,0,3],np.full(24,2.))

    def test_future_and_unsigned_out_of_order_measurements_rejected(self):
        for ids in [self.ids+24,np.array([0,16,8,24],np.uint64)]:
            with self.assertRaises(ValueError):
                fractional_memory_continuation(self.plan,self.plan[:len(ids)],ids,native_stride=8,recovery_fraction=.5,speed_multiplier=2.)

    def test_dream_first_feedback_is_exact_nominal_then_lag_changes_commands(self):
        args=dict(next_end=49,available_frame=24,nominal_boundaries=1,history_labels='commanded',native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
        first=issue_fractional_future(self.plan,self.plan,self.history(-6),self.ids,feedback_index=0,**args)
        self.assertFalse(first.receipt['applied'])
        np.testing.assert_array_equal(first.commands,self.plan)
        later=issue_fractional_future(self.plan,self.plan,self.history(-6),self.ids,feedback_index=1,**args)
        self.assertTrue(later.receipt['applied'])
        np.testing.assert_array_equal(later.commands[:25],self.plan[:25])
        desired=self.run_policy(-6).commands
        bridge=np.asarray(later.receipt['memory_bridge'])
        np.testing.assert_allclose(np.linalg.inv(self.plan[24])@later.commands[25:],
                                   np.linalg.inv(self.history(-6)[-1])@desired[25:],atol=1e-12)
        np.testing.assert_allclose(bridge@self.history(-6)[-1],self.plan[24])

    def test_nominal_bootstrap_rejects_already_modified_future(self):
        issued=self.plan.copy();issued[25:,0,3]+=1
        with self.assertRaisesRegex(ValueError,'untouched planned future'):
            issue_fractional_future(self.plan,issued,self.history(-6),self.ids,
                next_end=49,available_frame=24,feedback_index=0,nominal_boundaries=1,
                history_labels='commanded',native_stride=8,recovery_fraction=.5,speed_multiplier=2.)

    def test_issuance_preserves_unissued_tail_and_rejects_future_measurements(self):
        args=dict(next_end=33,available_frame=24,feedback_index=1,nominal_boundaries=1,
                  history_labels='measured',native_stride=8,recovery_fraction=.5,speed_multiplier=2.)
        r=issue_fractional_future(self.plan,self.plan,self.history(-6),self.ids,**args)
        np.testing.assert_array_equal(r.commands[33:],self.plan[33:])
        with self.assertRaises(ValueError):
            issue_fractional_future(self.plan,self.plan,self.history(-6),self.ids+1,**args)

    def test_delayed_memory_anchor_bridges_at_native_reference_not_rgb_endpoint(self):
        from scipy.spatial.transform import Rotation
        history=self.history(-6)
        history[21,:3,:3]=Rotation.from_euler('z',20,degrees=True).as_matrix()
        args=dict(next_end=33,available_frame=24,feedback_index=1,nominal_boundaries=1,
                  history_labels='commanded',native_stride=4,recovery_fraction=.5,
                  speed_multiplier=2.,memory_anchor_frame=21)
        issued=issue_fractional_future(self.plan,self.plan,history,self.ids,**args)
        desired=fractional_memory_continuation(self.plan[:33],history,self.ids,
                  native_stride=4,recovery_fraction=.5,speed_multiplier=2.)
        np.testing.assert_allclose(np.linalg.inv(self.plan[21])@issued.commands[25:33],
                  np.linalg.inv(history[21])@desired.commands[25:33],atol=1e-12)
        np.testing.assert_array_equal(issued.commands[:25],self.plan[:25])
        np.testing.assert_array_equal(issued.commands[33:],self.plan[33:])
        self.assertEqual(issued.receipt['memory_anchor_frame'],21)
        self.assertEqual(issued.receipt['available_frame'],24)
        # The endpoint bridge is a different operation when the native anchor
        # and newest observation disagree in orientation or translation.
        wrong=self.plan[24]@np.linalg.inv(history[-1])@desired.commands[25:33]
        self.assertGreater(np.max(np.abs(issued.commands[25:33]-wrong)),.1)

    def test_memory_anchor_requires_explicit_causal_observation(self):
        args=dict(next_end=33,available_frame=24,feedback_index=1,nominal_boundaries=1,
                  history_labels='commanded',native_stride=4,recovery_fraction=.5,
                  speed_multiplier=2.)
        for anchor in [-1,25,True,20.5]:
            with self.assertRaisesRegex(ValueError,'Memory anchor'):
                issue_fractional_future(self.plan,self.plan,self.history(-6),self.ids,
                                        memory_anchor_frame=anchor,**args)
        ids=np.array([0,16,24])
        with self.assertRaisesRegex(ValueError,'Memory anchor'):
            issue_fractional_future(self.plan,self.plan,self.history(-6)[ids],ids,
                                    memory_anchor_frame=21,**args)


if __name__=='__main__':unittest.main()
