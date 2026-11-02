import unittest
import numpy as np
from camera_loop.memory_rate import bounded_memory_continuation


class MemoryRateTests(unittest.TestCase):
    def setUp(self):
        self.plan = np.repeat(np.eye(4)[None], 49, axis=0)
        self.plan[:, 0, 3] = np.arange(49)*.001
        self.ids = np.arange(25)
        self.measured = self.plan[self.ids].copy()
        self.measured[:, 0, 3] *= 6

    def test_overshoot_retains_lag_without_reverse_or_command_jump(self):
        r = bounded_memory_continuation(self.plan, self.measured, self.ids, native_stride=8)
        self.assertTrue(r.receipt['applied'])
        chain = np.concatenate([self.measured[-1:, :3, 3], r.commands[25:, :3, 3]])
        velocity = np.diff(chain, axis=0)
        self.assertTrue((velocity[:, 0] >= -1e-12).all())
        self.assertLessEqual(np.linalg.norm(velocity, axis=1).max(), r.receipt['speed_ceiling']+1e-12)
        incoming = np.array(r.receipt['incoming_velocity'])[None]
        self.assertLessEqual(np.linalg.norm(np.diff(np.concatenate([incoming, velocity]), axis=0), axis=1).max(), r.receipt['acceleration_ceiling']+1e-12)
        self.assertGreater(r.commands[-1, 0, 3], self.plan[-1, 0, 3])
        np.testing.assert_array_equal(r.commands[:25], self.plan[:25])

    def test_unit_reexpression_preserves_policy(self):
        a = bounded_memory_continuation(self.plan, self.measured, self.ids, native_stride=8)
        for scale in [.01, 100.]:
            p = self.plan.copy(); m = self.measured.copy()
            p[:, :3, 3] *= scale; m[:, :3, 3] *= scale
            b = bounded_memory_continuation(p, m, self.ids, native_stride=8)
            np.testing.assert_allclose(b.commands[:, :3, 3]/scale, a.commands[:, :3, 3], atol=1e-12)

    def test_identity_is_byte_exact(self):
        r = bounded_memory_continuation(self.plan, self.plan[self.ids], self.ids, native_stride=8)
        self.assertFalse(r.receipt['applied'])
        np.testing.assert_array_equal(r.commands, self.plan)

    def test_reversed_incoming_velocity_abstains_without_mutation(self):
        m = self.measured.copy(); m[-8:, 0, 3] = m[-9, 0, 3]-np.arange(1,9)*.01
        r = bounded_memory_continuation(self.plan, m, self.ids, native_stride=8)
        self.assertFalse(r.receipt['applied'])
        np.testing.assert_array_equal(r.commands, self.plan)

    def test_rotation_and_past_are_preserved(self):
        angle = .2; p = self.plan.copy()
        p[25:, :3, :3] = [[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
        r = bounded_memory_continuation(p, self.measured, self.ids, native_stride=8)
        np.testing.assert_array_equal(r.commands[:, :3, :3], p[:, :3, :3])
        np.testing.assert_array_equal(self.plan[self.ids, 0, 3], self.ids*.001)

    def test_intended_native_reversal_is_outside_pilot_scope(self):
        p = self.plan.copy(); p[33:, 0, 3] = p[32, 0, 3]-np.arange(1,17)*.002
        r = bounded_memory_continuation(p, self.measured, self.ids, native_stride=8)
        self.assertFalse(r.receipt['applied'])
        self.assertEqual(r.receipt['reason'], 'planned_native_chunk_not_forward_monotone')
        np.testing.assert_array_equal(r.commands, p)

    def test_future_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            bounded_memory_continuation(self.plan, self.measured, self.ids+24, native_stride=8)


if __name__ == '__main__':
    unittest.main()
