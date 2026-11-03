import os
import types
import unittest
from unittest.mock import patch
import torch
from camera_loop.resource_residency import observer_residency, release_host_allocator


class ResourceResidency(unittest.TestCase):
    def model(self, fail_transfer=False):
        events=[]
        class Model:
            device=torch.device('cpu')
            def parameters(self):
                return iter([types.SimpleNamespace(device=self.device,numel=lambda:4,element_size=lambda:2)])
            def buffers(self):return iter([])
            def to(self, device):
                self.device=device;events.append(str(device))
                if device.type=='cuda' and fail_transfer:raise RuntimeError('transfer failed')
                return self
        return Model(),events

    def test_default_is_noop(self):
        with patch.dict(os.environ,{},clear=True):
            with observer_residency(object()):pass

    def test_restore_after_observer_failure(self):
        model,events=self.model()
        with patch.dict(os.environ,{},clear=True),patch('torch.cuda.empty_cache'):
            with self.assertRaisesRegex(RuntimeError,'observer failed'):
                with observer_residency(model,enabled=True):
                    self.assertEqual(model.device,torch.device('cuda:0'))
                    events.append('observer_closed')
                    raise RuntimeError('observer failed')
        self.assertEqual(events,['cuda:0','observer_closed','cpu'])

    def test_restore_after_partial_transfer_failure(self):
        model,events=self.model(True)
        with patch.dict(os.environ,{},clear=True),patch('torch.cuda.empty_cache'):
            with self.assertRaisesRegex(RuntimeError,'transfer failed'):
                with observer_residency(model,enabled=True):self.fail('Body should not run')
        self.assertEqual(events,['cuda:0','cpu'])

    def test_reject_mixed_devices_before_mutation(self):
        model,events=self.model()
        model.buffers=lambda:iter([types.SimpleNamespace(device=torch.device('cuda:0'))])
        with self.assertRaisesRegex(RuntimeError,'one original'):
            with observer_residency(model,enabled=True):self.fail('Body should not run')
        self.assertEqual(events,[])

    def test_trim_lifecycle_preserves_failure_restoration(self):
        model,events=self.model()
        with patch.dict(os.environ,{'CAMERA_HOST_ALLOCATOR_TRIM':'1'},clear=True),patch('torch.cuda.empty_cache'),patch(
                'camera_loop.resource_residency.release_host_allocator',side_effect=lambda:events.append('trim')):
            with self.assertRaisesRegex(RuntimeError,'observer failed'):
                with observer_residency(model,enabled=True):
                    events.append('observer_closed')
                    raise RuntimeError('observer failed')
        self.assertEqual(events,['cuda:0','trim','observer_closed','trim','cpu','trim'])
        self.assertEqual(model.device,torch.device('cpu'))

    def test_real_trim_preserves_live_tensor_and_rng(self):
        tensor=torch.arange(1024,dtype=torch.float32)
        before=tensor.clone();rng=torch.random.get_rng_state().clone()
        pointer=tensor.data_ptr()
        with patch.dict(os.environ,{'CAMERA_HOST_ALLOCATOR_TRIM':'1'}):
            result=release_host_allocator()
        self.assertIsInstance(result,bool)
        self.assertEqual(tensor.data_ptr(),pointer)
        self.assertTrue(torch.equal(tensor,before))
        self.assertTrue(torch.equal(torch.random.get_rng_state(),rng))
