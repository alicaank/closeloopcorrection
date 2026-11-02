import os
import types
import unittest
from unittest.mock import patch
import torch
from camera_loop.resource_residency import observer_residency


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
