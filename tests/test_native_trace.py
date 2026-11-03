import json
from pathlib import Path
import random
import tempfile
import types
import unittest

import numpy as np
import torch

from camera_loop.native_trace import NativeTrace, digest, rng_digest, worldplay_trace


class NativeTraceTests(unittest.TestCase):
    def test_hash_preserves_values_and_rng_including_noncontiguous_bfloat16(self):
        x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T
        old = x.clone()
        before = rng_digest()
        self.assertEqual(digest(x), digest(x.contiguous()))
        self.assertNotEqual(digest(x)['sha256'], digest(x + 1)['sha256'])
        self.assertEqual(digest(torch.tensor(1.0))['shape'], [])
        self.assertTrue(torch.equal(old, x))
        self.assertEqual(before, rng_digest())

    def test_wrapper_and_hooks_preserve_rng_output_and_cleanup_after_error(self):
        class Model(torch.nn.Module):
            def forward(self, x, *, scale):
                return x * scale + torch.rand_like(x)

        class Host:
            def call(self, x):
                return model(x, scale=2)

        model, host = Model(), Host()
        x = torch.arange(4, dtype=torch.float32)
        seed = torch.random.get_rng_state()
        expected = host.call(x)
        after = torch.random.get_rng_state()
        torch.random.set_rng_state(seed)
        with tempfile.TemporaryDirectory() as td:
            trace = NativeTrace(Path(td) / 'trace.jsonl')
            trace.wrap(host, 'call', 'host')
            trace.module(model, 'model')
            try:
                actual = host.call(x)
            finally:
                trace.close()
            self.assertTrue(torch.equal(expected, actual))
            self.assertTrue(torch.equal(after, torch.random.get_rng_state()))
            rows = [json.loads(s) for s in trace.path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],
                             ['host.before', 'model.before', 'model.after', 'host.after'])
            self.assertNotIn('call', vars(host))
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model._forward_pre_hooks)

    def test_worldplay_context_cleans_up_on_exception_and_leaves_observer_inputs(self):
        pipe = types.SimpleNamespace(transformer=torch.nn.Identity(), ctx=None,
            prepare_latents=lambda *args: torch.ones(2),
            decode_next_latent=lambda: np.ones((1, 1, 2, 2, 3), np.float32))
        response = types.SimpleNamespace(observe=lambda frames, index, end:
                                         (np.eye(4), {}, np.eye(4)))
        rows = lambda lo, hi: (lo, hi)
        roller = types.SimpleNamespace(wp=types.SimpleNamespace(pipe=pipe),
                                       response=response, wp_rows=rows)
        observe = response.observe
        frames = [np.zeros((2, 2, 3), np.uint8)]
        before = rng_digest()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError, 'test error'):
                with worldplay_trace(roller, Path(td) / 'trace.jsonl'):
                    self.assertEqual(roller.wp_rows(0, 4), (0, 4))
                    response.observe(frames, 3, 8)
                    raise RuntimeError('test error')
        self.assertIs(roller.wp_rows, rows)
        self.assertIs(response.observe, observe)
        self.assertEqual(before, rng_digest())
        self.assertFalse(frames[0].any())
        self.assertFalse(pipe.transformer._forward_hooks)


if __name__ == '__main__':
    unittest.main()
