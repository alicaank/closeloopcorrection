import unittest
import numpy as np
import torch
from camera_loop.video_conversion import history_rgb


class HistoryConversion(unittest.TestCase):
    def test_matches_each_legacy_formula_including_rounding_and_clipping(self):
        # Include quantization ties, clipping, and noncontiguous history views.
        values = torch.cat((torch.linspace(-1.2, 1.2, 258),
                            (torch.arange(256) + .5) / 127.5 - 1))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            x = values.to(dtype).repeat(3 * 19 * 2).reshape(3, 19, 2, -1)[:, ::2]
            original = x.clone()
            for formula in ('endpoint', 'memory'):
                y = x.cpu().permute(1, 2, 3, 0).float()
                expected = (((y*.5+.5).clamp(0,1)*255).round().byte() if formula == 'endpoint'
                            else ((y+1)*127.5).round().clamp(0,255).byte()).numpy()
                for chunk in (1, 3, 8, 100):
                    np.testing.assert_array_equal(history_rgb(x, formula=formula, chunk_frames=chunk), expected)
            self.assertTrue(torch.equal(x, original))

    def test_empty_history_and_invalid_contracts(self):
        x = torch.empty(3, 0, 2, 4)
        self.assertEqual(history_rgb(x, formula='memory').shape, (0, 2, 4, 3))
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError): history_rgb(x, formula='memory', chunk_frames=size)
        with self.assertRaises(ValueError): history_rgb(x, formula='unknown')
        with self.assertRaises(ValueError): history_rgb(torch.empty(2, 3, 4), formula='memory')
