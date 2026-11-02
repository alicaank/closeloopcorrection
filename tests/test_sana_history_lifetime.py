import contextlib
import importlib.util
from pathlib import Path
import types
import unittest
import weakref
import numpy as np
import camera_loop


# Minimal resource-lifetime fixture; GPU qualification also runs the exact
# released history-update block in the installed native source.
SOURCE='''def update(self):
        kv_prefix_per_layer = [{'k':p[0], 'v':p[1]} for p in self.history]
        prefix = kv_prefix_per_layer[-1]
        hk = self.history[-1]
        captured_kv_post = None
        block_kv_post = [(np.ones((2,3))*i, np.ones((2,3))*(i+1)) for i in range(self._n_layers)]
        with profiler.section("history_update"):
            for layer_idx in range(self._n_layers):
                raw_pair = block_kv_post[layer_idx]
                raw_k, raw_v = raw_pair
                new_k, new_v = raw_k, raw_v
                old = self.history[layer_idx]
                self.history[layer_idx] = (np.concatenate([old[0],new_k]),np.concatenate([old[1],new_v]))
                raw_k = None
                raw_v = None
                self.live_old.append(sum(r() is not None for r in self.refs))
        return self.history
'''.replace('\n    ','\n')


class SanaHistoryLifetime(unittest.TestCase):
    def patch(self, source):
        p=Path(camera_loop.__file__).parent/'bundled/templates/analysis/plan_base_restart_20261008/new_models_loop_20261028/sana_kv_offload.py'
        spec=importlib.util.spec_from_file_location('sana_lifetime',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        return m.release_history_temporaries(source)

    def run_fixture(self, source):
        history=[(np.full((3,3),i),np.full((3,3),-i)) for i in range(5)]
        obj=types.SimpleNamespace(history=history,_n_layers=5,live_old=[],refs=[weakref.ref(x) for row in history for x in row])
        scope=dict(np=np,profiler=types.SimpleNamespace(section=lambda name:contextlib.nullcontext()))
        exec(source,scope);result=scope['update'](obj)
        return result,obj.live_old

    def test_releases_consumed_old_history_without_changing_values(self):
        original,old_live=self.run_fixture(SOURCE)
        updated,new_live=self.run_fixture(self.patch(SOURCE))
        for a,b in zip(original,updated):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(old_live[-1],10)
        self.assertEqual(new_live[-1],0)
        self.assertEqual(new_live,[8,6,4,2,0])

    def test_native_source_change_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError,'contract changed'):
            self.patch(SOURCE.replace('raw_v = None','raw_v = 1'))


if __name__=='__main__':unittest.main()
