"""Run the frozen WorldPlay host with opt-in read-only diagnostic hooks."""
import importlib.util
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
host = R / 'hosts/analysis/plan_base_restart_20261008/fair_common_20261025/wp_fair.py'
spec = importlib.util.spec_from_file_location('worldplay_identity_host', host)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from camera_loop.native_trace import worldplay_trace

original = module.Roller.run


def traced_run(self):
    assert self.response is not None
    assert self.arm['name'] in ('A', 'S'), 'Diagnostic never enables correction'
    with worldplay_trace(self, self.response.out / 'native_trace.jsonl'):
        return original(self)


module.Roller.run = traced_run
if __name__ == '__main__':
    module.main()
