"""Resume only terminal disk holds; retain original manifests, statuses and logs.

Derive a runner from each frozen runner. Change only status/log destination and
skip already completed jobs. Runtime paths, job argv, source verification,
environment, resource guards and GPU locks stay as originally declared.
"""
from pathlib import Path
import difflib
import hashlib
import json
import time

import psutil

W = Path('/home/hj/Alican')
A = Path(__file__).resolve().parent
ROOTS = {
    'matrix_memory': W/'analysis/memory_generalization_20261103/matrix_observer_lifetime',
    'legacy_dreamx': W/'analysis/legacy_rejoin_matched_20261103/dreamx_host',
    **{f'legacy_{m}': W/f'analysis/legacy_rejoin_matched_20261103/hosts/{m}' for m in ['lingbot','sana','matrixgame','lyra']},
    'lyra_memory': W/'analysis/memory_generalization_20261103/lyra_observer_resources',
    'sana_transition': W/'analysis/memory_generalization_20261103/sana_observer_transition',
    'dreamx_future': W/'analysis/dreamx_future_command_20261103',
    'lingbot_gauge': W/'analysis/lingbot_shared_gauge_20261103/pilot',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    prepared = []
    for label, root in ROOTS.items():
        destination = A/label
        assert not destination.exists(), destination
        status = json.loads((root/'status.json').read_text())
        manifest = json.loads((root/'manifest.json').read_text())
        proof = json.loads((root/'cpu_validation.json').read_text())
        assert status['phase'] == 'held' and not psutil.pid_exists(status['pid'])
        assert status['failures'] == ["RuntimeError('4 GiB launch reserve; originals retained')"]
        assert not status['held_cases']
        assert proof['passed'] and proof['manifest_sha256'] == sha(root/'manifest.json') == status['manifest_sha256']
        for path, digest in manifest['sources'].items():
            assert sha(path) == digest, path
        completed = []
        logs = {}
        for job in manifest['jobs']:
            previous = status['jobs'].get(job['name'])
            if previous is None:
                continue
            assert previous['state'] == 'completed' and previous['returncode'] == 0 and previous['stop_reason'] is None
            assert previous['argv'] == job['argv']
            completed.append(job['name'])
            path = root/'logs'/f"{job['name']}.log"
            logs[str(path)] = sha(path)
        assert len(completed) == len(status['jobs'])
        source = (root/'run_queue.py').read_text()
        modified = replace_once(source, "R=Path(__file__).resolve().parent", f"R=Path({str(root)!r});A=Path(__file__).resolve().parent")
        # No job/runtime/output destination is rewritten.
        for name in ['status.json','status.tmp']:
            modified = modified.replace(f"R/'{name}'", f"A/'{name}'")
        modified = modified.replace("R/'logs'", "A/'logs'")
        check = (f"assert sha(R/'status.json') == {sha(root/'status.json')!r}\n"
                 f"        assert sha(R/'run_queue.py') == {sha(root/'run_queue.py')!r}\n"
                 "        assert sha(A/'resume.json') == RESUME_SHA\n        ")
        modified = replace_once(modified, 'def verify():\n        ', 'def verify():\n        '+check)
        modified = replace_once(modified, "for entry in m['jobs']:\n", f"for entry in m['jobs']:\n            if entry['name'] in {set(completed)!r}:continue\n")
        modified = replace_once(modified, "def save():\n", "state['resume_provenance']=json.loads((A/'resume.json').read_text())\n    def save():\n")
        destination.mkdir();(destination/'logs').mkdir()
        record = dict(label=label, original_root=str(root), original_pid=status['pid'], original_status_sha256=sha(root/'status.json'),
                      original_manifest_sha256=sha(root/'manifest.json'), original_runner_sha256=sha(root/'run_queue.py'),
                      completed_jobs_skipped=completed, completed_log_sha256=logs,
                      remaining_jobs=[j['name'] for j in manifest['jobs'] if j['name'] not in completed],
                      unchanged=['job argv','native runtime/source paths','environment','4 GiB launch reserve','2 GiB disk/host runtime reserve','five GPU locks','model protocol','horizons','seeds','output paths'],
                      changed=['separate attempt status/log paths','skip original successfully completed jobs'], prepared=time.time())
        (destination/'resume.json').write_text(json.dumps(record,indent=2)+'\n')
        modified = replace_once(modified, 'def main():\n', f"RESUME_SHA={sha(destination/'resume.json')!r}\ndef main():\n")
        compile(modified,str(destination/'run_queue.py'),'exec')
        (destination/'run_queue.py').write_text(modified)
        (destination/'source_change.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile=str(root/'run_queue.py'),tofile=str(destination/'run_queue.py'))))
        prepared.append(dict(label=label, root=str(destination), runner_sha256=sha(destination/'run_queue.py'),
                             resume_sha256=sha(destination/'resume.json'), skipped=len(completed), remaining=len(record['remaining_jobs']),
                             frozen_source_files_verified=len(manifest['sources'])))
    (A/'preparation.json').write_text(json.dumps({'passed':True,'attempts':prepared},indent=2)+'\n')
    print(json.dumps(prepared,indent=2))


if __name__ == '__main__':
    main()
