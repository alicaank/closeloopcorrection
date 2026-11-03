"""Reuse qualified immutable RGB or generate and retain a fresh matched arm."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from evidence import artifact_files, eligibility, receipt_paths, sha

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--case', required=True)
p.add_argument('--arm', choices=['B','S','L','M'], required=True)
a = p.parse_args()
R = a.root.resolve()
sys.path.insert(0,str(R/'source'))
from score_memory_pair import load_frames
from camera_loop.lossless_archive import archive_rgb
m = json.loads((R/'manifest.json').read_text())
c = m['protocol']['cases'][a.case]
spec = m['arm_jobs'][a.case][a.arm]
binding_path = R/'bindings'/f'{a.case}_{a.arm}.json'
if binding_path.exists(): raise FileExistsError('Preserve completed output binding')
reason = 'new_intervention_or_sham'
reused = False
if a.arm in ['B','M']:
    reused,reason = eligibility(c,a.arm,spec['reference'],m['protocol']['parent_overlay_hashes'])
directory = Path(spec['reference']) if reused else Path(spec['output'])
if not reused:
    subprocess.run(spec['argv'],check=True)
    meta = json.loads((directory/'run.json').read_text())
    expected_arm = {'B':'baseline','S':'sham','L':'correction','M':'baseline'}[a.arm]
    if (meta.get('model'),meta.get('scene'),meta.get('seed'),meta.get('arm')) != (
            c['model'],c['scene'],c['seed'],expected_arm):
        raise ValueError('Fresh run identity differs from requested scientific arm')
    policy = 'measured' if a.arm=='M' else 'commanded'
    for kind,path in receipt_paths(c['model'],directory).items():
        r = json.loads(path.read_text())
        if not r.get('completed') or r.get('memory_policy')!=policy:
            raise ValueError('Fresh native host did not complete with the requested memory policy')
        if r.get('overlay_sha256')!=m['protocol']['copied_overlay_hashes'][kind]:
            raise ValueError('Fresh host source receipt differs from frozen generated source')
frames,fmt = load_frames(directory,c['scene'])
if not fmt.startswith('lossless') or frames.dtype!=np.uint8 or len(frames)!=c['frames']:
    raise ValueError('Incomplete or non-lossless output cannot be reused or scored')
rgb_hash = hashlib.sha256(frames.tobytes()).hexdigest()
if reused and (directory/'retention.json').is_file():
    retained = json.loads((directory/'retention.json').read_text())
    if retained.get('pixel_exact') is not True or retained.get('rgb_sha256')!=rgb_hash:
        raise ValueError('Reference lossless archive failed its retained raw-RGB hash')
if not reused:
    if not directory.resolve().is_relative_to(R/'runs'):
        raise ValueError('Retention must be restricted to newly owned outputs')
    raw = [directory/'frames.npz'] if fmt=='lossless_npz' else sorted((directory/(c['scene']+'_frames')).glob('frame_*.png'))
    if not raw: raise ValueError('Expected fresh native lossless files')
    receipt = archive_rgb(frames,directory/'frames.lossless.mkv',
        Path('/home/hj/Alican/analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg'))
    receipt.update(replaced_files=[str(p) for p in raw],original_bytes=sum(p.stat().st_size for p in raw))
    (directory/'retention.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for path in raw: path.unlink()
binding = dict(case=a.case,arm=a.arm,directory=str(directory),reused=reused,decision=reason,
    frames=len(frames),rgb_sha256=rgb_hash,shape=list(frames.shape),
    files={str(path):sha(path) for path in artifact_files(c['model'],directory)},
    parent_source_manifest_sha256=m['protocol']['parent_manifest_sha256'],
    comparison_qualified=False,note='Per-arm source/RGB qualification; shared-prefix gate remains required')
binding_path.write_text(json.dumps(binding,indent=2)+'\n')
print(json.dumps({k:v for k,v in binding.items() if k not in ['files']}))
