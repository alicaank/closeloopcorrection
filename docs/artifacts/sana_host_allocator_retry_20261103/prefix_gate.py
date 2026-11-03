"""Require full 121-frame completion and both preserved baseline prefixes."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
sys.path[:0]=[str(R/'source'),str(R/'helpers')]
from score_memory_pair import load_frames
from evidence import verify_binding
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--case',required=True);a=p.parse_args()
assert a.root.resolve()==R
m=json.loads((R/'manifest.json').read_text());c=m['protocol']['cases'][a.case]
binding=json.loads((R/'bindings'/f'{a.case}_B.json').read_text());verify_binding(binding)
frames,fmt=load_frames(Path(binding['directory']),c['scene'])
assert fmt.startswith('lossless') and len(frames)==121
checks=[]
for ref in m['protocol']['resource_retry']['prefix_references']:
    path=Path(ref['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    with np.load(path) as z:original=z['frames']
    exact=bool(np.array_equal(frames[:len(original)],original))
    checks.append(dict(reference=str(path),frames=len(original),exact=exact))
result=dict(passed=all(x['exact'] for x in checks),complete_frames=len(frames),checks=checks)
(R/'gates'/f'{a.case}_resource_prefix.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
if not result['passed']:raise SystemExit('Resource retry changed preserved baseline RGB')
