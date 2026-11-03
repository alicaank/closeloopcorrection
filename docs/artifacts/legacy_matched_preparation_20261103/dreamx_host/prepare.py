"""Freeze DreamX's eight original cases within the five-model matched study."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
W = Path('/home/hj/Alican')
R = Path(__file__).resolve().parent
P = W/'analysis/memory_generalization_20261103/cohort_resume'
B = W/'analysis/dreamx_camera_grid_20261103/batch'
PY = W/'baselines/venv_vmem/bin/python'
NATIVE = W/'analysis/plan_base_restart_20261008/model_readiness_20261027/envs/dreamx/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20), b''): h.update(block)
    return h.hexdigest()

assert not (R/'manifest.json').exists(), 'Preserve an existing frozen study'
parent = json.loads((P/'freeze.json').read_text())
for path,h in {**parent['sources'],**parent['inputs']}.items():
    assert sha(path)==h,path
prepared = json.loads((B.parent/'qualification.json').read_text())
verified = json.loads((B.parent/'verification.json').read_text())
assert prepared['passed'] and verified['passed']
assert verified['preparation_receipt_sha256']==sha(B.parent/'qualification.json')
assert verified['all_prepared_output_hashes_unchanged']
for path,h in prepared['outputs'].items(): assert sha(B/path)==h,path
# The existing post-preparation verification qualifies an identity-map guard
# added after export. Bind that exact verified revision, never arbitrary drift.
grid = str(W/'camera-closed-loop/src/camera_loop/camera_grid.py')
prepared_sources = dict(prepared['sources'])
prepared_sources[grid] = verified['current_camera_grid_sha256']
for path,h in prepared_sources.items(): assert sha(path)==h,path
original = parent['protocol']
cases = {}
for seed in original['seeds']:
    for scene in original['scenes']:
        key = f'dreamx_{scene}_s{seed}'
        cases[key] = dict(model='dreamx',scene=scene,seed=seed,batch=str(B),frames=69,
            latents=18,prefix_latents=6,first_affected_frame=21)
assert len(cases)==8
protocol = dict(method='dreamx_legacy_memory_matched_camera_grid_v1',cases=cases,
    scope='All eight original DreamX cases within the full 40-case, five-model legacy comparison. Other models remain required.',
    arms={'A':'Observer-free native commands and cache',
          'S':'Same causal observer as L/M, native commands and cache',
          'L':'Historical rejoin curve; temporary measured encoding anchor; future chunk committed after native KV pass',
          'M':'Measured-history camera labels only during the native context-noise KV pass; future commands unchanged'},
    primary_comparisons=['S versus L','S versus M','L versus M'],
    controls='A/S must be raw-RGB and native-receipt identical for the full 69 frames. All arms use the same prepared anchor grid, BF16 native intrinsics, scene, seed, horizon and targets.',
    observation='Shared query of the current chunk camera labels and preceding camera anchor, plus fixed frame zero; full emitted causal RGB prefix; before the context-noise cache pass. All nonterminal boundaries starting at latent 6.',
    changed_from_historical=['Initial camera-grid correction','Shared full causal observation window rather than truncating three available RGB frames','Observation before context-noise pass for all observed arms','Explicit registration/SE3 quality rejection; no correction on rejection'],
    policy='Unmodified legacy rejoin function and immediate first-boundary correction. No speed cap, non-reversal projection, learned gain, probes or inverse control.',
    gates=['Complete A/S raw RGB, initial input/noise and all native boundary receipts exact',
           'S/L/M raw frames 0 through 20 exact; native receipts exact through the first pre-cache intervention point',
           'L never changes context conditioning; M never changes future camera tensors'],
    scoring='One independent offline VIPE contract; primary mean fixed-time position frames 21–68. Also whole-rollout, endpoint and angular errors, invalid registrations, actuator counts and visual quality. Original reference poses may be VIPE-derived, not independent ground truth.',
    qualification='CPU checks establish encoding/actuator contracts, not GPU parity, differential authority or efficacy.',
    storage='Retain exact raw RGB as round-trip-verified FFV1; 4 GiB launch / 2 GiB runtime disk guard, 2 GiB host-memory guard. Do not overwrite older attempts.')
(R/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
env = dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=os.pathsep.join([str(R/'source'),str(R)]),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for name,argv in [
    ('final_cpu_tests',[str(PY),str(R/'test_boundary_hook.py')]),
    ('source_compile',[str(PY),str(R/'qualify_source.py')]),
    ('native_cli',[str(NATIVE),str(R/'run.py'),'--help'])]:
    with (R/'logs'/f'{name}.log').open('x') as f:
        subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
jobs = []
for key,c in cases.items():
    for arm in ['A','S','L','M']:
        if arm=='L':
            jobs.append(dict(name=key+'_identity',case=key,argv=[str(PY),str(R/'check.py'),'--case',key,'--kind','identity'],required=True))
        jobs.append(dict(name=key+'_'+arm,case=key,
            argv=[str(NATIVE),str(R/'run.py'),'--batch',str(B),'--scene',c['scene'],'--seed',str(c['seed']),
                '--out',str(R/'runs'/key/arm),'--arm',arm,'--latents','18','--prefix-latents','6'],
            required=key==next(iter(cases))))
        jobs.append(dict(name=key+'_'+arm+'_retain',case=key,
            argv=[str(PY),str(R/'retain.py'),'--case',key,'--arm',arm],required=True))
    jobs.append(dict(name=key+'_attribution',case=key,argv=[str(PY),str(R/'check.py'),'--case',key,'--kind','attribution'],required=True))
    jobs.append(dict(name=key+'_score',case=key,argv=[str(PY),str(R/'score.py'),'--case',key],required=False))
sources = {p:h for p,h in parent['sources'].items() if not Path(p).is_relative_to(P)}
sources.update(prepared_sources)
files = [*R.glob('*.py'),R/'protocol.json',R/'source_validation.json',P/'freeze.json',
    B.parent/'qualification.json',B.parent/'verification.json',
    *[p for p in (R/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts],
    *[p for p in B.rglob('*') if p.is_file()]]
sources.update({str(p):sha(p) for p in files})
manifest = dict(protocol=protocol,jobs=jobs,sources=sources,created=time.time(),
    parent_scientific_freeze_sha256=sha(P/'freeze.json'))
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
proof = dict(passed=True,manifest_sha256=sha(R/'manifest.json'),gpu_qualified=False,
    checks=['Eight actual-native-encoder CPU tests','Runtime overlay compiles with pre/post cache separation',
            'Native interpreter CLI','Parent freeze and all four calibrated scene inputs verified'])
(R/'cpu_validation.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(cases=len(cases),generations=32,jobs=len(jobs),sources=len(sources),manifest_sha256=proof['manifest_sha256'])))
