import argparse,json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'source'))
from score_memory_pair import load_frames
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--gate',choices=['identity','matched'],required=True);a=p.parse_args()
case=json.loads((R/'cases.json').read_text())[a.case]
def source(arm):return R/'runs'/a.case/arm/'baseline'
def read(arm,name):return json.loads((source(arm)/name).read_text())
def frames(arm):return load_frames(source(arm),case['scene'])[0]
passed=True;checks={}
if a.gate=='identity':
    baseline,identity=frames('A'),frames('I')
    checks['full_77_rgb_exact']=bool(len(baseline)==len(identity)==77 and np.array_equal(baseline,identity))
    for chunk in [0,4]:
        x,y=read('A',f'cache_{chunk:02d}.json'),read('I',f'cache_{chunk:02d}.json')
        for k in ['self_cache','cross_cache','native_generator_sha256']:checks[f'chunk_{chunk}_{k}']=x[k]==y[k]
    x,y=read('A','gauge_receipt.json'),read('I','gauge_receipt.json')
    for k in ['initial_noise_sha256','first_generated_latent_sha256','future_camera_sha256','clean_camera_sha256']:checks[k]=x[k]==y[k]
else:
    baseline=frames('A');g,m=frames('G'),frames('M')
    checks['shared_emitted_prefix']=bool(len(baseline)==len(g)==len(m)==77 and np.array_equal(baseline[:13],g[:13]) and np.array_equal(g[:13],m[:13]))
    x,y=read('G','gauge_receipt.json'),read('M','gauge_receipt.json')
    for k in ['initial_noise_sha256','first_generated_latent_sha256','observed_rgb_sha256','future_camera_sha256','denominator','observation_accepted']:checks[k]=x[k]==y[k]
    if x['observation_accepted'] and y['observation_accepted']:
        checks['same_accepted_measurement']=bool(np.array_equal(np.asarray(x['measured_poses']),np.asarray(y['measured_poses'])))
    for arm in ['A','I','G','M']:
        r=read(arm,'gauge_receipt.json');checks[arm+'_rng_preserved']=r['rng_before']==r['rng_after']
passed=all(checks.values());out=R/'gates';out.mkdir(exist_ok=True)
result=dict(passed=passed,case=a.case,gate=a.gate,checks=checks)
(out/f'{a.case}_{a.gate}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if not passed:raise SystemExit('Required attribution gate failed')
