"""Freeze decoder attribution on all four original MatrixGame short pairs."""
from pathlib import Path
import hashlib,json,time,py_compile
R=Path(__file__).resolve().parent;W=Path('/home/hj/Alican');OLD=W/'analysis/measured_memory_v1_20261101'
REPO=W/'sota_repos/matrix_game/Matrix-Game-3.5';PY=W/'baselines/venv_mg35/bin/python'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
assert not (R/'manifest.json').exists()
assert json.loads((R/'cpu_load.json').read_text())['passed']
parent=json.loads((OLD/'freeze.json').read_text())
pairs=[p for p in parent['pairs'] if p['model']=='matrixgame'];assert len(pairs)==4
jobs=[];inputs={};specs=[]
for pair in pairs:
    for arm in ('commanded','measured'):
        root=Path(pair[arm]);files=list((root/'raw/run/validation/_temp').glob('*history_temp.pth'));assert len(files)==1
        rgb=root/'frames.npz';assert rgb.is_file()
        name=pair['key']+'_'+arm
        selected={str(p):sha(p) for p in [files[0],rgb,OLD/'scores'/pair['key']/'score.json',root/'stream_export.json']}
        spec=dict(case=pair['key'],arm=arm,latents=str(files[0]),original_rgb=str(rgb),inputs=selected,out=str(R/'runs'/name))
        sp=R/'specs'/f'{name}.json';sp.write_text(json.dumps(spec,indent=2)+'\n');specs.append(sp);inputs.update(selected)
        jobs.append(dict(name=name,case=name,argv=[str(PY),str(R/'decode.py'),'--spec',str(sp)],required=True))
checkpoint=REPO/'checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth';inputs[str(checkpoint)]=sha(checkpoint)
config=dict(scope='Offline decoder attribution only. All four original MatrixGame short-cohort pairs, both arms; no additional video generation and no replacement of causal outputs/scores.',
    cases=[p['key'] for p in pairs],arms=['commanded','measured'],frames=169,
    control='Decode saved latents [0:22] and [21:43] separately; concatenate first 85 RGB frames and last 84 RGB frames. Require exact original RGB equality before attributing any difference.',
    treatment='Decode the same complete 43-latent history with the released decoder and identical BF16/spatial tiling.',
    measurements=['Exact block-replay control','Equality of previously emitted first 85 RGB frames','Per-frame joint-vs-stream RGB difference','Adjacent-frame RGB difference, mean and contrast around seam 84→85','Lossless archived joint output for visual review'],
    limitations='An offline export change is not a deployed causal correction. RGB differences are not camera velocities. No new aggregate method efficacy claim is made.',
    resource='VAE-only loading, shared GPU locks, 4 GiB launch and 2 GiB runtime disk/host reserve. Original native inputs and outputs preserved.')
(R/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
sources={p:h for p,h in parent['sources'].items() if Path(p).is_relative_to(REPO)}
for p,h in sources.items():assert sha(p)==h,p
for p in [*R.glob('*.py'),R/'protocol.json',R/'cpu_load.json',*specs,OLD/'freeze.json']:
    sources[str(p)]=sha(p)
sources.update(inputs)
for p in R.glob('*.py'):py_compile.compile(str(p),doraise=True)
m=dict(protocol=config,jobs=jobs,sources=sources,created=time.time())
(R/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(R/'cpu_validation.json').write_text(json.dumps(dict(passed=True,manifest_sha256=sha(R/'manifest.json'),checks=['Actual released BF16 VAE-only CPU load','All eight saved latent/RGB inputs present and hashed','Native source freeze checked','Diagnostic scripts compile'],gpu_decode_qualified=False),indent=2)+'\n')
print(json.dumps(dict(jobs=len(jobs),native_sources=len(sources),new_generation_jobs=0)))
