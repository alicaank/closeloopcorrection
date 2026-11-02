import argparse,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R),str(R/'source')]
p=argparse.ArgumentParser();p.add_argument('--model',choices=['sana','lyra'],required=True);p.add_argument('--arm',choices=['M','F'],required=True);p.add_argument('--batch',required=True);p.add_argument('--scene',required=True);p.add_argument('--seed',required=True);p.add_argument('--out',required=True);p.add_argument('--check-source',action='store_true');a=p.parse_args()
os.environ.update(LONG_BATCH=a.batch,MEMORY_RATE_ARM=a.arm,MEMORY_RATE_OUT=a.out,CAMERA_MEMORY_POLICY='measured',GV_SCALE_CACHE=str(Path(a.out).parent/'scale_calibration'))
import rate_hooks,overlays
rate_hooks.install(a.model)
source=rate_hooks.patch_source(a.model,overlays.driver_source(a.model));path=overlays.DRIVERS[a.model]
sys.path.insert(0,str(path.parent));common=['--batch',a.batch,'--scene',a.scene,'--seed',a.seed,'--out',a.out,'--arm','baseline','--frames','49' if a.model=='sana' else '161']
if a.check_source:
    compile(source,str(path),'exec')
    if a.model=='lyra':compile(rate_hooks.patch_source('lyra_child',overlays.driver_source('lyra_child')),str(overlays.DRIVERS['lyra_child']),'exec')
    print('Compiled',a.model)
else:
    sys.argv=[str(path)]+common;exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
