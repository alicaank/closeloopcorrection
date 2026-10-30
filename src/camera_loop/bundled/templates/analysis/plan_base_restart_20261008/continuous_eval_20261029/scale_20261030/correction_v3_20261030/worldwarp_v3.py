"""Isolated two-chunk scale continuity pilot. No implicit repair of changed depth gauge."""
import argparse,json,sys,os
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(add_help=False);p.add_argument('--continuation-check',choices=['report','enforce'],default='enforce');p.add_argument('--max-continuation-scale-ratio',type=float,default=1.2)
p.add_argument('--observer-quality',choices=['report','enforce'],default='report')
a,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
chunks=int(rest[rest.index('--chunks')+1]) if '--chunks' in rest else 2
if chunks!=2:raise ValueError('This validation pilot requires exactly two chunks; later chunks need separately matched anchor depths')
os.environ['GV_OBSERVER_QUALITY']=a.observer_quality
from checked_common import install
install()
meta=json.loads((R/'source_manifest.json').read_text())['worldwarp_run.py'];source=(R/'snapshots/worldwarp_run.py').read_text()
old='def calibrate_native(camera_poses,depth_maps,is_first_chunk):';assert source.count(old)==1
source=source.replace(old,'def calibrate_native(camera_poses,depth_maps,is_first_chunk,video_tensor):')
needle="   assert scale_state, 'Initial anchor scale must be frozen before continuation'\n   return camera_poses"
assert source.count(needle)==1
source=source.replace(needle,"   assert scale_state, 'Initial anchor scale must be frozen before continuation'\n   from continuation_guard import check_native\n   check_native(depth_maps,depth,video_tensor,anchor,a.out,mode=V3_MODE,max_ratio=V3_MAX_RATIO)\n   return camera_poses")
needle='_calibrate_native(camera_poses, depth_maps, is_first_chunk)';assert source.count(needle)==1
source=source.replace(needle,'_calibrate_native(camera_poses, depth_maps, is_first_chunk, video_tensor)')
# The compiled native method resolves the callback via its captured scope.
exec(compile(source,meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original'],V3_MODE=a.continuation_check,V3_MAX_RATIO=a.max_continuation_scale_ratio))
