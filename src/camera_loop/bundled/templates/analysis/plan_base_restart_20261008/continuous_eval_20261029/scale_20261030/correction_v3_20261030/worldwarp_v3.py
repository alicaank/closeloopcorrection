"""Runtime shared-frame unit transfer at every WorldWarp geometry window."""
import argparse,json,sys,os
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--continuation-check',choices=['enforce'],default='enforce')
p.add_argument('--observer-quality',choices=['report','enforce'],default='report')
a,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
os.environ['GV_OBSERVER_QUALITY']=a.observer_quality
from checked_common import install
install()
meta=json.loads((R/'source_manifest.json').read_text())['worldwarp_run.py']
source=(R/'snapshots/worldwarp_run.py').read_text()
exec(compile(source,meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original']))
