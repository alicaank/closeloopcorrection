"""Lyra launcher selecting the guarded observer in its child inference process."""
import argparse,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(add_help=False);p.add_argument('--observer-quality',choices=['report','enforce'],default='enforce');a,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
os.environ['GV_OBSERVER_QUALITY']=a.observer_quality
meta=json.loads((R/'source_manifest.json').read_text())['lyra_run.py'];source=(R/'snapshots/lyra_run.py').read_text()
needle="str(R/'lyra_plugin.py')";assert source.count(needle)==1;source=source.replace(needle,repr(str(R/'lyra_plugin_v3.py')))
exec(compile(source,meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original']))
