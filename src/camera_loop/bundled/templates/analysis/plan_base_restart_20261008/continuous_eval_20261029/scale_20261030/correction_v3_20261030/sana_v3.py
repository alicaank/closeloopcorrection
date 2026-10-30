"""SANA adapter with opt-in quality-checked external observer, unchanged model code."""
import argparse,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(add_help=False);p.add_argument('--observer-quality',choices=['report','enforce'],default='enforce');a,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
os.environ['GV_OBSERVER_QUALITY']=a.observer_quality
from checked_common import install
install()
meta=json.loads((R/'source_manifest.json').read_text())['sana_run.py']
exec(compile((R/'snapshots/sana_run.py').read_text(),meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original']))
