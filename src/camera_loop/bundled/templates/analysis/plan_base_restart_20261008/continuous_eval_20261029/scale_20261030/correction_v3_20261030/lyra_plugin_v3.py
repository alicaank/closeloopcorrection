import json
from pathlib import Path
R=Path(__file__).resolve().parent
from checked_common import install
install()
meta=json.loads((R/'source_manifest.json').read_text())['lyra_plugin.py']
exec(compile((R/'snapshots/lyra_plugin.py').read_text(),meta['original'],'exec'),dict(__name__='__main__',__file__=meta['original']))
