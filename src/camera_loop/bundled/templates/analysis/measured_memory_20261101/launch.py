"""Run a released adapter with measured-memory support in its native environment.

python launch.py MODEL --memory-policy measured -- ORIGINAL_DRIVER_ARGUMENTS
Use baseline/native commands to isolate memory; correction also enables the
existing driver's future-command controller. No calibration probes are run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from overlays import DRIVERS,driver_source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=DRIVERS)
    parser.add_argument('--memory-policy',choices=['commanded','identity','measured'],default=os.environ.get('CAMERA_MEMORY_POLICY','measured'))
    parser.add_argument('--check-source',action='store_true')
    argv=sys.argv[1:];split=argv.index('--') if '--' in argv else len(argv)
    args=parser.parse_args(argv[:split]);driver_args=argv[split+1:]
    if any(x=='--memory-policy' or x.startswith('--memory-policy=') for x in driver_args):
        parser.error('Set --memory-policy before -- so the run receipt and native policy agree')
    os.environ['CAMERA_MEMORY_POLICY']=args.memory_policy
    source=driver_source(args.model);path=DRIVERS[args.model]
    if args.check_source:
        print(json.dumps(dict(model=args.model,source=str(path),sha256=hashlib.sha256(source.encode()).hexdigest(),compiled=True)));return
    sys.argv=[str(path)]+driver_args
    # Match Python's ordinary script-directory import behavior, including
    # native helper imports made before a driver adds its own search paths.
    sys.path.insert(0,str(path.parent))
    # Keep the historical driver's arm/command receipt and label memory policy
    # separately: baseline + measured is a memory-only intervention.
    out=None
    for flag in ['--out','--output_path']:
        if flag in driver_args:out=Path(driver_args[driver_args.index(flag)+1]).resolve();break
    record_allowed=out is not None and (not out.exists() or args.model.endswith('_child'))
    receipt=dict(model=args.model,memory_policy=args.memory_policy,source=str(path),
        overlay_sha256=hashlib.sha256(source.encode()).hexdigest(),completed=False)
    try:
        exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
        receipt['completed']=True
    finally:
        if record_allowed and out.is_dir():
            filename='memory_policy_'+args.model+'.json'
            if not (out/filename).exists():
                with (out/filename).open('x') as f:f.write(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
