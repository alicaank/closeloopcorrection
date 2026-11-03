"""Replace closed, byte-identical analysis artifacts with hardlinks; keep paths."""
import collections
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

ROOT = Path('/home/hj/Alican')
HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def open_inodes():
    found = set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try:
            for fd in (process/'fd').iterdir():
                try:
                    s = fd.stat(); found.add((s.st_dev, s.st_ino))
                except OSError: pass
        except OSError: pass
    return found


def main():
    receipt_path = HERE/'receipt.json'
    if receipt_path.exists(): raise FileExistsError('Existing execution receipt retained')
    plan = HERE/'hash_candidates.json'; groups = json.loads(plan.read_text())['groups']
    opened = open_inodes()
    receipt = dict(started=time.time(), plan_sha256=sha(plan),
                   scope='Closed regular .npz/.mp4/.mkv files inside analysis only; no checkpoint trees.',
                   free_before=shutil.disk_usage(ROOT).free, groups=[], skipped=[],
                   replaced_paths=0, released_allocated_bytes=0)
    def save():
        tmp=HERE/'receipt.tmp'; tmp.write_text(json.dumps(receipt,indent=2)+'\n'); tmp.replace(receipt_path)
    save()
    for group in groups:
        paths = [Path(x['path']) for x in group['files']]
        states = {p:p.stat() for p in paths}
        if any(p.is_symlink() or not p.resolve().is_relative_to(ROOT/'analysis')
               or p.suffix not in ('.npz','.mp4','.mkv') or not stat.S_ISREG(states[p].st_mode)
               for p in paths): raise ValueError('Outside authorized artifact scope')
        if any((s.st_dev,s.st_ino) in opened for s in states.values()):
            receipt['skipped'].append(dict(sha256=group['sha256'],reason='open file handle')); continue
        if len({(s.st_dev,s.st_mode,s.st_uid,s.st_gid) for s in states.values()}) != 1:
            receipt['skipped'].append(dict(sha256=group['sha256'],reason='device or permission mismatch')); continue
        for p in paths:
            if states[p].st_size != group['size'] or sha(p) != group['sha256']:
                raise ValueError('Candidate changed: '+str(p))
        source = max(paths, key=lambda p:states[p].st_nlink)
        source_inode = (states[source].st_dev,states[source].st_ino)
        old_counts = collections.Counter((s.st_dev,s.st_ino) for s in states.values())
        entry = dict(sha256=group['sha256'], source=str(source), replacements=[])
        receipt['groups'].append(entry); save()
        for p in paths:
            before=states[p]
            if (before.st_dev,before.st_ino)==source_inode: continue
            current=p.stat()
            if (current.st_ino,current.st_size,current.st_mtime_ns)!=(before.st_ino,before.st_size,before.st_mtime_ns):
                raise ValueError('File changed during verification: '+str(p))
            temp=p.with_name(p.name+f'.exact-dedup-{os.getpid()}')
            if temp.exists(): raise FileExistsError(temp)
            os.link(source,temp)
            try: os.replace(temp,p)
            finally: temp.unlink(missing_ok=True)
            assert p.stat().st_ino==source.stat().st_ino and sha(p)==group['sha256']
            entry['replacements'].append(dict(path=str(p),old_inode=before.st_ino,new_inode=p.stat().st_ino,size=before.st_size))
            receipt['replaced_paths']+=1; save()
        for inode,count in old_counts.items():
            if inode==source_inode:continue
            s=next(s for s in states.values() if (s.st_dev,s.st_ino)==inode)
            if count==s.st_nlink:receipt['released_allocated_bytes']+=s.st_blocks*512
        save()
    for group in receipt['groups']:
        for path in [group['source'],*[r['path'] for r in group['replacements']]]:
            assert sha(Path(path))==group['sha256'],path
    receipt.update(finished=time.time(),free_after=shutil.disk_usage(ROOT).free,all_paths_and_sha256_preserved=True)
    save()
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('groups','skipped')}))


if __name__=='__main__': main()
