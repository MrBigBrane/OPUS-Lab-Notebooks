#!/usr/bin/env python3
"""Package only small result/diagnostic files, never model weights or GPU caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compression',choices=['stored','deflate'],default='stored',
                   help='stored avoids CPU-heavy compression on a transfer Pod')
    args=p.parse_args()
    root=args.root.resolve()
    if not root.is_dir():p.error('The results root does not exist')
    output=args.output.resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    allowed={'.json','.jsonl','.yaml','.yml','.csv','.log','.xml','.txt','.md','.ptx','.sha256'}
    manifest=[]
    compression = zipfile.ZIP_STORED if args.compression == 'stored' else zipfile.ZIP_DEFLATED
    print(f'PACKAGING root={root} compression={args.compression}', flush=True)
    with zipfile.ZipFile(output,'w',compression) as z:
        for file in sorted(root.rglob('*')):
            if not file.is_file() or file.is_symlink() or file.resolve()==output:continue
            relative=file.relative_to(root)
            if any(s in {'hf','models','cache','triton','cuda'} for s in relative.parts):continue
            if file.suffix.lower() not in allowed:continue
            data=file.read_bytes()
            name=str(relative).replace('\\','/')
            z.writestr(name,data)
            manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            if len(manifest) % 25 == 0:
                print(f'PACKAGED_FILES={len(manifest)} last={name}', flush=True)
        z.writestr('HANDOFF_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    print(f'FILES={len(manifest)} BYTES={output.stat().st_size}')
    print(f'UPLOAD_ARCHIVE={output}')
    return 0


if __name__=='__main__':raise SystemExit(main())
