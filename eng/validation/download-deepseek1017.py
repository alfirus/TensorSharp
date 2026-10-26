#!/usr/bin/env python3
"""Download a pinned published GGUF quant and verify every LFS SHA-256.

Requires huggingface_hub. Evidence is written outside the model directory.
Files remain in their publisher layout and are never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--quant', default='UD-IQ2_XXS')
    parser.add_argument('--revision', default='60ad3651641b5c97679c807e064ab1d7a3ca1455')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--companions', action='store_true')
    args = parser.parse_args()
    repo = 'unsloth/DeepSeek-V4.1-Flash-1017-GGUF'
    info = HfApi().model_info(repo, revision=args.revision, files_metadata=True)
    names = [s.rfilename for s in info.siblings
             if s.rfilename.startswith(args.quant + '/') and s.rfilename.endswith('.gguf')]
    if args.companions:
        names += ['dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf', 'mmproj-BF16.gguf']
    selected = [s for s in info.siblings if s.rfilename in names]
    if not selected or len(selected) != len(names):
        raise RuntimeError('Requested quant/companion files are missing')
    records = [{'path': s.rfilename, 'size': s.size, 'sha256': s.lfs.sha256,
                'verified': False} for s in selected]
    evidence = {'repo': repo, 'revision': info.sha, 'quant': args.quant,
                'started_unix': time.time(), 'files': records, 'complete': False}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.evidence.with_suffix('.tmp')
        temporary.write_text(json.dumps(evidence, indent=2) + '\n')
        temporary.replace(args.evidence)

    save()
    print(json.dumps({'revision': info.sha, 'files': len(names),
                      'bytes': sum(s.size for s in selected)}), flush=True)
    snapshot_download(repo, revision=info.sha, allow_patterns=names,
                      local_dir=args.directory, max_workers=args.workers)
    for record in records:
        path = args.directory / record['path']
        if path.stat().st_size != record['size']:
            raise RuntimeError(f'Wrong size: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        record['actual_sha256'] = actual
        record['verified'] = actual == record['sha256']
        save()
        if not record['verified']:
            raise RuntimeError(f'SHA-256 mismatch: {path}')
        print(f'VERIFIED {path.name}', flush=True)
    evidence['complete'] = True
    evidence['finished_unix'] = time.time()
    save()


if __name__ == '__main__':
    main()
