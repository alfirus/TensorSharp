#!/usr/bin/env python3
"""Seed an app-private OCR download cache from the committed checksum manifest.

This maintenance/offline-distribution tool does not install packages, execute
downloaded binaries, resolve dependencies, or modify user settings. --output is
the downloads directory itself, containing <sha256>/<fileName> entries.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == entry['sha256']


def fetch(entry, output):
    name, digest = entry['fileName'], entry['sha256']
    if Path(name).name != name or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid manifest cache path')
    target = Path(output) / digest / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify(target, entry):
        return {'fileName': name, 'sha256': digest, 'bytes': entry['bytes'], 'cached': True}
    temporary = target.with_name(name + '.' + str(os.getpid()) + '.partial')
    request = urllib.request.Request(entry['url'], headers={'User-Agent': 'TensorSharp-OCR-Cache/1'})
    try:
        size = 0
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as stream:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                size += len(block)
                if size > entry['bytes']:
                    raise ValueError('Download exceeds pinned size: ' + name)
                stream.write(block)
        if not verify(temporary, entry):
            raise ValueError('Download differs from pinned size/SHA256: ' + name)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {'fileName': name, 'sha256': digest, 'bytes': entry['bytes'], 'cached': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(Path(__file__).resolve().parents[2] / 'TensorSharp.Chat/Knowledge/OcrAssets/manifest.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--rid', action='append', help='Repeat to seed multiple runtimes; omit for all')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be 1..16')
    manifest = json.loads(Path(args.manifest).read_text())
    selected = args.rid or [runtime['rid'] for runtime in manifest['runtimes']]
    if set(selected) - {runtime['rid'] for runtime in manifest['runtimes']}:
        parser.error('Unknown runtime identifier')
    assets = list(manifest['languages'])
    for runtime in manifest['runtimes']:
        if runtime['rid'] in selected:
            assets.extend([runtime['bootstrap'], *runtime['packages']])
    assets = list({(entry['sha256'], entry['fileName']): entry for entry in assets}.values())
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda entry: fetch(entry, args.output), assets))
    report = {'manifestId': manifest['id'], 'manifestSha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        'runtimes': selected, 'verifiedFiles': results, 'totalBytes': sum(entry['bytes'] for entry in results)}
    Path(args.output, 'seed-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Verified ' + str(len(results)) + ' files / ' + str(report['totalBytes']) + ' bytes', flush=True)


if __name__ == '__main__':
    main()
