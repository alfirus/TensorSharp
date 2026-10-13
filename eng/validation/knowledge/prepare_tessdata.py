#!/usr/bin/env python3
"""Explicit validation-only download of pinned official Tesseract language data.

This is never called by product startup or document ingestion. It downloads only
the files pinned beside this script, verifies their byte count and SHA256,
and records the source manifest in the specified artifact directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def prepare(output, data_profile='best'):
    if data_profile not in ('fast', 'best'):
        raise ValueError('Unknown data profile: ' + data_profile)
    manifest_path = Path(__file__).with_name('tessdata-' + data_profile + '.json')
    manifest = json.loads(manifest_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        name = entry['language'] + '.traineddata'
        target = output / name
        data = target.read_bytes() if target.exists() else None
        if data is None or hashlib.sha256(data).hexdigest() != entry['sha256']:
            url = 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_' + data_profile + '/' + manifest['revision'] + '/' + name
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read(entry['bytes'] + 1)
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('Pinned language data verification failed: ' + name)
            temporary = target.with_suffix('.partial')
            temporary.write_bytes(data)
            temporary.replace(target)
        if len(data) != entry['bytes']:
            raise ValueError('Pinned language data size differs: ' + name)
        print(name + ': verified ' + entry['sha256'], flush=True)
    (output / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-profile', choices=('fast', 'best'), default='best')
    args = parser.parse_args()
    prepare(args.output, args.data_profile)
