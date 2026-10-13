#!/usr/bin/env python3
"""Maintainer-only refresh of pinned OCR dependency closures, never product startup.

Solves three platforms using conda-forge only, writes exact URL/size/SHA256 locks,
and retains solver output for review. Run seed_cache.py and audit_packages.py
afterward, then validate on real operating systems before shipping the refresh.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micromamba', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--id', default='ocr-2026-10-11.1')
    parser.add_argument('--bootstrap-release', default='2.9.0-0')
    parser.add_argument('--poppler', default='26.09.0')
    parser.add_argument('--tesseract', default='5.5.3')
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[2] / 'TensorSharp.Chat/Knowledge/OcrAssets/manifest.json'))
    args = parser.parse_args()
    evidence = Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request('https://api.github.com/repos/mamba-org/micromamba-releases/releases/tags/' + args.bootstrap_release,
        headers={'User-Agent': 'TensorSharp-OCR-Maintainer/1'})
    release = json.load(urllib.request.urlopen(request, timeout=60))
    (evidence / 'micromamba-release.json').write_text(json.dumps(release, indent=2) + '\n')
    assets = {entry['name']: entry for entry in release['assets']}
    runtimes = []
    for rid, platform in [('win-x64', 'win-64'), ('linux-x64', 'linux-64'), ('linux-arm64', 'linux-aarch64')]:
        command = [str(Path(args.micromamba).resolve()), 'create', '--no-rc', '--root-prefix', str(evidence / ('solve-' + platform)),
            '--prefix', str(evidence / ('unused-prefix-' + platform)), '--platform', platform,
            '--override-channels', '--channel', 'conda-forge', '--strict-channel-priority', '--dry-run', '--json',
            'poppler=' + args.poppler, 'tesseract=' + args.tesseract]
        # Build-time cross-platform solve only. Remove ambient Conda/Mamba
        # overrides, then declare the oldest glibc target this lock requests.
        environment = {key: value for key, value in os.environ.items() if not key.startswith(('CONDA_', 'MAMBA_'))}
        environment['CONDA_OVERRIDE_GLIBC'] = '2.17'
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        (evidence / ('solve-' + platform + '.json')).write_text(result.stdout)
        (evidence / ('solve-' + platform + '.err')).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError('Dependency solve failed: ' + platform)
        solved = json.loads(result.stdout)
        if not solved.get('success'):
            raise RuntimeError('Dependency solve has no solution: ' + platform)
        asset = assets['micromamba-' + platform + ('.exe' if rid == 'win-x64' else '')]
        digest = asset.get('digest', '')
        if not digest.startswith('sha256:') or len(digest) != 71:
            raise ValueError('Release asset lacks SHA256: ' + asset['name'])
        bootstrap = {'url': asset['browser_download_url'], 'sha256': digest[7:], 'bytes': asset['size'],
            'fileName': 'micromamba.exe' if rid == 'win-x64' else 'micromamba', 'version': args.bootstrap_release,
            'license': 'BSD-3-Clause', 'sourceUrl': 'https://github.com/mamba-org/mamba/tree/' + args.bootstrap_release.rsplit('-', 1)[0]}
        packages = [{key: package[key] for key in ['url', 'sha256', 'name', 'version', 'license', 'build']} |
            {'bytes': package['size'], 'fileName': package['fn'], 'dependencies': package['depends'],
             'packageMetadataUrl': 'https://api.anaconda.org/package/conda-forge/' + package['name']}
            for package in solved['actions']['LINK']]
        paths = {'rendererPath': 'Library/bin/pdftoppm.exe', 'ocrPath': 'Library/bin/tesseract.exe',
            'environment': {'PATH': 'Library/bin'}} if rid == 'win-x64' else {
            'rendererPath': 'bin/pdftoppm', 'ocrPath': 'bin/tesseract',
            'environment': {'FONTCONFIG_FILE': 'etc/fonts/fonts.conf', 'FONTCONFIG_PATH': 'etc/fonts'}}
        runtimes.append({'rid': rid, 'platform': platform, 'bootstrap': bootstrap, 'packages': packages, **paths})
    data = json.loads((Path(__file__).resolve().parents[1] / 'validation/knowledge/tessdata-best.json').read_text())
    languages = [entry | {'url': data['repository'].replace('https://github.com/', 'https://raw.githubusercontent.com/') + '/' +
        data['revision'] + '/' + entry['language'] + '.traineddata', 'fileName': entry['language'] + '.traineddata',
        'license': 'Apache-2.0', 'sourceUrl': data['repository'] + '/tree/' + data['revision']} for entry in data['files']]
    Path(args.output).write_text(json.dumps({'version': 1, 'id': args.id, 'runtimes': runtimes, 'languages': languages}, indent=2) + '\n')


if __name__ == '__main__':
    main()
