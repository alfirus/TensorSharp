#!/usr/bin/env python3
"""Record source recipes/licenses from SHA256-verified native OCR packages.

Maintainer-only tool. Requires bsdtar with zstd support for .conda metadata;
does not extract or execute package payloads. Run seed_cache.py first.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile
from seed_cache import verify


def inspect_package(entry, downloads, tar):
    archive = Path(downloads) / entry['sha256'] / entry['fileName']
    if not verify(archive, entry):
        raise ValueError('Missing or invalid package: ' + entry['fileName'])
    with tempfile.TemporaryDirectory(prefix='ts-ocr-metadata-') as temporary:
        metadata = archive
        if archive.suffix == '.conda':
            with zipfile.ZipFile(archive) as package:
                name = next(name for name in package.namelist() if name.startswith('info-'))
                metadata = Path(temporary) / 'info.tar.zst'
                metadata.write_bytes(package.read(name))
        names = subprocess.check_output([tar, '-tf', str(metadata)], text=True).splitlines()

        def read(name):
            return subprocess.check_output([tar, '-xOf', str(metadata), name])

        about = json.loads(read('info/about.json')) if 'info/about.json' in names else {}
        recipe_name = next((name for name in ['info/recipe/rendered_recipe.yaml', 'info/recipe/meta.yaml', 'info/recipe/recipe.yaml'] if name in names), None)
        recipe = read(recipe_name).decode('utf-8', 'replace') if recipe_name else None
        licenses = []
        for name in names:
            if name.startswith('info/licenses/') and not name.endswith('/'):
                data = read(name)
                licenses.append({'path': name, 'sha256': hashlib.sha256(data).hexdigest(), 'text': data.decode('utf-8', 'replace')})
        extra = about.get('extra') or {}
        return {'fileName': entry['fileName'], 'sha256': entry['sha256'], 'license': entry['license'],
            'name': entry['name'], 'version': entry['version'], 'build': entry['build'], 'url': entry['url'],
            'sourceUrl': about.get('dev_url') or about.get('home'), 'feedstockUrl': extra.get('remote_url'),
            'feedstockRevision': extra.get('sha'), 'licenses': licenses, 'sourceRecipePath': recipe_name,
            'sourceRecipe': recipe, 'about': about}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=str(Path(__file__).resolve().parents[2] / 'TensorSharp.Chat/Knowledge/OcrAssets/manifest.json'))
    parser.add_argument('--downloads', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tar', default='tar')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--bootstrap-release-json', help='Saved official GitHub release metadata, for static binary third-party notices')
    parser.add_argument('--bootstrap-archives', help='Directory containing the release .tar.bz2 archives')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    packages = {package['sha256']: package for runtime in manifest['runtimes'] for package in runtime['packages']}
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be 1..16')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda entry: inspect_package(entry, args.downloads, args.tar), packages.values()))
    index = {record['sha256']: record for record in records}
    for runtime in manifest['runtimes']:
        for package in runtime['packages']:
            record = index[package['sha256']]
            for key in ['sourceUrl', 'feedstockUrl', 'feedstockRevision']:
                if record[key]:
                    package[key] = record[key]
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + '\n')
    bootstraps = []
    if args.bootstrap_release_json or args.bootstrap_archives:
        if not args.bootstrap_release_json or not args.bootstrap_archives:
            parser.error('Both bootstrap metadata arguments are required')
        release = json.loads(Path(args.bootstrap_release_json).read_text())
        assets = {entry['name']: entry for entry in release['assets']}
        with tempfile.TemporaryDirectory(prefix='ts-bootstrap-notices-') as temporary:
            for runtime in manifest['runtimes']:
                entry = assets['micromamba-' + runtime['platform'] + '.tar.bz2']
                package = {'name': 'micromamba', 'version': release['tag_name'], 'build': runtime['platform'],
                    'fileName': entry['name'], 'url': entry['browser_download_url'], 'bytes': entry['size'],
                    'sha256': entry['digest'].removeprefix('sha256:'), 'license': 'BSD-3-Clause'}
                source = Path(args.bootstrap_archives) / entry['name']
                if not verify(source, package):
                    raise ValueError('Bootstrap source archive is not verified: ' + entry['name'])
                target = Path(temporary) / package['sha256'] / package['fileName']
                target.parent.mkdir(parents=True)
                target.symlink_to(source.resolve())
                bootstraps.append(inspect_package(package, temporary, args.tar))
    Path(args.output).write_text(json.dumps({'manifestId': manifest['id'], 'packages': records, 'bootstraps': bootstraps}, indent=2) + '\n')
    print('Audited ' + str(len(records)) + ' unique package archives; ' + str(sum(len(record['licenses']) for record in records)) + ' license texts')


if __name__ == '__main__':
    main()
