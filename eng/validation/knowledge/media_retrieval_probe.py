#!/usr/bin/env python3
"""Validate image import, reindex, retrieval and originals on an isolated TensorAgentHost.

Start the host with an ignored artifacts/ root and a COPY of the library to diagnose.
This probe mutates that copy. It refuses connections outside an ignored evidence
directory. It never logs connection credentials or opens the user's library.
Generated chat answers can be checked separately with answer_quality_probe.py.
"""
import argparse
import hashlib
import json
import mimetypes
import pathlib
import time
import urllib.request
import urllib.parse
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--expected-text', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    repository = pathlib.Path(__file__).resolve().parents[3]
    allowed = [repository / 'artifacts', repository / 'docs' / 'validation']
    connection_path = pathlib.Path(args.connection).resolve()
    output = pathlib.Path(args.output).resolve()
    for path in (connection_path, output):
        if not any(path.is_relative_to(root) for root in allowed):
            raise ValueError('Connection and output must be in artifacts/ or docs/validation/')
    if not (connection_path.parent / 'data' / 'knowledge').is_dir():
        raise ValueError('Use an isolated TensorAgentHost with its data/knowledge beside connection.json')
    connection = json.loads(connection_path.read_text())
    if pathlib.Path(connection.get('dataRoot') or '').resolve() != connection_path.parent / 'data':
        raise ValueError('Connection must identify this isolated host data root; rebuild TensorAgentHost if missing')
    endpoint = urllib.parse.urlsplit(connection['baseUrl'])
    if endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', '::1', 'localhost'):
        raise ValueError('Use the isolated loopback TensorAgentHost')
    output.mkdir(parents=True, exist_ok=True)
    source = pathlib.Path(args.image)
    data = source.read_bytes()
    report = {
        'model': pathlib.Path(connection.get('model') or '').name,
        'backend': connection.get('backend'),
        'image': source.name,
        'imageSha256': hashlib.sha256(data).hexdigest(),
        'query': args.query,
        'checks': [],
        'limitations': [
            'One local image on one loaded model/backend; no audio/video or cross-device validation.',
            'Original hashes and retrieval identity do not establish caption accuracy.',
            'No native image embeddings or generated-answer quality measured by this probe.',
        ],
    }

    def request(path, body=None, content_type='application/json'):
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
        with urllib.request.urlopen(urllib.request.Request(
                connection['baseUrl'].rstrip('/') + path, data=body,
                headers={'Cookie': connection['cookie'], 'Content-Type': content_type}),
                timeout=240) as response:
            return response.read()

    def js(path, body=None):
        return json.loads(request(path, body))

    def check(name, passed, **details):
        report['checks'].append({'name': name, 'passed': bool(passed), **details})
        if not passed:
            raise AssertionError(name)

    def upload(collection):
        boundary = 'knowledgeprobe' + uuid.uuid4().hex
        name = source.name.replace('"', '%22').replace('\r', '').replace('\n', '')
        content_type = mimetypes.guess_type(source.name)[0] or 'application/octet-stream'
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
                f'filename="{name}"\r\nContent-Type: {content_type}\r\n\r\n').encode()
        body += data + f'\r\n--{boundary}--\r\n'.encode()
        return json.loads(request(f'/api/knowledge/collections/{collection}/documents',
                                 body, f'multipart/form-data; boundary={boundary}'))

    started = time.perf_counter()
    try:
        collections = js('/api/knowledge/collections')['collections']
        matches = [c for c in collections if args.collection in (c['name'], c['id'])]
        if len(matches) != 1:
            raise ValueError('Select one existing collection by ID or unique name')
        collection = matches[0]['id']
        imported = upload(collection)
        check('image_import', not imported.get('errors') and len(imported['documents']) == 1)
        document = imported['documents'][0]
        route = f'/api/knowledge/collections/{collection}/documents/{document["id"]}'
        repeated = upload(collection)
        check('identical_import_is_idempotent', not repeated.get('errors')
              and repeated['documents'][0]['id'] == document['id']
              and repeated['documents'][0]['version'] == document['version'])
        reindexed = js(route + '/reindex', {})['document']
        check('reindex_derived_content', reindexed['version'] == document['version'] + 1
              and reindexed['status'] != 'MetadataOnly' and reindexed['chunkCount'] > 0,
              status=reindexed['status'], chunks=reindexed['chunkCount'],
              extractor=reindexed.get('extractorId'), embeddings=reindexed.get('embeddingModelId'))
        preview = js(route + f'/preview?version={reindexed["version"]}')
        check('visible_text_is_indexed', args.expected_text.casefold() in preview['text'].casefold())
        check('original_bytes_preserved', request(route + f'/source?version={reindexed["version"]}') == data)
        search = js('/api/knowledge/search', {'query': args.query,
                    'collectionIds': [collection], 'limit': 8})['results']
        check('image_ranks_first_for_image_query', bool(search) and search[0]['documentId'] == document['id'],
              results=[{'fileName': hit['fileName'], 'modality': hit.get('modality')} for hit in search])
        filename_search = js('/api/knowledge/search', {'query': source.stem,
                             'collectionIds': [collection], 'limit': 8})['results']
        check('filename_search_survives_reindex', any(hit['documentId'] == document['id'] for hit in filename_search))
        empty = js('/api/knowledge/search', {'query': args.query, 'collectionIds': []})['results']
        check('empty_scope_returns_no_sources', not empty)
        report['passed'] = True
    except Exception as error:
        report['passed'] = False
        report['error'] = str(error)
    report['seconds'] = time.perf_counter() - started
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
