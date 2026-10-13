#!/usr/bin/env python3
"""Measure a real TensorAgent knowledge answer, using an isolated validation host.

Does not mutate knowledge collections. Saves a conversation in the supplied host;
use TensorAgentHost with an ignored artifacts/ root and a copy of any local library.
Connection credentials are read locally and never included in the result report.
"""
import argparse
import json
import pathlib
import re
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection', required=True)
    parser.add_argument('--collection', required=True, help='Exact collection name or ID')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', required=True, help='Ignored artifacts/ or docs/validation/ directory')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--think', action='store_true')
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    connection = json.loads(pathlib.Path(args.connection).read_text())
    base = connection['baseUrl'].rstrip('/')
    headers = {'Content-Type': 'application/json', 'Cookie': connection['cookie']}

    def request(path, body=None, method=None):
        return urllib.request.urlopen(urllib.request.Request(base + path,
            data=None if body is None else json.dumps(body).encode(), headers=headers,
            method=method), timeout=args.timeout)

    with request('/api/knowledge/collections') as response:
        payload = json.load(response)
        collections = payload['collections']
    matched = [c for c in collections if args.collection in (c['id'], c['name'])]
    if len(matched) != 1:
        raise ValueError('Select exactly one existing collection by ID or unique name')
    collection = matched[0]['id']
    with request('/api/sessions?conversation=new', {}, 'POST') as response:
        session = json.load(response)
    body = {'sessionId': session['sessionId'], 'knowledgeCollections': [collection],
        'skills': [], 'skills_discovery': False, 'think': args.think,
        'messages': [{'role': 'user', 'content': args.prompt}], 'maxTokens': args.max_tokens}
    frames, timings, answer, sources = [], [], '', []
    started = time.perf_counter()
    expired = threading.Event()
    turn_id = None

    def expire():
        expired.set()
        try:
            if turn_id:
                with request('/api/agent/turns/' + turn_id + '/stop', {}, 'POST') as response:
                    response.read()
        except Exception:
            pass

    deadline = threading.Timer(args.timeout, expire)
    deadline.daemon = True
    deadline.start()
    error = None
    try:
        with request('/api/chat', body, 'POST') as response, (output / 'frames.ndjson').open('w') as log:
            turn_id = response.headers.get('X-TensorAgent-Turn')
            for raw in response:
                if expired.is_set():
                    break
                if not raw.startswith(b'data: '):
                    continue
                frame = json.loads(raw[6:])
                elapsed = time.perf_counter() - started
                frames.append(frame)
                timings.append(elapsed)
                log.write(json.dumps({'seconds': elapsed, 'frame': frame}, ensure_ascii=False) + '\n')
                log.flush()
                if 'token' in frame:
                    answer += frame['token'] or ''
                if 'replace' in frame:
                    answer = frame['replace'] or ''
                if 'knowledgeSources' in frame:
                    sources = frame['knowledgeSources']
                if 'skill' in frame or 'skill_step' in frame or frame.get('done') or frame.get('error'):
                    print(json.dumps({'seconds': round(elapsed, 2), 'frame': frame}, ensure_ascii=False), flush=True)
    except Exception as exception:
        error = str(exception)
    finally:
        deadline.cancel()
    source_urls = set(re.findall(r'/api/knowledge/collections/[a-f0-9]{32}/documents/[a-f0-9]{32}/source\?version=\d+', answer))
    source_checks = []
    for source in sources:
        url = source['sourceUrl']
        check = {'url': url, 'citedInAnswer': url in source_urls
            or bool(source.get('reference') and source['reference'] in answer)}
        try:
            with request(url) as response:
                check['status'] = response.status
        except Exception as exception:
            check['error'] = str(exception)
        source_checks.append(check)
    tool_frames = [frame for frame in frames if 'skill_step' in frame]
    report = {'model': pathlib.Path(connection.get('model') or '').name,
        'backend': connection.get('backend'), 'contextLength': connection.get('contextLength'),
        'think': args.think, 'seconds': time.perf_counter() - started,
        'timeout': expired.is_set(), 'error': error, 'prompt': args.prompt,
        'done': any(f.get('done') for f in frames), 'answer': answer,
        'terminal': next((f for f in reversed(frames) if f.get('done')), None),
        'toolSteps': tool_frames, 'sourceChecks': source_checks,
        'firstAnswerSeconds': next((t for f, t in zip(frames, timings) if f.get('token')), None),
        'limitations': ['One local run; not a statistical benchmark or cross-device validation.',
            'Source-use checks verify citations, not the semantic accuracy of every claim.']}
    report['passed'] = bool(report['done'] and answer and source_checks and not error and not expired.is_set()
        and not any(f.get('error') or f.get('aborted') or f.get('truncated') for f in frames)
        and all(s.get('status') == 200 and s['citedInAnswer'] for s in source_checks))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (output / 'answer.txt').write_text(answer)
    print(json.dumps({k: v for k, v in report.items() if k not in ('answer', 'toolSteps', 'sourceChecks')}, ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
