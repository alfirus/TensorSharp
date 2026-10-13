#!/usr/bin/env python3
"""Exercise knowledge on real OpenAI/Ollama HTTP adapters without starting a host.

All source data is original synthetic fixture content. Creates isolated temporary
collections and deletes only those collections. Raw replies/results go to ignored
artifacts. No model downloads, host/model changes, or answer-only agent passes.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import pathlib
import random
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROUTES = {'chat': '/v1/chat/completions', 'responses': '/v1/responses',
          'ollama': '/api/chat/ollama', 'generate': '/api/generate'}
SOURCE = re.compile(r'/api/knowledge/collections/[a-f0-9]{32}/documents/[^\s)\]\"<>`]+/source\?version=\d+')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def exact_number(text, value):
    return re.search(r'(?<!\d)' + re.escape(str(value)) + r'(?!\d)', text.replace(',', '')) is not None


def chinese_phrase_checks(text, phrases):
    compact=''.join(text.split())
    normalized=all(phrase in compact for phrase in phrases)
    return {'expected':phrases,'comparison':'whitespace-insensitive',
        'rawExact':all(phrase in text for phrase in phrases),'whitespaceNormalized':normalized,
        'found':[phrase for phrase in phrases if phrase in compact],'passed':normalized}


def protocol_body(protocol, model, prompt, stream, scope, max_tokens=384, tools=False):
    body = {'model': model, 'stream': stream, 'knowledgeCollections': scope,
            'skills': [], 'skills_discovery': False, 'think': False, 'temperature': 0}
    if not tools:
        body['tool_choice'] = 'none'
    if protocol == 'responses':
        body.update(input=prompt, max_output_tokens=max_tokens, store=False)
    elif protocol == 'generate':
        body.update(prompt=prompt, options={'temperature': 0, 'num_predict': max_tokens})
    else:
        body['messages'] = [{'role': 'user', 'content': prompt}]
        if protocol == 'ollama':
            body['options'] = {'temperature': 0, 'num_predict': max_tokens}
        else:
            body['max_tokens'] = max_tokens
    return body


def response_text(value):
    return ''.join(part.get('text', '') for item in value.get('output', [])
                   if item.get('type') == 'message' for part in item.get('content', [])
                   if part.get('type') == 'output_text')


def parse_reply(protocol, raw, streaming):
    text = raw.decode('utf-8')
    if not streaming:
        payload = json.loads(text)
        if protocol == 'chat':
            answer = payload['choices'][0]['message'].get('content') or ''
            complete = payload['choices'][0].get('finish_reason') not in ('length', None)
        elif protocol == 'responses':
            answer, complete = response_text(payload), payload.get('status') == 'completed'
        elif protocol == 'ollama':
            answer, complete = payload.get('message', {}).get('content', ''), payload.get('done') is True and payload.get('done_reason') != 'length'
        else:
            answer, complete = payload.get('response', ''), payload.get('done') is True and payload.get('done_reason') != 'length'
        return answer, payload, [payload], complete
    events = []
    for line in text.splitlines():
        if protocol in ('chat', 'responses'):
            if not line.startswith('data:'):
                continue
            line = line[5:].strip()
            if line == '[DONE]':
                continue
        if line.strip():
            events.append(json.loads(line))
    answer, terminal, complete = '', {}, False
    for event in events:
        if protocol == 'chat':
            for choice in event.get('choices', []):
                answer += choice.get('delta', {}).get('content') or ''
                if choice.get('finish_reason'):
                    complete = choice['finish_reason'] != 'length'
            if 'knowledge_sources' in event or any(c.get('finish_reason') for c in event.get('choices', [])):
                terminal.update(event)
        elif protocol == 'responses':
            if event.get('type') == 'response.output_text.delta':
                answer += event.get('delta', '')
            if event.get('type') == 'response.completed':
                terminal = event.get('response', {})
                complete = terminal.get('status') == 'completed'
                if response_text(terminal) != answer:
                    raise AssertionError('Responses terminal text differs from streamed deltas')
        else:
            answer += event.get('response', '') if protocol == 'generate' else event.get('message', {}).get('content', '')
            if event.get('done'):
                terminal, complete = event, event.get('done_reason') != 'length'
    return answer, terminal, events, complete


class Probe:
    def __init__(self, args):
        self.args = args
        connection = json.loads(pathlib.Path(args.connection).read_text()) if args.connection else {}
        self.base = (args.base_url or connection.get('baseUrl', '')).rstrip('/')
        self.model = args.model or pathlib.Path(connection.get('model') or '').name
        if not self.base or not self.model:
            raise ValueError('Provide --connection or both --base-url and --model')
        self.headers = {'Content-Type': 'application/json'}
        if connection.get('cookie'):
            self.headers['Cookie'] = connection['cookie']
        token = os.environ.get('TS_PROBE_BEARER_TOKEN')
        if token:
            self.headers['Authorization'] = 'Bearer ' + token
        self.out = pathlib.Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.collections, self.originals = [], {}
        self.report = {'model': self.model, 'backend': connection.get('backend'), 'baseUrl': self.base,
            'samplesPerSelectedCase': args.samples, 'cases': [], 'pdf': {'status': 'not_run'},
            'agent': {'status': 'not_run'}, 'passed': False,
            'limitations': ['Sequential local requests; no concurrent-client or cross-system comparison.',
                'Knowledge-scoped answer deltas may be buffered by the adapter until citation normalization finishes.',
                'Random source code controls check answer grounding, not general semantic quality.',
                'PDFs are clean synthetic scans; Chinese coverage is separately reported.',
                'Agent pass requires authoritative successful tool invocations and exact code output, not an answer alone.']}

    def save(self):
        (self.out/'report.json').write_text(json.dumps(self.report, indent=2, ensure_ascii=False)+'\n')

    def request(self, path, body=None, method=None, content_type='application/json'):
        if body is not None and not isinstance(body, bytes):
            body = json.dumps(body).encode()
        headers = dict(self.headers, **{'Content-Type': content_type})
        request = urllib.request.Request(self.base+path, data=body, headers=headers, method=method)
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                chunks, first = [], None
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    if first is None:
                        first = time.perf_counter()-started
                    chunks.append(chunk)
                return response.status, b''.join(chunks), {'seconds': time.perf_counter()-started,
                    'firstByteSeconds': first, 'contentType': response.headers.get('Content-Type')}
        except urllib.error.HTTPError as error:
            return error.code, error.read(), {'seconds': time.perf_counter()-started,
                'contentType': error.headers.get('Content-Type')}

    def js(self, path, body=None, method=None):
        status, raw, _ = self.request(path, body, method)
        if status >= 400:
            raise AssertionError(f'{path}: HTTP{status} {raw[:1000]!r}')
        return json.loads(raw)

    def collection(self, label):
        value = self.js('/api/knowledge/collections', {'name':label+' '+uuid.uuid4().hex[:8]})['collection']['id']
        self.collections.append(value)
        return value

    def upload(self, collection, name, data, content_type='application/octet-stream'):
        boundary = 'knowledge'+uuid.uuid4().hex
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
                f'Content-Type: {content_type}\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
        status, raw, timing = self.request('/api/knowledge/collections/'+collection+'/documents', body,
            content_type='multipart/form-data; boundary='+boundary)
        if status >= 400:
            raise AssertionError(f'Upload {name}: HTTP{status} {raw[:1000]!r}')
        result = json.loads(raw)
        if result.get('errors') or len(result.get('documents', [])) != 1:
            raise AssertionError(result)
        doc = result['documents'][0]
        self.originals[(collection, doc['id'], doc['version'])] = digest(data)
        return doc, timing

    def sources(self, sources, scope):
        checks = []
        for source in sources:
            key = (source['collectionId'], source['documentId'], source['version'])
            assert key[0] in scope, 'Source metadata escaped selected collection'
            assert key in self.originals, 'Source identity/version was not uploaded by this probe'
            expected_url = f'/api/knowledge/collections/{key[0]}/documents/{key[1]}/source?version={key[2]}'
            assert source['sourceUrl'] == expected_url, 'Source URL differs from authoritative identity'
            status, raw, _ = self.request(expected_url)
            assert status == 200 and digest(raw) == self.originals[key], 'Citation does not fetch the exact original'
            checks.append(expected_url)
        return checks

    def inference(self, protocol, stream, label, prompt, scope, expected=None, forbidden=None, agent=False):
        body = protocol_body(protocol, self.model, prompt, stream, scope, 1024 if agent else 384, tools=agent)
        stem = f'{protocol}-'+('stream' if stream else 'json')+'-'+label
        (self.out/(stem+'-request.json')).write_text(json.dumps(body,indent=2))
        status, raw, timing = self.request(ROUTES[protocol], body)
        (self.out/(stem+('.sse' if protocol in ('chat','responses') and stream else '.ndjson' if stream else '.json'))).write_bytes(raw)
        result = {'protocol':protocol, 'stream':stream, 'label':label, 'status':status, **timing, 'passed':False}
        self.report['cases'].append(result)
        try:
            assert status == 200, f'HTTP{status}: {raw[:1000]!r}'
            answer, terminal, events, complete = parse_reply(protocol, raw, stream)
            result.update(answer=answer, terminal=terminal, complete=complete)
            assert complete, 'Response has no successful terminal event or was truncated'
            assert not any(e.get('error') for e in events), 'Protocol reported error in stream'
            sources = terminal.get('knowledge_sources', [])
            result['sourceChecks'] = self.sources(sources, scope)
            if expected:
                assert expected in answer, f'Expected grounded value {expected!r}'
                assert sources, 'Selected evidence missing authoritative knowledge_sources'
                written = SOURCE.findall(answer)
                assert written, 'No source URL in grounded answer'
                assert all(url in result['sourceChecks'] for url in written), 'Model-written source URL is not retrieved evidence'
            if forbidden:
                assert forbidden not in answer, 'Unselected source value leaked into answer'
                if not scope:
                    assert not sources, 'Unselected request unexpectedly has sources'
            result['passed'] = True
        except Exception as error:
            result['error'] = f'{type(error).__name__}: {error}'
        self.save()
        if not agent:
            print(json.dumps({key:result.get(key) for key in ('protocol','stream','label','passed','seconds','error')},ensure_ascii=False),flush=True)
        return result

    def invalid_scope(self, protocol, stream):
        for scope in ('all', [7], ['f'*32]):
            body = protocol_body(protocol, self.model, 'Read the selected sources.', stream, scope)
            status, raw, timing = self.request(ROUTES[protocol], body)
            self.report['cases'].append({'protocol':protocol,'stream':stream,'label':'invalid-scope',
                'scope':scope,'status':status,'passed':status == 400,'body':raw.decode(errors='replace')[:1500],**timing})
        self.save()

    def pdf(self):
        if self.args.skip_pdf:
            self.report['pdf']['reason'] = '--skip-pdf requested'
            return
        directory = pathlib.Path(self.args.fixtures)
        manifest = json.loads((directory/'manifest.json').read_text())
        cid = self.collection('Protocol PDF probe')
        report = {'status':'running','pages':[],'uploads':[],'chinese':[],'collectionId':cid}
        self.report['pdf'] = report
        try:
            for fixture in manifest['files']:
                data = (directory/fixture['fileName']).read_bytes()
                assert digest(data) == fixture['sha256'], 'PDF fixture hash mismatch'
                doc, timing = self.upload(cid, fixture['fileName'], data, 'application/pdf')
                report['uploads'].append({'document':doc,**timing})
                for page in fixture['pages']:
                    started = time.perf_counter()
                    hits = self.js('/api/knowledge/search',{'query':page['query'],'collectionIds':[cid],'limit':50,'maxCharacters':32000})['results']
                    matching = [h for h in hits if h['documentId']==doc['id'] and (h.get('location') or {}).get('page')==page['page']]
                    text = '\n'.join(h['text'] for h in matching)
                    code = re.sub(r'\W','',page['expectedCode']) in re.sub(r'\W','',text)
                    total = not page.get('expectedTotal') or exact_number(text,page['expectedTotal'])
                    entry={'file':fixture['fileName'],'page':page['page'],'selectableText':page['selectableText'],
                        'passed':bool(matching) and code and total,'codeMatched':code,'totalMatched':total,
                        'seconds':time.perf_counter()-started,'results':matching}
                    report['pages'].append(entry)
                    if matching:
                        self.sources(matching,[cid])
                        read = self.js('/api/knowledge/read',{'reference':matching[0]['reference'],'collectionIds':[cid],'maxCharacters':32000})
                        assert read['results'], 'PDF source reference unreadable'
                    if page.get('chineseExpected'):
                        report['chinese'].append({'file':fixture['fileName'],'page':page['page'],
                            **chinese_phrase_checks(text,page['chineseExpected'])})
                    self.save()
            report['status']='passed' if all(p['passed'] for p in report['pages']) else 'failed'
            if self.args.require_chinese and not all(p['passed'] for p in report['chinese']):
                report['status']='failed'
            first=manifest['files'][0]['pages'][0]
            for protocol in self.args.protocols:
                for stream in self.args.modes:
                    self.inference(protocol,stream,'pdf-grounding',
                        'What is the Cedar observation code in scanned-four-pages.pdf? Copy the code exactly and cite its source URL and page number.',
                        [cid],expected=first['expectedCode'])
        except Exception as error:
            report.update(status='failed',error=f'{type(error).__name__}: {error}')
        self.save()

    def agent(self):
        if not self.args.agent:
            self.report['agent']['reason']='--agent not selected; code execution not exercised'
            return
        cid=self.collection('Protocol complete CSV probe')
        rng=random.Random(481731)
        amounts=[rng.randint(-9371,24583) for _ in range(500)]
        buffer=io.StringIO(newline='')
        writer=csv.writer(buffer,lineterminator='\n')
        writer.writerow(['row_id','amount_cents'])
        writer.writerows(enumerate(amounts,1))
        data=buffer.getvalue().encode()
        (self.out/'protocol-ledger.csv').write_bytes(data)
        doc,timing=self.upload(cid,'protocol-ledger.csv',data,'text/csv')
        report={'status':'running','document':doc,'upload':timing,'rows':len(amounts),'expectedSum':sum(amounts),
            'promptMode':'provided-python-recipe' if self.args.agent_recipe else 'model-authored-python',
            'recipe':'literal-multiline-v1' if self.args.agent_recipe else None,'cases':[]}
        self.report['agent']=report
        for protocol in self.args.protocols:
            if protocol=='generate':
                report['cases'].append({'protocol':protocol,'status':'not_supported','reason':'Raw generate has no agent tool loop'})
                continue
            for stream in self.args.modes:
                prompt=(f"Compute an exact aggregate over COMPLETE protocol-ledger.csv. First call knowledge_open with collection_id='{cid}', "
                    f"document_id='{doc['id']}', version={doc['version']}. Use the returned private workspace path in shell running Python csv.DictReader. "
                    "Read EVERY row, count rows and sum integer amount_cents. Have Python print ROWS=<count> and TOTAL_CENTS=<sum>. "
                    "Do not sum excerpts or calculate mentally. Report those verified values and cite the original source URL.")
                if self.args.agent_recipe:
                    prompt += (" Use this exact multiline Python recipe, replacing PATH_FROM_KNOWLEDGE_OPEN "
                        "with the returned path. Preserve the newlines and indentation; use a script or shell heredoc:\n"
                        "```python\nimport csv\nwith open('PATH_FROM_KNOWLEDGE_OPEN', newline='') as f:\n"
                        "    rows = list(csv.DictReader(f))\nprint('ROWS=' + str(len(rows)))\n"
                        "print('TOTAL_CENTS=' + str(sum(int(row['amount_cents']) for row in rows)))\n```\n")
                result=self.inference(protocol,stream,'whole-csv',prompt,[cid],agent=True)
                steps=result.get('terminal',{}).get('knowledge_tool_steps',[])
                opened=any(s.get('tool')=='knowledge_open' and s.get('ok') is True for s in steps)
                shells=[s for s in steps if s.get('tool')=='shell' and s.get('ok') is True]
                code='\n'.join(s.get('output') or '' for s in shells)
                answer=result.get('answer','')
                output_exact=exact_number(code,len(amounts)) and exact_number(code,sum(amounts))
                answer_exact=exact_number(answer,len(amounts)) and exact_number(answer,sum(amounts))
                citations=SOURCE.findall(answer)
                citations_valid=bool(citations) and all(url in result.get('sourceChecks',[]) for url in citations)
                passed=result['passed'] and opened and bool(shells) and output_exact and answer_exact and citations_valid
                result.update(passed=passed,agentChecks={'successfulOpen':opened,'successfulShell':bool(shells),
                    'exactCodeOutput':output_exact,'exactAnswer':answer_exact,'validAnswerCitations':citations_valid,'steps':steps})
                if not passed and not result.get('error'):
                    result['error']='Agent proof failed: ' + ', '.join(key for key,value in result['agentChecks'].items() if value is False)
                report['cases'].append({'protocol':protocol,'stream':stream,'status':'passed' if passed else 'failed'})
                print(json.dumps({'protocol':protocol,'stream':stream,'label':'whole-csv','passed':passed,
                    'seconds':result['seconds'],'checks':result['agentChecks']},ensure_ascii=False),flush=True)
                self.save()
        executed=[c for c in report['cases'] if c['status']!='not_supported']
        report['status']='passed' if executed and all(c['status']=='passed' for c in executed) else 'not_run' if not executed else 'failed'

    def run(self):
        try:
            if self.args.agent_only:
                self.agent()
                self.report['coverage']={'protocols':self.args.protocols,
                    'modes':['stream' if mode else 'json' for mode in self.args.modes],
                    'scope':'not_run','pdf':'not_run','agent':self.report['agent']['status'],'chineseRequired':False}
                self.report['passed']=bool(self.report['cases']) and all(c['passed'] for c in self.report['cases']) and self.report['agent']['status']=='passed'
                self.save()
                return 0 if self.report['passed'] else 1
            cid=self.collection('Protocol scope probe')
            secret='QUASAR-'+uuid.uuid4().hex[:12].upper()
            data=f'The Quasar rendezvous code is {secret}. It belongs only to this uploaded record.\n'.encode()
            (self.out/'protocol-secret.txt').write_bytes(data)
            doc,timing=self.upload(cid,'protocol-secret.txt',data,'text/plain')
            self.report['source']={'document':doc,'sha256':digest(data),'upload':timing}
            other=self.collection('Protocol unrelated scope')
            self.upload(other,'unrelated-note.txt',b'The unrelated Boreal workshop has no rendezvous information.\n','text/plain')
            prompt='What is the Quasar rendezvous code? Use selected knowledge, copy its value exactly, and cite the source URL. If unavailable, say unknown.'
            for protocol in self.args.protocols:
                for stream in self.args.modes:
                    self.invalid_scope(protocol,stream)
                    self.inference(protocol,stream,'unselected',prompt,[],forbidden=secret)
                    self.inference(protocol,stream,'other-collection',prompt,[other],forbidden=secret)
                    for sample in range(self.args.samples):
                        self.inference(protocol,stream,f'selected-{sample+1}',prompt,[cid],expected=secret)
            self.pdf()
            self.agent()
            timings={}
            for protocol in self.args.protocols:
                for stream in self.args.modes:
                    samples=[c['seconds'] for c in self.report['cases'] if c['protocol']==protocol and c['stream']==stream and c['label'].startswith('selected-')]
                    if samples:
                        timings[protocol+('-stream' if stream else '-json')]={'count':len(samples),'p50Seconds':statistics.median(samples),
                            'p95Seconds':sorted(samples)[math.ceil(len(samples)*.95)-1],'allSeconds':samples,
                            'interpretation':'End-to-end sequential request latency; includes model inference and retrieval. Small samples are diagnostic only.'}
            self.report['latency']=timings
            self.report['coverage']={'protocols':self.args.protocols,'modes':['stream' if mode else 'json' for mode in self.args.modes],
                'pdf':self.report['pdf']['status'],'agent':self.report['agent']['status'],
                'chineseRequired':self.args.require_chinese}
            self.report['passed']=all(c['passed'] for c in self.report['cases']) and self.report['pdf']['status']!='failed' and self.report['agent']['status']!='failed'
            self.save()
            return 0 if self.report['passed'] else 1
        except Exception as error:
            self.report.update(passed=False,error=f'{type(error).__name__}: {error}')
            self.save()
            raise
        finally:
            for cid in reversed(self.collections):
                try:
                    self.js('/api/knowledge/collections/'+cid,method='DELETE')
                except Exception as error:
                    self.report.setdefault('cleanupErrors',[]).append(str(error))
            self.save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection')
    parser.add_argument('--base-url')
    parser.add_argument('--model')
    parser.add_argument('--output',required=True)
    parser.add_argument('--protocols',nargs='+',choices=list(ROUTES),default=list(ROUTES))
    parser.add_argument('--modes',nargs='+',choices=['stream','json'],default=['stream','json'])
    parser.add_argument('--samples',type=int,default=1)
    parser.add_argument('--timeout',type=int,default=900)
    parser.add_argument('--fixtures',default='InferenceWeb.Tests/Fixtures/Knowledge')
    parser.add_argument('--skip-pdf',action='store_true')
    parser.add_argument('--require-chinese',action='store_true')
    parser.add_argument('--agent',action='store_true')
    parser.add_argument('--agent-recipe',action='store_true',help='Provide safe Python CSV recipe; reports guided integration separately from model-authored code')
    parser.add_argument('--agent-only',action='store_true',help='Run only complete-CSV agent cases; scope and PDF coverage remain not_run')
    args=parser.parse_args()
    if args.samples<1 or args.samples>100:
        parser.error('--samples must be 1..100')
    args.modes=[m=='stream' for m in args.modes]
    if args.agent_only:
        args.agent=True
    return Probe(args).run()


if __name__=='__main__':
    raise SystemExit(main())
