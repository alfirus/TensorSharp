#!/usr/bin/env python3
"""Exercise the actual TensorAgent HTTP library with committed synthetic fixtures.
Run TensorAgentHost first. Evidence goes only into an ignored output directory.
This is a small correctness screen, not representative OCR/ASR/video accuracy.
"""
import argparse, hashlib, json, pathlib, re, time, urllib.request, urllib.error, uuid
p=argparse.ArgumentParser();p.add_argument('--connection',required=True);p.add_argument('--output',required=True);p.add_argument('--fixtures',default='InferenceWeb.Tests/Fixtures/JevAttachments');p.add_argument('--skip-media',action='store_true');a=p.parse_args()
connection=json.loads(pathlib.Path(a.connection).read_text());out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
report={'model':connection['model'],'backend':connection['backend'],'cases':[], 'limitations':['Small original synthetic fixtures; not representative OCR, ASR or video-action quality.','Video fixture contains repeated still frames.','All timings include HTTP and model work; no concurrent workload.']}
def request(path,body=None,method=None,content_type='application/json'):
    if body is not None and not isinstance(body,bytes):body=json.dumps(body).encode()
    req=urllib.request.Request(connection['baseUrl']+path,data=body,method=method,headers={'Cookie':connection['cookie'],'Content-Type':content_type})
    with urllib.request.urlopen(req,timeout=600) as response:return response.read()
def js(path,body=None,method=None):return json.loads(request(path,body,method))
def upload(cid,name,data):
    boundary='tsknowledge'+uuid.uuid4().hex
    body=(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()+data+f'\r\n--{boundary}--\r\n'.encode())
    return json.loads(request(f'/api/knowledge/collections/{cid}/documents',body,content_type=f'multipart/form-data; boundary={boundary}'))
def save(): (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
try:
    collection=js('/api/knowledge/collections',{'name':'Knowledge end-to-end probe '+uuid.uuid4().hex[:8]})['collection'];cid=collection['id'];report['collectionId']=cid
    names=['ticket.txt','ticket.csv','ticket.json','ticket.html','ticket.docx','ticket.xlsx','ticket.pptx','ticket.pdf']
    if not a.skip_media:names+=['signal.png','ticket.wav','signal.mp4']
    for name in names:
        data=(pathlib.Path(a.fixtures)/name).read_bytes();started=time.perf_counter();result=upload(cid,name,data)
        entry={'file':name,'sha256':hashlib.sha256(data).hexdigest(),'seconds':time.perf_counter()-started,'result':result};report['cases'].append(entry);save();print(name,round(entry['seconds'],3),result,flush=True)
        assert not result.get('errors'),result
        assert len(result['documents'])==1,result
        doc=result['documents'][0];assert doc['chunkCount']>0 and doc['status']!='MetadataOnly',result
        raw=request(f'/api/knowledge/collections/{cid}/documents/{doc["id"]}/source?version={doc["version"]}');assert raw==data,'original changed'
    queries=['refund duplicate charge'] if a.skip_media else ['green traffic signal','service is down extremely angry','refund duplicate charge']
    for query in queries:
        started=time.perf_counter();hits=js('/api/knowledge/search',{'query':query,'collectionIds':[cid],'limit':8})['results'];report.setdefault('searches',[]).append({'query':query,'seconds':time.perf_counter()-started,'results':hits});save();print('search',query,[h['fileName'] for h in hits],flush=True)
        assert hits,'no search evidence'
        if query=='green traffic signal' and not a.skip_media:
            assert {h['fileName'] for h in hits[:2]}=={'signal.png','signal.mp4'},'visual evidence did not rank in top two'
        if query=='service is down extremely angry' and not a.skip_media:
            assert hits[0]['fileName']=='ticket.wav','audio evidence did not rank first'
            reference='The service is down with a software error. Fix it immediately. I am extremely angry.'
            expected=re.findall(r'\w+',reference.lower());actual=re.findall(r'\w+',hits[0]['text'].lower());row=list(range(len(actual)+1))
            for i,word in enumerate(expected,1):
                next_row=[i]
                for j,heard in enumerate(actual,1):next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(word!=heard)))
                row=next_row
            report['audioFixtureWordErrorRate']=row[-1]/len(expected)
    secret='The Larkspur rendezvous code is AMBER-KIWI-731. This private decision was recorded by Mira.'
    source=upload(cid,'private-code.txt',secret.encode())['documents'][0]
    session=js('/api/sessions',{});started=time.perf_counter()
    raw=request('/api/chat',{'sessionId':session['sessionId'],'knowledgeCollections':[cid], 'skills':[],'skills_discovery':False,'messages':[{'role':'user','content':'What is the Larkspur rendezvous code? Use the selected knowledge and cite the source URL.'}], 'maxTokens':256,'think':False})
    (out/'chat.sse').write_bytes(raw)
    events=[]
    for line in raw.decode().splitlines():
        if line.startswith('data: '):
            try:events.append(json.loads(line[6:]))
            except ValueError:pass
    answer=''
    for event in events:
        if 'replace' in event: answer=event['replace']
        elif 'token' in event: answer+=event['token']
    report['inference']={'seconds':time.perf_counter()-started,'answer':answer,'events':events,
        'hostCitationRepairCount':sum(event.get('knowledgeCitationRepairs',0) for event in events)};save();print('answer',answer,flush=True)
    assert 'AMBER-KIWI-731' in answer,events
    authoritative=[]
    for event in events:
        if 'knowledgeSources' in event:authoritative=event['knowledgeSources']
    report['inference']['authoritativeSources']=authoritative
    assert authoritative,'missing authoritative source metadata'
    for item in authoritative:assert request(item['sourceUrl']),'authoritative source did not resolve'
    report['inference']['authoritativeSourceDownloadsPassed']=True
    scoped=js('/api/knowledge/search',{'query':'AMBER-KIWI-731','collectionIds':[]})['results'];assert not scoped
    report['functionalPassed']=True
    # Final answer links are a separate quality check from authoritative source
    # chips. Record every failure and keep the overall probe failing when a model
    # invents a link, even if its answer and the host's source identities are right.
    urls=re.findall(r'/api/knowledge/collections/[^\s)\]\"<>]+',answer)
    citation_checks=[]
    for url in urls:
        try:
            assert request(url.rstrip('.')),'empty citation source'
            citation_checks.append({'url':url,'passed':True})
        except Exception as error:citation_checks.append({'url':url,'passed':False,'error':str(error)})
    report['inference']['finalAnswerCitationChecks']=citation_checks
    report['finalAnswerCitationsPassed']=bool(citation_checks) and all(item['passed'] for item in citation_checks)
    save()
    assert report['finalAnswerCitationsPassed'],'The final answer omitted or invented a source URL; authoritative source downloads passed.'
    report['passed']=True;save()
except Exception as error:
    report['passed']=False;report['error']=str(error);save();raise
