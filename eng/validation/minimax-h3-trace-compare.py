#!/usr/bin/env python3
"""Compare matched SD public-callback and direct H3 boundary snapshots."""
import argparse
import json
from pathlib import Path
import numpy as np


def metrics(reference, candidate):
    a,b=reference.astype(np.float64),candidate.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f'shape mismatch: {a.shape} vs {b.shape}')
    error=a-b
    return {'elements':a.size,'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
            'cosine':float(np.dot(a.ravel(),b.ravel())/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),
            'relative_l2':float(np.linalg.norm(error)/max(np.linalg.norm(a),1e-30)),
            'max_abs':float(np.max(np.abs(error))),
            'rms_reference':float(np.sqrt(np.mean(a*a))), 'rms_candidate':float(np.sqrt(np.mean(b*b)))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--text-tokens',type=int,default=22)
    parser.add_argument('--audio-frames',type=int,default=8)
    args=parser.parse_args()
    items=[json.loads(path.read_text()) for path in args.reference.glob('*.f32.json')]
    nodes={item['id']:item for item in map(json.loads,(args.reference/'nodes.jsonl').read_text().splitlines())}
    def load(item): return np.fromfile(args.reference/item['file'],dtype='<f4')
    def compare(item,candidate):
        return {'reference_file':item['file'],'candidate_file':candidate.name,
                **metrics(load(item),np.fromfile(candidate,dtype='<f4'))}
    def ancestors(item):
        found=set()
        todo=[item]
        while todo:
            node=todo.pop()
            for source in node.get('sources',[]):
                if source['id'] in found: continue
                found.add(source['id'])
                yield source
                if source['id'] in nodes: todo.append(nodes[source['id']])
    report={'scope':'synchronized diagnostic boundaries, excluded from performance timings','inputs':{},'blocks':[]}
    for suffix,candidate in [('audio_patch_proj.weight','physical-audio-input.f32'),
                             ('video_patch_proj.weight','video-input.f32'),
                             ('condition_proj.weight','text-hidden-input.f32')]:
        projected=[item for item in items if item['file'].startswith('projection-') and
                   any(suffix in source['name'] for source in item['sources'])]
        if len(projected)!=1: raise ValueError(f'require exactly one projection for {suffix}, found {len(projected)}')
        source_id=projected[0]['sources'][1]['id']
        input_item=next(item for item in items if item['id']==source_id)
        report['inputs'][candidate]=compare(input_item,args.candidate/candidate)
    packed_file=args.candidate/'packed-in.f32'
    candidates=[item for item in items if item['file'].startswith('packed-in-') and
                np.prod(item['ne'])==packed_file.stat().st_size//4]
    if not candidates: raise ValueError('missing complete packed input')
    # The first block also rebuilds modulation/residual spans with concatenation;
    # the earliest full-sequence concat is the unmodulated packed input.
    report['packed_input']=compare(min(candidates,key=lambda item:item['id']),packed_file)
    packed_reference=load(min(candidates,key=lambda item:item['id'])).reshape(-1,candidates[0]['ne'][0])
    packed_candidate=np.fromfile(packed_file,dtype='<f4').reshape(packed_reference.shape)
    boundaries=[0,args.text_tokens,args.text_tokens+2*args.audio_frames,packed_reference.shape[0]]
    report['packed_input']['segments']={label:metrics(packed_reference[start:end],packed_candidate[start:end])
                                     for label,start,end in zip(('text','audio','video'),boundaries,boundaries[1:])}
    report['text_refiner']={}
    for name in ('condition-proj','refiner-0','refiner-1'):
        candidate=args.candidate/f'{name}.f32'
        if not candidate.is_file():
            continue
        selected=[item for item in items if item['file'].startswith(name+'-')]
        if len(selected)!=1:
            raise ValueError(f'require exactly one {name} boundary, found {len(selected)}')
        report['text_refiner'][name]=compare(selected[0],candidate)
    for block in range(50):
        selected=[item for item in items if item['file'].startswith(f'block-{block}-') and
                  item['name']==f'ggml_runner_cut:minimax_h3.blocks.{block}|hidden_states|']
        if len(selected)!=1: raise ValueError(f'require one block {block}, found {len(selected)}')
        item=selected[0]
        candidate=args.candidate/f'block-{block}.f32'
        result={'block':block,**compare(item,candidate)}
        a=load(item).reshape(item['ne'][1],item['ne'][0])
        b=np.fromfile(candidate,dtype='<f4').reshape(a.shape)
        boundaries=[0,args.text_tokens,args.text_tokens+2*args.audio_frames,a.shape[0]]
        result['segments']={label:metrics(a[start:end],b[start:end]) for label,start,end in
                            zip(('text','audio','video'),boundaries,boundaries[1:])}
        report['blocks'].append(result)
    report['raw_outputs']={}
    for name,width in [('audio',32),('video',96)]:
        selected=[item for item in items if item['file'].startswith('final-') and item['op']=='ADD' and
                  item['ne'][0]==width and
                  any(f'final_layer.{name}_out.weight' in source['name'] for source in ancestors(item))]
        if len(selected)!=1: raise ValueError(f'require one raw {name} output, found {len(selected)}')
        report['raw_outputs'][name]=compare(selected[0],args.candidate/f'raw-{name}-output.f32')
    carrier=[item for item in items if item['file'].startswith('final-') and item['op']=='ADD' and
             item['ne']==[args.audio_frames,2,32,1] and
             any('final_layer.audio_out.weight' in source['name'] for source in ancestors(item))]
    if len(carrier)!=1: raise ValueError(f'require one carrier audio velocity, found {len(carrier)}')
    item=carrier[0]
    reference=load(item).reshape(32,2,args.audio_frames).transpose(1,2,0).ravel()
    path=args.candidate/'carrier-audio-velocity.f32'
    report['carrier_audio_velocity']={'reference_file':item['file'],'candidate_file':path.name,
                                      **metrics(reference,np.fromfile(path,dtype='<f4'))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'inputs':report['inputs'],'packed_input':report['packed_input'],
                      'text_refiner':report['text_refiner'],
                      'block0':report['blocks'][0],'block49':report['blocks'][-1],
                      'raw_outputs':report['raw_outputs'],'carrier_audio_velocity':report['carrier_audio_velocity']},indent=2))


if __name__=='__main__':
    main()
