#!/usr/bin/env python3
"""Compare matched native generation reports, timing and raw RGB differences.

Pixel differences are descriptive and do not establish perceptual-quality parity.
The recorded model path must match; unchanged checkpoint identity is established
by the surrounding source/model manifest, not inferred from equal path strings.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-bitwise',action='store_true',help='Require exact RGB and both raw audio channels for data-movement changes.')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    ds=[json.loads(x.read_text()) for x in [a.before,a.after]]
    if any(not d['completed'] for d in ds): raise ValueError('Generation incomplete')
    for k in ['settings','prompt','mode','sigmas','model']:
        if ds[0][k]!=ds[1][k]:raise ValueError('Generation contract differs: '+k)
    def sources(d):return [{k:v for k,v in x.items() if k!='path'} for x in d['reference_inputs']]
    if sources(ds[0])!=sources(ds[1]):raise ValueError('Reference source identity/position/strength differs')
    if ds[0].get('identity_adapter',{}).get('sha256')!=ds[1].get('identity_adapter',{}).get('sha256'):raise ValueError('Ingredients adapter identity differs')
    if len(ds[0]['executions'])!=len(ds[1]['executions']):raise ValueError('Execution count differs')
    rows=[]
    for x,y in zip(ds[0]['executions'],ds[1]['executions']):
        if (x['repeat'],x['prompt'],x['kind'])!=(y['repeat'],y['prompt'],y['kind']):raise ValueError('Execution identity differs')
        paths=[Path(str(a.before)+f'.{x["repeat"]}.rgb.f32'),Path(str(a.after)+f'.{y["repeat"]}.rgb.f32')]
        for path,e in zip(paths,[x,y]):
            if sha(path)!=e['video_sha256']:raise ValueError('Recorded RGB checksum changed')
        arrays=[np.memmap(path,dtype='<f4',mode='r') for path in paths]
        if arrays[0].size!=arrays[1].size:raise ValueError('Raw RGB lengths differ')
        square=reference_square=maximum=0.;count=0
        for start in range(0,arrays[0].size,1<<20):
            old,new=[np.asarray(v[start:start+(1<<20)],dtype=np.float64) for v in arrays]
            if not np.isfinite(old).all() or not np.isfinite(new).all():raise ValueError('Nonfinite RGB')
            delta=new-old;square+=float(delta@delta);reference_square+=float(old@old);maximum=max(maximum,float(np.abs(delta).max()));count+=old.size
        mse=square/count
        if x['audio_samples']!=y['audio_samples']:raise ValueError('Raw audio sample counts differ')
        audio=[]
        for channel in range(2) if x['audio_samples'] else []:
            audio_paths=[Path(str(report)+f'.{execution["repeat"]}.audio-{channel}.f32') for report,execution in [(a.before,x),(a.after,y)]]
            hashes=[]
            for path in audio_paths:
                values=np.fromfile(path,dtype='<f4')
                if values.size!=x['audio_samples'] or not np.isfinite(values).all():raise ValueError('Invalid raw audio layout or nonfinite samples')
                hashes.append(sha(path))
            audio.append(dict(channel=channel,before_sha256=hashes[0],after_sha256=hashes[1],bitwise_equal=hashes[0]==hashes[1]))
        equal=x['video_sha256']==y['video_sha256'] and all(channel['bitwise_equal'] for channel in audio)
        if a.require_bitwise and not equal:raise ValueError('RGB/audio bitwise parity failed')
        rows.append(dict(repeat=x['repeat'],before_ms=x['generation_ms'],after_ms=y['generation_ms'],generation_speedup=x['generation_ms']/y['generation_ms'],raw_rgb_bitwise_equal=x['video_sha256']==y['video_sha256'],raw_audio=audio,max_absolute=maximum,rmse=math.sqrt(mse),relative_l2=math.sqrt(square/max(reference_square,1e-30)),psnr_db=-10*math.log10(mse) if mse else None,after_denoiser_diagnostics=y.get('denoiser_diagnostics')))
    out=dict(completed=True,scope=__doc__,input_contract_matched=True,require_bitwise=a.require_bitwise,perceptual_quality_pass_claimed=False,before=str(a.before),after=str(a.after),before_report_sha256=sha(a.before),after_report_sha256=sha(a.after),executions=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
