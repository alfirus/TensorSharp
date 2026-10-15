#!/usr/bin/env python3
"""Compare one H3 denoiser-step physical endpoints through the real stereo VAE.

At sigma1, video/audio shift12/3, carrier drift and endpoint scaling simplify
to cleanPhysical = physicalInput + rawPrediction. This is a one-step diagnostic,
not an independent oracle for the whole text encoder or eight-step generation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-input',required=True,type=Path)
    parser.add_argument('--torch-prediction',required=True,type=Path)
    parser.add_argument('--candidate-prediction',required=True,type=Path)
    parser.add_argument('--audio-probe',required=True,type=Path)
    parser.add_argument('--vae',required=True,type=Path)
    parser.add_argument('--reference-media',type=Path)
    parser.add_argument('--frames',type=int,default=8)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    expected=2*args.frames*32
    physical=np.fromfile(args.physical_input,dtype='<f4')
    if physical.size!=expected or not np.isfinite(physical).all():
        raise ValueError('Invalid physical input')
    spec=importlib.util.spec_from_file_location('h3refiner',Path(__file__).with_name('minimax-h3-refiner-reference.py'))
    reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    sources=[args.physical_input,args.torch_prediction,args.candidate_prediction,args.audio_probe,args.audio_probe.parent/'TensorSharp.Models.dll',args.audio_probe.parent/'TensorSharp.Backends.Cuda.dll',args.audio_probe.parent/'cuda_kernels/tensorsharp_kernels.ptx']
    report={'scope':'One denoiser-step endpoint on identical SD physical/text inputs; real public stereo VAE and PCM16 codec; eight-step/full-text truth not established',
            'sigma':1,'video_shift':12,'audio_shift':3,'endpoint_formula':'physicalInput + rawPrediction',
            'sources':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},'decodes':{}}
    waves,pcm={},{}
    for name,prediction in [('torch',args.torch_prediction),('tensorsharp',args.candidate_prediction)]:
        raw=np.fromfile(prediction,dtype='<f4')
        if raw.size!=expected or not np.isfinite(raw).all():
            raise ValueError('Invalid raw prediction')
        endpoint=np.add(physical,raw,dtype=np.float32)
        tokens=args.output/(name+'-endpoint.f32'); endpoint.tofile(tokens)
        wave=args.output/(name+'-waveform.f32')
        command=['dotnet',str(args.audio_probe.resolve()),'--vae',str(args.vae.resolve()),'--tokens',str(tokens.resolve()),'--frames',str(args.frames),'--output',str(wave.resolve())]
        with (args.output/(name+'-decode.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode!=0:
            raise RuntimeError(f'{name} decoder exited {result.returncode}')
        metadata=json.loads(Path(str(wave)+'.json').read_text())
        if not metadata['completed'] or metadata['channels']!=2 or metadata['sampleRate']!=32000:
            raise ValueError('Decoder did not complete stereo output')
        waves[name]=np.fromfile(wave,dtype='<f4')
        if not np.isfinite(waves[name]).all():
            raise ValueError('Nonfinite waveform')
        encoded=args.output/(name+'-waveform.wav')
        subprocess.run(['ffmpeg','-v','error','-f','f32le','-ar','32000','-ac','2','-i',str(wave),'-c:a','pcm_s16le',str(encoded)],check=True,timeout=60)
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(encoded),'-f','f32le','-acodec','pcm_f32le','pipe:1'],capture_output=True,check=True,timeout=60)
        pcm[name]=np.frombuffer(decoded.stdout,dtype='<f4')
        report['decodes'][name]={'argv':command,'exit_code':result.returncode,**metadata,'pcm16_file':str(encoded),'pcm16_sha256':hashlib.sha256(encoded.read_bytes()).hexdigest()}
    report['float32_waveform']=reference.metrics(waves['torch'],waves['tensorsharp'])
    report['pcm16_waveform']=reference.metrics(pcm['torch'],pcm['tensorsharp'])
    if args.reference_media:
        decoded=subprocess.run(['ffmpeg','-v','error','-i',str(args.reference_media),'-map','0:a:0','-f','f32le','-acodec','pcm_f32le','pipe:1'],capture_output=True,check=True,timeout=60)
        sd=np.frombuffer(decoded.stdout,dtype='<f4')
        report['clean_sdcpp_pcm16_vs_torch']=reference.metrics(pcm['torch'],sd)
    report['completed']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['completed','float32_waveform','pcm16_waveform','clean_sdcpp_pcm16_vs_torch'] if key in report},indent=2))


if __name__=='__main__':
    main()
