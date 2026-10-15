#!/usr/bin/env python3
"""Independent LTX 2.5 audio oracle from the unchanged pinned original modules.

The default emits a tiny full causal decoder + both AMP vocoders, using the real
checkpoint's architecture and BF16 filter buffers. --checkpoint instead evaluates
an actual downloaded companion and writes ignored numerical evidence.
"""
import argparse, copy, hashlib, json, struct, subprocess, sys
from pathlib import Path
REVISION = 'be6d6d0a69d11a0a399c462ed92606692e5f8697'

def write_safetensors(path,state,config):
    header={'__metadata__':{'config':json.dumps(config,separators=(',',':'))}}
    payload=bytearray()
    for key,value in state.items():
        raw=value.detach().float().contiguous().numpy().astype('<f4').tobytes()
        header[key]={'dtype':'F32','shape':list(value.shape),'data_offsets':[len(payload),len(payload)+len(raw)]}
        payload.extend(raw)
    h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
    path.write_bytes(struct.pack('<Q',len(h))+h+payload)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--buffer-checkpoint',type=Path,default=Path('/Users/zhongkaifu/work/models/ltx25/vae/ltx-2.5-audio-vae-bf16.safetensors'));p.add_argument('--header',type=Path,default=Path('docs/validation/ltx25/ltx-2.5-audio-vae-bf16-header.json'))
    p.add_argument('--output',type=Path,default=Path('InferenceWeb.Tests/Fixtures/Ltx25'));p.add_argument('--tokens',type=int,nargs='+',default=[1,2,3]);a=p.parse_args()
    rev=subprocess.check_output(['git','-C',str(a.reference),'rev-parse','HEAD'],text=True).strip()
    if rev!=REVISION or subprocess.check_output(['git','-C',str(a.reference),'status','--porcelain'],text=True).strip():raise SystemExit('The pinned original checkout must remain unchanged')
    sys.path.insert(0,str(a.reference.resolve()/'packages/ltx-core/src'))
    import numpy as np, torch
    from safetensors.torch import load_file
    from safetensors import safe_open
    from ltx_core.model.audio_vae.model_configurator import AudioDecoderConfigurator,VocoderConfigurator
    from ltx_core.model.audio_vae.ops import PerChannelStatistics
    torch.set_num_threads(1);torch.manual_seed(4619)
    config=json.loads(json.loads(a.header.read_text())['__metadata__']['config'])
    if not a.checkpoint:
        config=copy.deepcopy(config);config['audio_vae']['model']['params']['ddconfig']['ch']=4
        config['vocoder']['vocoder']['upsample_initial_channel']=64;config['vocoder']['bwe']['upsample_initial_channel']=32
    decoder=AudioDecoderConfigurator.from_metadata({'config':config}).eval()
    # Production statistics describe 8*16 packed latent features, independently
    # of the fixture's reduced hidden-channel widths.
    decoder.per_channel_statistics=PerChannelStatistics(latent_channels=128)
    vocoder=VocoderConfigurator.from_metadata({'config':config}).eval()
    if a.checkpoint:
        state=load_file(str(a.checkpoint))
        ds={k.removeprefix('audio_vae.'):v for k,v in state.items() if k.startswith('audio_vae.decoder.') or k.startswith('audio_vae.per_channel_statistics.')}
        # The official filter remaps audio_vae.decoder.* to decoder-local keys.
        ds={k.removeprefix('decoder.'):v for k,v in ds.items()};decoder.load_state_dict(ds,strict=True)
        vocoder.load_state_dict({k.removeprefix('vocoder.'):v for k,v in state.items() if k.startswith('vocoder.')},strict=True)
        decoder.float();vocoder.float()
    else:
        with torch.no_grad():
            for name,tensor in list(decoder.named_parameters())+list(vocoder.named_parameters()):
                tensor.copy_(torch.randn_like(tensor)*.025)
            decoder.per_channel_statistics.get_buffer('mean-of-means').copy_(torch.randn(128)*.1)
            decoder.per_channel_statistics.get_buffer('std-of-means').copy_(torch.rand(128)*.7+.4)
            # Actual checkpoint buffers use BF16; preserve this precision too.
            for module in (decoder,vocoder):
                for name,buffer in module.named_buffers():
                    if buffer.is_floating_point():buffer.copy_(buffer.bfloat16().float())
    if not a.checkpoint:
        with safe_open(str(a.buffer_checkpoint),framework='pt',device='cpu') as original_buffers:
            vocoder.mel_stft.load_state_dict({k.removeprefix('vocoder.mel_stft.'):original_buffers.get_tensor(k).float() for k in original_buffers.keys() if k.startswith('vocoder.mel_stft.')})
    a.output.mkdir(parents=True,exist_ok=True)
    fixture=a.output/'audio-decoder.bin'
    if not a.checkpoint:
        state={}
        for k,v in decoder.state_dict().items():
            state['audio_vae.'+k if k.startswith('per_channel_statistics.') else 'audio_vae.decoder.'+k]=v
        state.update({'vocoder.'+k:v for k,v in vocoder.state_dict().items()})
        write_safetensors(fixture,state,config)
    cases=[]
    for tokens in a.tokens:
        packed=torch.randn(1,tokens,128)*.2
        z=packed.reshape(1,tokens,8,16).permute(0,2,1,3)
        with torch.inference_mode():
            mel=decoder(z);native=vocoder.vocoder(mel)
            bwe_mel=vocoder.mel_stft.mel_spectrogram(native.reshape(2,-1))[0].reshape(1,2,64,-1).transpose(2,3)
            residual=vocoder.bwe_generator(bwe_mel);pcm=vocoder(mel)
        def flat(t):return t.detach().float().flatten().tolist()
        cases.append({'tokens':tokens,'latents':flat(packed),'mel':flat(mel.permute(0,2,3,1)),
                      'native16k':flat(native.transpose(1,2)),
                      'bweMel':flat(bwe_mel.transpose(2,3).reshape(1,128,-1).transpose(1,2)),
                      'bweResidual':flat(residual.transpose(1,2)),'pcm48k':flat(pcm.transpose(1,2))})
    source=a.checkpoint or fixture
    report={'reference_revision':rev,'checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'config':config,'cases':cases,'scope':'Full stereo causal decoder,16k AMP vocoder,checkpoint STFT/mel,48k learned BWE and Hann sinc skip; Float32.'}
    (a.output/('audio-actual-oracle.json' if a.checkpoint else 'audio-oracle.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'output':str(a.output),'checkpoint_sha256':report['checkpoint_sha256']}))
if __name__=='__main__':main()
