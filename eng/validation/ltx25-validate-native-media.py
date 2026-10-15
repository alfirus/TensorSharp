#!/usr/bin/env python3
"""Independently validate full native probe H264/AAC and raw float PCM sidecars."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--media-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    d=json.loads(a.report.read_text())
    if not d['completed']: raise ValueError('Native workflow did not complete')
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    if not ffmpeg or not ffprobe: raise FileNotFoundError('FFmpeg and ffprobe required')
    rows=[]
    for e in d['executions']:
        original=Path(e['media_path'])
        video=a.media_root/original.name if a.media_root else original
        if sha(video)!=e['media_sha256']: raise ValueError('Exported video checksum changed')
        metadata=json.loads(subprocess.run([ffprobe,'-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True,check=True).stdout)
        vs=[s for s in metadata['streams'] if s['codec_type']=='video'];au=[s for s in metadata['streams'] if s['codec_type']=='audio']
        if len(vs)!=1: raise ValueError('Expected one video stream')
        v=vs[0];settings=d['settings']
        if (v['codec_name'],v['width'],v['height'],int(v['nb_read_frames']),v['avg_frame_rate'])!=('h264',settings['Width'],settings['Height'],settings['Frames'],str(settings['Fps'])+'/1'): raise ValueError('Video codec/geometry differs')
        audio=[]
        if e['audio_muxed']:
            if len(au)!=1 or (au[0]['codec_name'],int(au[0]['sample_rate']),au[0]['channels'])!=('aac',48000,2): raise ValueError('Audio stream differs')
            for ch in range(2):
                path=Path(str(video).removesuffix('.mp4')+f'.audio-{ch}.f32')
                values=np.fromfile(path,dtype='<f4')
                if values.size!=e['audio_samples'] or not np.isfinite(values).all(): raise ValueError('Raw native PCM layout or samples differ')
                audio.append(dict(path=str(path),sha256=sha(path),samples=int(values.size),peak_absolute=float(np.abs(values).max())))
        command=[ffmpeg,'-v','error','-xerror','-i',str(video),'-map','0:v:0']
        if e['audio_muxed']: command+=['-map','0:a:0']
        subprocess.run(command+['-f','null','-'],capture_output=True,check=True)
        rows.append(dict(repeat=e['repeat'],video=str(video),original_video=str(original),video_sha256=sha(video),raw_pcm=audio,ffprobe=metadata,full_decode_passed=True))
    out=dict(status='passed',scope='Independent complete H264/AAC decode, exact geometry/count, finite Float32 stereo PCM sidecars. This does not establish perceptual quality.',report=str(a.report),report_sha256=sha(a.report),media_root_override=str(a.media_root) if a.media_root else None,media=rows,tools={n:dict(path=x,sha256=sha(Path(x))) for n,x in [('ffmpeg',ffmpeg),('ffprobe',ffprobe)]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status='passed',fully_decoded_media=len(rows))))


if __name__=='__main__': main()
