#!/usr/bin/env python3
"""Decode an exported clip into a source-panel preview and sampled video frames.

Visual review only. Source previews are not exact VAE preprocessing or quality
metrics. Audio validation and perceptual conclusions require separate evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--image',action='append',type=Path,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--height',type=int,required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--fps',type=int,default=24)
    p.add_argument('--samples',type=int,default=17)
    p.add_argument('--tile',type=int,default=256)
    p.add_argument('--ffmpeg',default=shutil.which('ffmpeg'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.ffmpeg or min(a.width,a.height,a.frames,a.samples,a.tile,a.fps)<=0: raise ValueError('Missing FFmpeg or invalid geometry')
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    raw=a.output/'decoded.rgb.u8'
    subprocess.run([a.ffmpeg,'-v','error','-xerror','-i',str(a.video),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24',str(raw)],check=True)
    if raw.stat().st_size!=a.frames*a.height*a.width*3: raise ValueError('Full decoded video byte count differs from contract')
    rgb=np.memmap(raw,dtype=np.uint8,mode='r',shape=(a.frames,a.height,a.width,3))
    indices=np.linspace(0,a.frames-1,a.samples).round().astype(int).tolist()
    h=max(1,round(a.tile*a.height/a.width)); rowheight=h+24
    sheet=Image.new('RGB',(a.tile*a.samples,rowheight*2),'#171717'); draw=ImageDraw.Draw(sheet)
    previews=[Image.open(path).convert('RGB') for path in a.image]
    for column in range(a.samples):
        source=previews[column%len(previews)]
        sheet.paste(ImageOps.fit(source,(a.tile,h)),(column*a.tile,24))
        frame=Image.fromarray(np.asarray(rgb[indices[column]])).resize((a.tile,h),Image.Resampling.LANCZOS)
        sheet.paste(frame,(column*a.tile,rowheight+24))
        draw.text((column*a.tile+3,rowheight+5),f'frame {indices[column]} / {indices[column]/a.fps:.2f}s',fill='white')
        frame.save(a.output/f'frame-{indices[column]:03d}.png')
    draw.text((5,5),'Uploaded panel previews (not exact preprocessing); sampled exported video below',fill='white')
    sheet.save(a.output/'contact-sheet.png')
    report=dict(completed=True,scope=__doc__,video=str(a.video.resolve()),video_sha256=sha(a.video),
                images=[dict(path=str(x.resolve()),sha256=sha(x)) for x in a.image],
                geometry=[a.frames,a.height,a.width,3],fps=a.fps,sampled_frames=indices,
                ffmpeg=dict(path=str(Path(a.ffmpeg).resolve()),sha256=sha(a.ffmpeg)),decoded_rgb_sha256=sha(raw),
                audio_validation_included=False,perceptual_assessment=None)
    (a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(completed=True,output=str(a.output),frames=a.frames)))


if __name__=='__main__': main()
