#!/usr/bin/env python3
"""Create deterministic synthetic H3 keyframe/reference inputs in ignored evidence storage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import wave
import zlib


def scene(path, width, height, center):
    rows = bytearray()
    radius = height*.19
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rgb = [170, 210, 235] if y < height*.7 else [110, 150, 175]
            dx, dy = x-center, y-height*.52
            distance = math.hypot(dx, dy)/radius
            if distance < 1:
                light = .45+.55*math.sqrt(1-distance*distance)
                rgb = [int(235*light), int(35*light), int(25*light)]
            rows.extend(rgb)
    def chunk(name, data):
        return struct.pack(">I",len(data))+name+data+struct.pack(">I",zlib.crc32(name+data)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+
                     chunk(b"IDAT",zlib.compress(bytes(rows)))+chunk(b"IEND",b""))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--width",type=int,default=256)
    parser.add_argument("--height",type=int,default=160)
    args=parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Positive geometry required")
    args.output.mkdir(parents=True,exist_ok=False)
    first,last,audio=[args.output/name for name in ("first.png","last.png","reference.wav")]
    scene(first,args.width,args.height,args.width*.35)
    scene(last,args.width,args.height,args.width*.65)
    with wave.open(str(audio),"wb") as stream:
        stream.setparams((2,2,32000,0,"NONE","not compressed"))
        pcm=bytearray()
        for index in range(32000):
            envelope=math.sin(math.pi*index/32000)**2
            value=round(3000*envelope*math.sin(2*math.pi*440*index/32000))
            pcm.extend(struct.pack("<hh",value,value))
        stream.writeframes(pcm)
    report={"purpose":"Synthetic conditioning execution coverage; not a perceptual-quality dataset",
            "geometry":[args.width,args.height],"audio":{"rate":32000,"channels":2,"seconds":1},
            "files":[{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in (first,last,audio)]}
    (args.output/"inputs.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    main()
