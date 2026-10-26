#!/usr/bin/env python3
"""Create tiny MXFP8/decoded-F32 fixtures from dsv41-fixture.py output.

Only generated fixtures are accepted. Publisher checkpoint files are never
rewritten. The split MXFP8 file exercises the same dedicated PLE layout as
Unsloth, alongside a decoded-F32 oracle with exactly the same learned values.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def string(value):
    raw = value.encode()
    return struct.pack("<Q",len(raw))+raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir",type=Path)
    parser.add_argument("output_dir",type=Path)
    parser.add_argument("--type",type=int,choices=[43,67],default=43)
    args = parser.parse_args()
    meta = json.loads((args.fixture_dir/"deepseek41.config.json").read_text())
    if not meta.get("fixture"): parser.error("only generated numerical fixtures are accepted")
    reader = GGUFReader(args.fixture_dir/"deepseek41-fixture.gguf")
    tables = [t for t in reader.tensors if ".engram_embd." in t.name]
    if len(tables)!=1 or tables[0].tensor_type.name!="F32" or tables[0].shape[0]%32:
        parser.error("expected one generated F32 Engram table with complete 32-value rows")
    table = tables[0]
    width,rows = map(int,table.shape)
    codes = np.arange(256,dtype=np.uint8)
    exponent, mantissa = (codes>>3)&15,codes&7
    lut = np.where(exponent==0,np.ldexp(mantissa.astype(np.float32),-9),
        np.ldexp(1+mantissa.astype(np.float32)/8,exponent.astype(np.int32)-7))
    lut = np.where(codes&128,-lut,lut).astype(np.float32)
    lut[(exponent==15)&(mantissa==7)] = np.inf
    quant = np.argmin(np.abs(table.data[:,:,None]-lut[None,None,:]),axis=2).astype(np.uint8)
    decoded = lut[quant]
    packed = np.empty((rows,width//32,33),dtype=np.uint8)
    packed[:,:,0]=127
    packed[:,:,1:]=quant.reshape(rows,width//32,32)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in ("deepseek41.config.json","tokens.json"):
        shutil.copyfile(args.fixture_dir/name,args.output_dir/name)
    field_values={name:field.contents() for name,field in reader.fields.items() if not name.startswith("GGUF.")}
    mapping={
        "deepseek41.engram.layer_ids":"deepseek41.ple.layers",
        "deepseek41.engram.max_ngram_size":"deepseek41.ple.ngram_size",
        "deepseek41.engram.head_count":"deepseek41.ple.heads_per_ngram",
        "deepseek41.engram.key_length":"deepseek41.embedding_length_per_layer_input",
        "deepseek41.engram.multipliers":"deepseek41.ple.layer_multipliers",
        "deepseek41.engram.primes":"deepseek41.ple.head_vocab_sizes",
        "deepseek41.engram.offsets":"deepseek41.ple.head_offsets",
        "deepseek41.engram.pad_id":"deepseek41.engram.pad_token_id"}
    first=args.output_dir/"deepseek41-fixture-00001-of-00002.gguf"
    oracle=args.output_dir/"deepseek41-decoded-f32.gguf"
    for path,split in ((first,True),(oracle,False)):
        writer=GGUFWriter(path,"deepseek41")
        for name,value in field_values.items():
            if name=="general.architecture" or name.startswith("split."):continue
            field=reader.fields[name]
            writer.add_key_value(mapping.get(name,name) if split else name,value,field.types[0],
                field.types[1] if len(field.types)>1 else None)
        if split:
            writer.add_key_value("split.count",2,GGUFValueType.UINT32)
            writer.add_key_value("split.no",0,GGUFValueType.UINT32)
        for tensor in reader.tensors:
            if tensor.name==table.name:
                if not split:writer.add_tensor(tensor.name,decoded)
            else:writer.add_tensor(tensor.name,tensor.data,raw_dtype=tensor.tensor_type)
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    header=struct.pack("<IIQQ",0x46554747,3,1,1)
    header+=string("general.alignment")+struct.pack("<II",4,32)
    header+=string("per_layer_token_embd.weight")+struct.pack("<IQQIQ",2,width,rows,args.type,0)
    header+=bytes((-len(header))%32)
    second=args.output_dir/"deepseek41-fixture-00002-of-00002.gguf"
    second.write_bytes(header+packed.tobytes())
    print(json.dumps({"model":str(first),"reference":str(oracle),"type":args.type,
        "row_values":width,"rows":rows,"packed_bytes":packed.nbytes}))


if __name__=="__main__":main()
