#!/usr/bin/env python3
"""Analytic screening of scaled residual Half planes for Gemma4 vision.

Reproduces the rejected prototype's conservative RMS/clamp input bounds and
unchanged3e-5 decomposition-error gate. This does not execute a projection or
prove the full numerical error/latency of a mixed matrix implementation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = GGUFReader(str(args.model), mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    def weight(name):
        tensor = tensors[name]
        return np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float32).reshape(tuple(reversed(tensor.shape)))
    def scalar(name, fallback):
        return float(weight(name).flat[0]) if name in tensors else fallback
    def bounds(prefix):
        return [scalar(prefix+suffix, fallback) for suffix, fallback in
            ((".input_min", -math.inf), (".input_max", math.inf), (".output_min", -math.inf), (".output_max", math.inf))]
    def clamp(bound, lo, hi):
        return max(abs(max(lo, min(hi, -bound))), abs(max(lo, min(hi, bound))))
    def clamped_l2(norm, lo, hi, inner):
        return norm if lo <= 0 <= hi else math.sqrt(inner)*clamp(norm, lo, hi)
    def metric(values, scale):
        high = values.astype(np.float16).astype(np.float32)
        low = ((values-high)*scale).astype(np.float16).astype(np.float32)/scale
        residual = values.astype(np.float64)-high.astype(np.float64)-low.astype(np.float64)
        norm = lambda x: float(np.max(np.linalg.norm(x.astype(np.float64), axis=1)))
        return dict(maximum=float(np.max(np.abs(values))), norm=math.nextafter(norm(values)*1.000001, math.inf),
            low_norm=math.nextafter(norm(low)*1.000001, math.inf),
            residual_norm=math.nextafter(norm(residual)*1.000001, math.inf))
    def extra_error(m, input_l2, inner, scale):
        floor = math.sqrt(inner)*2**-25/scale
        low = input_l2*2**-11+floor
        residual = input_l2*2**-22+floor
        return low*m["low_norm"] + residual*m["norm"] + (input_l2+residual)*m["residual_norm"]
    hidden = int(reader.fields["clip.vision.embedding_length"].contents())
    layers = int(reader.fields["clip.vision.block_count"].contents())
    heads = int(reader.fields["clip.vision.attention.head_count"].contents())
    cases = []
    for layer in range(layers):
        p = f"v.blk.{layer}."
        norm1 = math.sqrt(hidden)*float(np.max(np.abs(weight(p+"ln1.weight"))))*1.01
        norm2 = math.sqrt(hidden)*float(np.max(np.abs(weight(p+"ln2.weight"))))*1.01
        act_bounds = []
        for name in ("ffn_gate", "ffn_up"):
            values = weight(p+name+".weight"); lo,hi,olo,ohi = bounds(p+name)
            row_norm = float(np.max(np.linalg.norm(values.astype(np.float64),axis=1)))
            act_bounds.append(clamp(clamped_l2(norm2,lo,hi,hidden)*row_norm*1.01+3e-5,olo,ohi))
        for name in ("attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"):
            values = weight(p+name+".weight"); inner = values.shape[1]
            lo,hi,_,_ = bounds(p+name)
            if name in ("attn_q", "attn_k", "attn_v"):
                maximum, l2 = clamp(norm1,lo,hi), clamped_l2(norm1,lo,hi,inner)
            elif name == "attn_out":
                maximum = clamp(math.sqrt(hidden//heads)*1.01,lo,hi)
                l2 = clamped_l2(math.sqrt(hidden)*1.01,lo,hi,inner)
            elif name in ("ffn_gate", "ffn_up"):
                maximum, l2 = clamp(norm2,lo,hi), clamped_l2(norm2,lo,hi,inner)
            else:
                maximum = clamp(act_bounds[0]*act_bounds[1]*1.01,lo,hi)
                l2 = math.sqrt(inner)*maximum
            record = dict(tensor=p+name+".weight", inner=inner, columns=values.shape[0],
                conservative_input_maximum=maximum, conservative_input_l2=l2, decompositions=[])
            for scale in (1,1024):
                m = metric(values,scale); error = extra_error(m,l2,inner,scale)
                record["decompositions"].append(dict(residual_scale=scale, weight=m,
                    decomposition_error_bound=error, decomposition_gate=3e-5,
                    eligible=maximum<=16384 and m["maximum"]<=16384 and math.isfinite(error) and error<=3e-5))
            cases.append(record)
    counts = {str(scale):sum(case["decompositions"][index]["eligible"] for case in cases) for index,scale in enumerate((1,1024))}
    report = dict(completed=True,model=str(args.model.resolve()),model_sha256=fingerprint(args.model),
        tool_sha256=fingerprint(__file__),projection_count=len(cases),eligible_counts=counts,cases=cases,
        limitations="Analytic decomposition-error screening only, not total floating-point accumulation error, GPU correctness, range/fallback safety, model quality or performance. No runtime route is changed.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(projections=len(cases),eligible_counts=counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
