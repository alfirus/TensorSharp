#!/usr/bin/env python3
"""Compare exact native V4.1 pipeline stages against an unsplit fixture.

Generate the required tiny F32 fixture with eng/dsv41-fixture.py --f32.
This validates native graph/cache boundaries, not remote transport or model
quality. It never accepts a downloaded full-size checkpoint as a fixture.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--backend", default="CPU")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cut", type=int, default=3)
    parser.add_argument("--ubatch", type=int, default=8)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Alternative first shard, e.g. the owned MXFP8 fixture")
    parser.add_argument("--reference-model", type=Path, help="Identical decoded-F32 reference for an MXFP8 fixture")
    parser.add_argument("--cache-mb", type=int, default=0, help="Opt-in stage expert cache MiB per GPU; reference stays resident")
    parser.add_argument("--expect-cache-activity", action="store_true", help="Require actual GPU cache calls/hits in the report")
    parser.add_argument("--cpu-prefetch",action="store_true",help="Prefetch selected host expert ranges on stage CPU paths")
    parser.add_argument("--expert-pread",action="store_true",help="Read stage GPU-cache misses through bounded persistent-descriptor staging")
    parser.add_argument("--host-cache-mb",type=int,default=0,help="Process-wide exclusive quantized host victim cache MiB")
    parser.add_argument("--device-bridge",action="store_true",help="Keep stage cached-MoE activations/routes/results on CUDA")
    args = parser.parse_args()
    if args.cache_mb<0 or (args.expect_cache_activity and args.cache_mb==0):
        parser.error("cache MiB must be nonnegative, and --expect-cache-activity requires a positive cache")
    if args.expert_pread and not args.cache_mb:
        parser.error("--expert-pread requires --cache-mb")
    if args.host_cache_mb<0 or ((args.host_cache_mb or args.device_bridge) and not args.cache_mb):
        parser.error("host cache/device bridge require --cache-mb, and host cache must be nonnegative")
    meta = json.loads((args.fixture_dir / "deepseek41.config.json").read_text())
    if not meta.get("fixture"):
        parser.error("only the generated numerical fixture is accepted")
    cfg = meta["config"]["text_config"]
    ids = np.asarray(json.loads((args.fixture_dir / "tokens.json").read_text()), dtype=np.int32)
    if args.ubatch < 2 or args.ubatch > len(ids) or not 0 < args.cut < cfg["num_hidden_layers"]:
        parser.error("cut and ubatch must fit the fixture")
    os.environ["TS_DSV41_YOCO_KEEP"] = "0"
    os.environ["TS_DSV41_ENGRAM_WARM"] = "0"
    os.environ["TS_DSV41_ENGRAM_DEVICE"] = "0"
    os.environ.pop("TS_DSV41_EXPERT_CACHE_MB",None)
    os.environ.pop("TS_DSV41_CPU_MOE_PREFETCH",None)
    os.environ.pop("TS_DSV41_EXPERT_CACHE_PREAD",None)
    os.environ.pop("TS_DSV41_EXPERT_HOST_CACHE_MB",None)
    os.environ.pop("TS_DSV41_EXPERT_CACHE_DEVICE_BRIDGE",None)
    lib = C.CDLL(str(args.library.resolve()))
    def bind(name, argv, result=C.c_int):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argv, result
        return fn
    ptr, integer = C.c_void_p, C.c_int
    load = bind("TSGgml_Dsv4LoadModel", [C.c_char_p]+[integer]*4+[C.c_char_p,integer,C.c_char_p,integer], ptr)
    stage_load = bind("TSGgml_Dsv41LoadStage", [C.c_char_p]+[integer]*4+[C.c_char_p,integer,integer], ptr)
    forward = bind("TSGgml_Dsv4Forward", [ptr,ptr,integer,ptr])
    stage_forward = bind("TSGgml_Dsv41StageForward", [ptr,ptr,integer]+[ptr]*5+[integer])
    reset = bind("TSGgml_Dsv4ResetChecked", [ptr])
    truncate = bind("TSGgml_Dsv4Truncate", [ptr,integer])
    head_pos = bind("TSGgml_Dsv4NPast", [ptr])
    free = bind("TSGgml_Dsv4Free", [ptr], None)
    checkpoint = args.model or args.fixture_dir / "deepseek41-fixture.gguf"
    reference_model = args.reference_model or checkpoint
    encoded = str(checkpoint.resolve()).encode()
    common = [encoded,args.gpus,256,args.ubatch,2]
    handles = []
    checks = []
    worst = 0.0
    def check(ok, label, **values):
        checks.append({"check": label, "passed": bool(ok), **values})
        if not ok:
            raise AssertionError(label)
    def pointer(array):
        return None if array is None else array.ctypes.data
    def run_stage(handle, tokens, hi=None, pi=None, ho=None, po=None, logits=None, all_logits=False):
        return stage_forward(handle,pointer(tokens),len(tokens),pointer(hi),pointer(pi),pointer(ho),pointer(po),
                             pointer(logits),int(all_logits))
    report = {"library_sha256":hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "fixture_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "reference_fixture_sha256":hashlib.sha256(reference_model.read_bytes()).hexdigest(),
              "backend":args.backend,"gpus_per_handle":args.gpus,"cut":args.cut,"ubatch":args.ubatch,
              "expert_cache_mb_per_gpu":args.cache_mb,
              "cpu_expert_prefetch":args.cpu_prefetch,
              "coverage":"synthetic local stage graph and cache equivalence; transport and full model untested",
              "checks":checks}
    ggml = Path(__file__).resolve().parents[2] / "ExternalProjects" / "ggml"
    try:
        report["ggml_source_revision"] = subprocess.check_output(
            ["git","-C",str(ggml),"rev-parse","HEAD"],text=True).strip()
        report["ggml_source_status"] = subprocess.check_output(
            ["git","-C",str(ggml),"status","--short"],text=True).strip()
        if report["ggml_source_status"]:
            raise RuntimeError("ggml source checkout is modified")
    except subprocess.CalledProcessError:
        report["ggml_source_revision"] = "unavailable; use external build manifest"
    try:
        reference_common = [str(reference_model.resolve()).encode(),*common[1:]]
        full = load(*reference_common,None,0,args.backend.encode(),0); handles.append(full)
        if args.expert_pread: os.environ["TS_DSV41_EXPERT_CACHE_PREAD"]="1"
        report["expert_cache_pread"] = args.expert_pread
        if args.host_cache_mb: os.environ["TS_DSV41_EXPERT_HOST_CACHE_MB"]=str(args.host_cache_mb)
        if args.device_bridge: os.environ["TS_DSV41_EXPERT_CACHE_DEVICE_BRIDGE"]="1"
        report["expert_host_cache_mb"] = args.host_cache_mb
        report["expert_device_bridge"] = args.device_bridge
        if args.cpu_prefetch:os.environ["TS_DSV41_CPU_MOE_PREFETCH"]="1"
        if args.cache_mb:
            if args.cache_mb<1:raise ValueError("cache MiB must be positive")
            os.environ["TS_DSV41_EXPERT_CACHE_MB"]=str(args.cache_mb)
            os.environ["TS_DSV41_EXPERT_CACHE_DIAGNOSTICS"]="1"
        prefix = stage_load(*common,args.backend.encode(),0,args.cut); handles.append(prefix)
        suffix = stage_load(*common,args.backend.encode(),args.cut,cfg["num_hidden_layers"]); handles.append(suffix)
        check(all(handles), "all three handles load")
        bad = stage_load(*common,args.backend.encode(),2,cfg["num_hidden_layers"])
        if bad: free(bad)
        check(not bad, "cut crossing shared-cache source is refused")
        vocab, hidden, hc = cfg["vocab_size"], cfg["hidden_size"], cfg["hc_mult"]
        def compare(tokens, label):
            nonlocal worst
            tokens = np.ascontiguousarray(tokens)
            expected, actual = np.empty(vocab,np.float32), np.empty(vocab,np.float32)
            h = np.empty((len(tokens),hc,hidden),np.float32)
            p = np.empty((len(tokens),hc),np.float32)
            check(forward(full,pointer(tokens),len(tokens),pointer(expected)) == 0, label+" full forward")
            check(run_stage(prefix,tokens,ho=h,po=p) == 0, label+" prefix forward")
            check(np.isfinite(h).all() and np.isfinite(p).all(), label+" boundary finite")
            check(run_stage(suffix,tokens,hi=h,pi=p,logits=actual) == 0, label+" suffix forward")
            error = float(np.max(np.abs(expected-actual)))
            worst = max(worst,error)
            check(np.allclose(expected,actual,atol=2e-5,rtol=2e-5),label+" all vocabulary logits match",max_abs_error=error)
            check(head_pos(full)==head_pos(prefix)==head_pos(suffix),label+" cache heads match")
            return h,p,actual
        for prompt in (1,3,args.ubatch,min(len(ids)-4,args.ubatch+3)):
            for handle in handles: check(reset(handle)==1,"reset for prompt"+str(prompt))
            for start in range(0,prompt,args.ubatch): compare(ids[start:min(start+args.ubatch,prompt)],f"prefill{prompt}:{start}")
            for start in range(prompt,min(len(ids),prompt+10)): compare(ids[start:start+1],f"decode{prompt}:{start}")
        # Identical paired rewind; reset/re-prefill is the fallback when rings
        # cannot retain the requested position. No successful rewind is inferred.
        target = max(2,(head_pos(full)-2)//2*2)
        rewinds = [truncate(handle,target) for handle in handles]
        check(len(set(rewinds))==1,"truncate result agrees",results=rewinds)
        if rewinds[0]: compare(ids[target:target+1],"post-truncate")
        for handle in handles: check(reset(handle)==1,"reset before all-logits")
        chunk = np.ascontiguousarray(ids[:args.ubatch])
        h = np.empty((len(chunk),hc,hidden),np.float32); p = np.empty((len(chunk),hc),np.float32)
        rows = np.empty((len(chunk),vocab),np.float32)
        check(run_stage(prefix,chunk,ho=h,po=p)==0,"all-logits prefix")
        check(run_stage(suffix,chunk,hi=h,pi=p,logits=rows,all_logits=True)==0,"all-logits suffix")
        for i in range(len(chunk)):
            actual = np.empty(vocab,np.float32)
            one = np.ascontiguousarray(chunk[i:i+1])
            check(forward(full,pointer(one),1,pointer(actual))==0,f"all-logits unsplit row{i}")
            error = float(np.max(np.abs(rows[i]-actual))); worst=max(worst,error)
            check(np.allclose(rows[i],actual,atol=2e-5,rtol=2e-5),f"all-logits row{i} matches",max_abs_error=error)
        position = head_pos(suffix)
        check(run_stage(suffix,ids[:1],logits=np.empty(vocab,np.float32))<0,"missing inputs refused")
        check(head_pos(suffix)==position,"invalid call preserves cache head")
        if args.cache_mb:
            stat = bind("TSGgml_HostMoeExpertCacheStats",[C.POINTER(C.c_int64)]*5)
            values = [C.c_int64() for _ in range(5)]
            check(stat(*(C.byref(value) for value in values))==1,"cache statistics available")
            report["expert_cache"] = dict(zip(("reserved","budget","hits","misses","calls"),(value.value for value in values)))
            check(values[0].value<=values[1].value,"expert cache stays within aggregate reservation")
            if args.expect_cache_activity:
                check(values[4].value>0 and values[2].value>0,"actual cached GPU graphs and warm hits observed")
        report["status"]="passed"
    except Exception as error:
        report["status"]="failed"; report["error"]=str(error)
        raise
    finally:
        for handle in reversed(handles):
            if handle: free(handle)
        report["max_abs_error"]=worst
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks":len(checks),"max_abs_error":worst}))


if __name__ == "__main__":
    main()
