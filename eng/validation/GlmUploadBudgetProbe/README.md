# GLM uploader ownership probe

This small native GLM fixture exercises the real model loader. It compares an
unbudgeted load with a one-byte host-staging refusal followed by a same-process
successful retry. All 512 logits from 37 prefill tokens and three single-token
steps must be bit-exact. The refusal must leave no staging owner, and successful
loads must record a positive host payload high-water mark and release it.

Set these variables **before process launch**; changing the managed environment
in `Main` is not a reliable way to update an initialized native CRT on Windows:

```text
TS_GLM_LOAD_THREADS=2
TS_GLM_LOAD_CHUNK_MB=1
TS_GLM_MOE_MMAP=0
TS_CPU_MOE_THREADS=2
```

Build the project, then run:

```text
dotnet GlmUploadBudgetProbe.dll CPU <new-output-directory>
dotnet GlmUploadBudgetProbe.dll CUDA <another-new-output-directory>
```

Run each through `run-bounded-probe.py` for a process deadline and durable exit
record. Keep output in ignored `artifacts/` or `docs/validation/`. The report
records the actual native module hash; retain assembly/source identities and
the unchanged ggml dependency revision with the invoking validation manifest.

The fixture is GLM-DSA, not trained GLM 5.3 Flash. The common uploader is under
test, including a refused real native allocation request. This does not cover
full model weight ownership, mmap residency, allocator headers, FILE buffers,
thread stacks, driver staging, multi-GPU TP, semantic quality or throughput.
The portable C++ `glm_upload_buffer_test.cpp` separately injects allocation and
commit failure and exercises real strided file reads.
