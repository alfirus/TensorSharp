# DeepSeek 1017: TensorSharp and Maya on two A40 hosts

This campaign uses published, unchanged Unsloth GGUFs at
`60ad3651641b5c97679c807e064ab1d7a3ca1455`. Generated evidence belongs under
ignored `docs/validation/` or `artifacts/`. Do not commit weights, logs, reports,
or benchmark snapshots.

The experimental TensorSharp pipeline divides the released 40-layer architecture
at layer 20. The two ranges have independent shared-attention caches. Each process
uses its own local layer placement across two GPUs; exact F32 residual streams
and delayed hyperconnection gates cross a private TCP connection. The current
path serves one text sequence, with exact prefill and paired cache resets. It
does not expose speculative decoding, vision, tensor-parallel collectives,
multi-sequence slots, retained prefixes, or cache truncation. Unsupported
configuration is refused. Do not infer those features from a successful text run.

## Prepare and build

On each Linux host, provision the CUDA toolkit matching the driver separately.
`deepseek-maya-bootstrap.sh` installs build tools, a local .NET 10 SDK, and an
isolated Hugging Face Python environment. It does not install a GPU driver.

```bash
bash eng/validation/deepseek-maya-bootstrap.sh
/workspace/hf-env/bin/python eng/validation/download-deepseek1017.py \
  --directory /workspace/models/DeepSeek-V4.1-Flash-1017-GGUF \
  --quant UD-Q4_K_XL --workers 2 \
  --evidence /workspace/validation/q4-model-manifest.json
bash eng/validation/build-deepseek-maya-tensorsharp.sh
```

The downloader verifies every complete file against the publisher's LFS SHA-256.
The manifest becomes `complete` only after all hashes match. The pipeline requires
this manifest, checks the complete numbered shard set and local file sizes/times,
and compares a checkpoint identity covering all shard hashes before inference.
Do not edit a verified checkpoint or its verification manifest.

The build script pins upstream ggml, checks its Git worktree before and after the
build, and records the library hash. Native model behavior belongs in
TensorSharp-owned code. The upstream tree and downloaded GGUFs stay unchanged.
Keep the recorded dependency revision and library/source hashes with results.

## Start the two-node TensorSharp pipeline

Use an authenticated SSH local forward from the head host to the tail host:
`127.0.0.1:6202` on the head forwards to `127.0.0.1:6202` on the tail. The worker
only binds loopback. Use a dedicated forwarding-only key if the hosts do not
already have an appropriate authenticated connection.

On the tail host:

```bash
MODEL=/workspace/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf
export TS_DSV41_YOCO_KEEP=0 TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=0
/workspace/dotnet/dotnet \
  eng/validation/DeepSeekPipelineWorker/bin/Release/net10.0/DeepSeekPipelineWorker.dll \
  --model "$MODEL" --manifest /workspace/validation/q4-model-manifest.json \
  --listen 127.0.0.1:6202 --gpus 2 --context 4096 --ubatch 8 --threads 16
```

On the head host:

```bash
export TS_DSV41_PIPELINE_PEER=127.0.0.1:6202
export TS_DSV41_PIPELINE_MANIFEST=/workspace/validation/q4-model-manifest.json
export MAX_CONTEXT=4096 TS_DSV4_UBATCH=8 TS_DSV4_THREADS=16
export TS_DSV41_YOCO_KEEP=0 TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=0
/workspace/dotnet/dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model "$MODEL" --backend ggml_cuda --layer-split 2 --host 127.0.0.1 --port 5000
```

`--layer-split 2` describes each host's two local GPUs. The pipeline peer selects
the second host. `--tp` and `--tp-peers` remain separate features and cannot select
this DeepSeek pipeline. Preserve the stage placement summaries: host expert
offload and demand paging are capacity/performance limitations, even when all
four GPUs execute the request.

Both hosts must use pipeline protocol v3; older peers are refused during the
version handshake. `TS_DSV41_PIPELINE_PREFILL_OVERLAP=1` on the head opts into a
bounded two-chunk prefill window. Intermediate chunks use opcode 3 and advance
every stage/cache normally, returning only status and position; the final chunk
and ordinary decode return the complete exact F32 vocabulary. Enabling the
same setting on the tail coalesces replies into one write. The default remains
the synchronous path. Record the flag and negotiated microbatch width with each
real two-node measurement; no token or vocabulary approximation is applied.

`TS_DSV41_PIPELINE_GREEDY_TRANSPORT=1` on the head separately opts into exact
single-token greedy replies through `IGreedyForwardModel.TryForwardGreedyToken`.
Opcode 4 still computes the full F32 vocabulary on the tail, verifies every
coordinate is finite and selects the first maximum with strict `>` comparison.
Its reply is one 12-byte status/position/token frame. The runtime uses this
separate API only when plain greedy selection is eligible; prefill, logit
capture and callers needing logits retain full-row forwarding. The flag is off
by default. Record the flag and actual compact prediction count in benchmarks;
`TS_DSV41_PIPELINE_PERF=1` emits `greedy_reply=1` on compact tail replies.

For the measured Q4 A40 profile, use microbatch **8** on both hosts and the
following opt-in settings on both processes:

```bash
export TS_DSV41_EXPERT_CACHE_MB=36864
export TS_DSV41_EXPERT_CACHE_PREAD=1
export TS_DSV41_EXPERT_HOST_CACHE_MB=0
export TS_DSV41_EXPERT_CACHE_DEVICE_BRIDGE=0
export TS_DSV41_PIPELINE_PREFILL_OVERLAP=1
export TS_DSV4_UBATCH=8 TS_DSV4_THREADS=16 TS_CPU_MOE_THREADS=16
```

`EXPERT_CACHE_MB` reserves a separate quantized expert cache on each GPU.
The selected A40 profile leaves the experimental host tier and device bridge
disabled: their combined measured decode rate was lower on these hosts.
They remain separately opt-in and must be measured for another deployment.

With `TS_DSV41_EXPERT_HOST_CACHE_MB=65536`, the process-wide host tier retains
original quantized payloads evicted from
those GPU slots, consumes a host entry when promoting it, and bounds its raw
payload allocation, including exchanges, by `EXPERT_HOST_CACHE_MB`. It adds no
second resident CPU copy of a GPU-cached expert. Both cache tiers release entries
with their owning backend/model. A miss in both tiers reads exact selected
file ranges through persistent descriptors and bounded CUDA-pinned staging
(64 MiB per backend by default). A failed source read or upload invalidates
replacement tags and fails the forward before they can be reused.

With `TS_DSV41_EXPERT_CACHE_DEVICE_BRIDGE=1`, the device bridge keeps expert inputs, routing weights and results on the
owning GPU, downloads selected IDs and finite-status flags, and retains complete
CPU fallback when the GPU helper declines. These flags default to disabled;
they do not narrow arithmetic or skip experts. Spans above 8 use the CPU expert
path, so increasing the microbatch is a separate profile requiring measurement.
The 64 GiB host ceiling is intended for this campaign's approximately 93 GiB
container allowance; it is a ceiling on dynamically populated cache payloads,
not a claim that every model's working set fits. Preserve actual cache counters,
container memory telemetry and cold/warm timings with benchmark results.

The campaign's OpenSSH 9.6p1 transport uses a forwarding key restricted to
`permitopen="127.0.0.1:6202"` and the fixed command
`command="/usr/bin/sleep infinity"`. Start that tunnel with a command session
(`ssh -f -n -T ... keepalive`), redirecting its standard streams to a log or
`/dev/null`. The fixed command lets OpenSSH initialize the server transport's
TCP_NODELAY setting while preventing arbitrary commands through that key.
The normal `-N` forwarding setup and the command-session setup had materially
different latency on these hosts; preserve actual payload-size echo and model
timings. See the pinned upstream
[session initialization](https://github.com/openssh/openssh-portable/blob/V_9_6_P1/session.c#L484)
and [socket configuration](https://github.com/openssh/openssh-portable/blob/V_9_6_P1/packet.c#L2018).

For the HTTP host, use `--no-prefix-cache --no-continuous-batching --no-webui
--no-skills --no-multi-agent` with this experimental single-sequence pipeline.
Built-in delegation is independent of skills and otherwise adds a tool catalog
to the prompt. For plain greedy measurements, omit `tool_choice`, send
`"multi_agent": false`, and verify actual prompt counts and tail
`greedy_reply=1` diagnostics. Explicit `tool_choice: "none"` enforces a
no-tools grammar and correctly uses the full-logit path. Validate server/API
behavior separately from the raw engine measurements.

## Correctness and measurement

Run `validate-deepseek41-pipeline.py` against generated CPU and CUDA fixtures
before running the full checkpoint. It compares complete vocabulary rows with
the unsplit executor, exercises prefill/decode/chunks/reset/rewind, and rejects
unsafe cuts and invalid calls. Synthetic equivalence does not establish model
quality or remote execution.

Build `DeepSeekEngineBench` normally with native rebuilds disabled after the
native library is built. Use `--export-cases` to tokenize the declared text suite
once. Pass that same JSON to both raw engine runners. The TensorSharp runner
asserts cache heads on every call, records actual IDs, and resets before every
case. `run-maya-engine-benchmark.py --reset-between-cases` forces a different
prefix in an unmeasured guard request and asserts zero reused prompt tokens.
Discard all warmups. Record the guard's existence when comparing prefill.

For decode, compare `(generated_tokens - 1) / (last_token_time - first_token_time)`
from both clients. Maya's native `n / decode_time` is a separate metric. Neither
SSE frame counts nor requested output limits are actual token counts. Preserve
failure and timeout rows; an unavailable case is not a pass.

Use `deepseek-maya-http-benchmark.py` for separately measured server/API behavior.
Its README explains model/request identity gates and limitations. Use an untimed
teacher-forced session for full-vocabulary comparisons; `DeepSeekEngineBench`
accepts `forced_ids` in cases and `--capture-logits DIRECTORY`. Capture sessions
are explicitly excluded from performance measurements.

Stock Maya supports local multi-GPU execution; its audited source has no network
transport for one model across these two VMs. Compare local TensorSharp and Maya
with equal resources, and report the four-GPU TensorSharp pipeline separately.
Do not present a four-GPU-versus-two-GPU ratio as equal-hardware superiority.

Unsloth UD-IQ2_XXS is useful as a smaller TensorSharp control but its Q5_K Engram
and Q6_K dense tensors are refused by the audited stock Maya revision. Preserve
that actual refusal and use the documented UD-Q4_K_XL variant for a common-file
comparison. Do not silently change the model's table precision or modify Maya
and label the result stock.
