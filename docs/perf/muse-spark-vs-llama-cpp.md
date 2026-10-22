# Muse Spark 816B: TensorSharp vs llama.cpp on six A40 GPUs

This report compares TensorSharp's native Muse Spark executor with llama.cpp
on the same checkpoint, machine and requests. Each engine ran alone on the
machine, one after the other. Ratios above 1.00x mean TensorSharp is better:
higher throughput or lower latency.

## Summary

TensorSharp was faster in every measured scenario and passed more of the
answer-checking fixtures (15 of 16, against 13 of 16). The table compares
llama.cpp's main configuration. When llama.cpp's host memory is NUMA-balanced
instead, its single-request decode comes within 3% of TensorSharp's (within
this host's noise), but its prefill falls below half (see
[llama.cpp with NUMA-balanced memory](#llamacpp-with-numa-balanced-memory)).

| Scenario | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| Decode, one request | 11.05 tok/s | 9.92 tok/s | **1.11x** |
| Time to first token, short prompt | 1.11 s | 2.60 s | **2.35x** |
| Prefill, 4K-token prompt | 162 tok/s | 76 tok/s | **2.12x** |
| Prefill, 16K-token prompt | 161 tok/s | 84 tok/s | **1.92x** |
| Four concurrent requests, total output | 19.35 tok/s | 17.66 tok/s | **1.10x** |
| Second turn of a 2.3K-token conversation, first token | 0.86 s | 1.60 s | **1.86x** |
| New conversation sharing a 2.3K-token system prompt, first token | 0.84 s | 1.14 s | **1.36x** |
| Load time | about 5 min | about 21.5 min | **4.3x** |

The largest gains come from four changes in TensorSharp:

- **One graph per micro-batch.** A TensorSharp-owned executor runs the whole
  model as one GGML graph per micro-batch, across all six GPUs and the CPU.
  On an earlier five-GPU campaign, the per-operation path decoded at 0.72x
  llama.cpp.
- **Expert placement.** A placement plan keeps as many gate, up and down
  expert stacks on the GPUs as fit and places each stack separately.
- **NUMA-local host experts.** Host experts sit in memory local to the CPU
  workers that read them, with pinned workers and 2048-token prefill
  micro-batches. Host expert bandwidth rose from about 100 to 165–188 GB/s,
  and TensorSharp's own 4K prefill went from 97 to 162 tok/s.
- **Cache reuse across conversations.** The template preamble and shared
  system prompts are checkpointed once and cloned into new conversations.

## Setup

| | TensorSharp | llama.cpp |
|---|---|---|
| Source | branch `codex/muse-spark-inference` at `c58ed9ed` plus the native executor changes described in [muse-spark.md](../models/muse-spark.md#native-whole-model-executor), uncommitted when measured and since committed as `80c64c19` | `0a1ef309c` (build 11796), unmodified |
| ggml | upstream `a05c9922`, unmodified (`git status` clean) | in-tree |
| Model | `unsloth/Muse-Spark-Open-816B-A42B-GGUF`, UD-Q3_K_XL, 10 shards, 374 GiB | same files |
| Projector | `mmproj-Q8_0.gguf` (2.2 GB) | same file |
| Context | 32,768 tokens; five planned slots (four requests plus the shared-prefix checkpoint) | `-c 32768 -np 4 --kv-unified` |
| KV cache | F16 | F16 |
| Placement | 6 GPUs by layer; expert stacks placed by `TS_MUSE_SPARK_CPU_MOE=auto`: 240.4 GiB on GPUs, 131.8 GiB of experts on the host | llama.cpp's automatic fit; 130 GiB resident in host memory |
| CPU threads | 40 pinned NUMA-aware expert workers | `-t 48` |
| Load | about 5 minutes (302 s) | about 21.5 minutes with `--load-mode none` |

Hardware: six NVIDIA A40 (46 GB, PCIe 4.0 x16, driver 595.91.07, CUDA 12.8)
and two Xeon Gold 6342 sockets (two NUMA nodes). The container had a 280 GiB
memory limit and a 48-CPU quota. The host is shared, so repeated runs vary by
several percent; every number below is the median of three repeats.

Launch commands:

```bash
# TensorSharp
MAX_CONTEXT=32768 TS_MUSE_SPARK_PLAN_SLOTS=5 TS_MUSE_SPARK_CPU_MOE=auto \
TS_MUSE_SPARK_DEV0_RESERVE_MB=3072 TS_CODE_EXEC=0 \
dotnet TensorSharp.Server.Host.dll --model <shard 1> --mmproj mmproj-Q8_0.gguf \
  --backend ggml_cuda --layer-split 6 --n-cpu-moe 21 --no-webui --no-skills --no-multi-agent

# llama.cpp
llama-server -m <shard 1> --mmproj mmproj-Q8_0.gguf --jinja -fa on \
  -c 32768 -np 4 --kv-unified -t 48 --load-mode none \
  --chat-template-file llama-matched-template.jinja
```

The llama.cpp template comes from
`eng/validation/prepare_muse_spark_llama_reference.py`. It preserves the
released template and matches TensorSharp's date and answer-prefix handling.
llama.cpp's default memory-mapped load did not finish inside the 280 GiB
limit, so it used `--load-mode none`.

## Method

[`eng/muse-spark-bench.py`](../../eng/muse-spark-bench.py) sends the same
OpenAI-compatible streaming requests to each server. Requests are greedy with
thinking disabled. Decode rate excludes the first token:
`(completion_tokens - 1) / (last_token_time - first_token_time)`. Prefill rate
is uncached prompt tokens divided by time to first token (TTFT).

| Scenario | What it measures |
|---|---|
| Short prompt | A ~167-token prompt and 256 generated tokens: decode speed |
| Long prompt 4K / 16K | A document with one fact in the middle, then a question about it: prefill speed and the answer |
| 4 concurrent requests | Four short prompts at once, 256 tokens each: aggregate throughput |
| Multi-turn KV reuse | A ~2.3K-token system prompt, then a second turn of the same conversation |
| Shared system prompt | Four new conversations that share one ~2.3K-token system prompt (an agent pattern) |

[`eng/muse-spark-validate.py`](../../eng/muse-spark-validate.py) checks
answers in 16 scenarios: short English and Chinese answers, a fixed
512-token decode, JSON, JSON schema, Unicode JSON, an 8K-token retrieval,
multi-turn memory, tool round trips, typed tool arguments, a dependent agent
workflow, parallel isolation, cache reuse, image layout, image chart OCR and
staggered arrivals.
[`eng/muse-spark-report.py`](../../eng/muse-spark-report.py) builds the tables
below from the raw JSON. The raw files are kept in the ignored
`docs/validation/muse-spark/final-2026-10-22/` directory.

## Performance

Medians of three repeats. TTFT is time to first token.

| Scenario | Metric | TensorSharp | llama.cpp | TensorSharp advantage |
|---|---|---:|---:|---:|
| Short prompt (167 tokens), 256 generated | decode tok/s | 11.05 | 9.92 | **1.11x** |
| | TTFT s | 1.11 | 2.60 | **2.35x** |
| Long prompt (4,075 tokens) | prefill tok/s | 162.1 | 76.4 | **2.12x** |
| | TTFT s | 24.35 | 53.36 | **2.19x** |
| | fact found | 3/3 | 3/3 | |
| Long prompt (15,767 tokens) | prefill tok/s | 160.9 | 84.0 | **1.92x** |
| | TTFT s | 97.99 | 187.67 | **1.92x** |
| | fact found | 3/3 | 3/3 | |
| 4 concurrent requests, 256 tokens each | aggregate tok/s | 19.35 | 17.66 | **1.10x** |
| | per-request decode tok/s | 5.31 | 4.84 | **1.10x** |
| | TTFT s | 3.52 | 5.32 | **1.51x** |
| Multi-turn reuse (2.3K-token prefix) | turn-1 TTFT s | 16.56 | 31.31 | **1.89x** |
| | turn-2 TTFT s | 0.86 | 1.60 | **1.86x** |
| | turn-2 cached tokens | 2,338 | 2,337 | |
| New conversations sharing a 2.3K-token system prompt | first TTFT s | 16.12 | 31.03 | **1.93x** |
| | later TTFT s | 0.84 | 1.14 | **1.36x** |

The four-request scenario shows MoE CPU offload and batching together. Each
decode step reads up to 32 different experts per layer, and about 37% of the
expert bytes live in host memory in both engines.

## Answer quality

Each scenario ran once per engine.

| Scenario | TensorSharp | llama.cpp |
|---|---|---|
| Short answer (English, Chinese) | pass, pass | pass, pass |
| Fixed 512-token decode | **fail**: coherent answer, cut off before it reached collisions | **fail**: stopped after 143 of 512 tokens |
| JSON, JSON schema, Unicode JSON | pass, pass, pass | pass, pass, pass |
| 8K-token retrieval | pass | pass |
| Multi-turn memory | pass | pass |
| Tool round trip, typed tool arguments | pass, pass | pass, pass |
| Dependent agent workflow | pass | pass |
| Parallel request isolation | pass | pass |
| Cache reuse | pass | **fail**: the second turn did not reuse the whole first prompt |
| Image layout, chart reading | pass, pass | pass, pass |
| Staggered arrivals | pass | **fail**: stopped after 431 of 512 tokens |
| **Total** | **15 / 16** | **13 / 16** |

The fixed-decode fixture requires exactly 512 tokens that mention collisions
and the hash-to-index step. TensorSharp's answer is accurate and detailed: it
covers hash functions and `index = h(k) mod m`, but the 512-token budget ends
before collisions. The same fixture passed on an earlier build of this
branch. The difference is numerical: the prompt now resumes from the
127-token template checkpoint and prefills 44 tokens, instead of all 171 in
one batch, so greedy decoding takes a different branch at a near-tie in its
first sentence. Unit tests check that the cloned checkpoint matches an
independent reference decoder. The fixtures check integration, not general
model quality, and their wall times are not equal work: llama.cpp's shorter
decode wall time reflects 143 generated tokens.

## llama.cpp with NUMA-balanced memory

In the main run, llama.cpp's host weights sat almost entirely on one NUMA
node (see the limits below). To check how much that cost it, llama.cpp was
also run with `GGML_CUDA_NO_PINNED=1 --numa distribute`. Its host weights then
use ordinary memory, which the kernel's automatic NUMA balancing migrates
toward the threads that read it: from 27/103 GiB on nodes 0/1 at load to
65/72 GiB after about 40 minutes of requests. The steady-state rows are four
repeats taken after that migration.

| Scenario | TensorSharp | llama.cpp, main run | llama.cpp, balanced memory |
|---|---:|---:|---:|
| Decode, one request (tok/s) | 11.05 | 9.92 | 10.72 (steady state) |
| Four concurrent requests, total (tok/s) | 19.35 | 17.66 | 17.47 (steady state) |
| Four concurrent requests, per request (tok/s) | 5.31 | 4.84 | 5.13 (steady state) |
| Prefill, 4K-token prompt (tok/s) | 162 | 76 | 35 |
| Time to first token, short prompt (s) | 1.11 | 2.60 | 6.1 |

Balanced memory brings llama.cpp's single-request decode within 3% of
TensorSharp's, which is within this host's run-to-run noise. Without pinned
memory, though, every prefill copy to the GPUs comes from pageable memory, and
llama.cpp's prefill fell to less than half its main-run rate. No single
llama.cpp configuration measured here matched TensorSharp on both decode and
prefill.

## Limits of this comparison

- **One machine, shared host.** All numbers come from one VM whose CPUs and
  memory bandwidth are shared with other tenants. In the four-request
  scenario, repeats within one run spanned up to 8% for TensorSharp (15.9–17.2
  tok/s on an earlier build) and up to 17% for llama.cpp (15.1–17.7 tok/s in
  its first run). Medians of three repeats damp this, but no confidence
  intervals are claimed.
- **Two llama.cpp runs.** llama.cpp's short-prompt and four-request rows come
  from a second run on the same server. In the first run, a random
  `Run N-<number>.` prefix on those prompts made the model refuse two of three
  short prompts ("We cannot provide that explanation because it would violate
  our safety policy regarding the requested run ID."). The benchmark now
  prefixes prompts with `Request <n>:`, the same for both engines.
  TensorSharp's rows all come from one run of that version.
- **llama.cpp memory placement.** `--load-mode none` places llama.cpp's host
  weights in CUDA pinned memory. Its single loader thread put 129.7 of
  133 GiB on NUMA node 0, so half of its 48 threads read across sockets.
  TensorSharp places host experts on each worker's node. The container
  forbids memory-policy calls (`numactl --interleave` fails), so llama.cpp
  could not interleave. In its first run right after loading, llama.cpp
  decoded at 7.5 tok/s; the 9.92 tok/s in the table comes from a later run on
  the same server. The section above shows the alternative llama.cpp
  configuration.
- **Prompt cache policy.** Both engines reuse the chat template's fixed
  preamble (126–127 tokens) across requests. llama.cpp also reuses any
  identical earlier prompt. TensorSharp keeps unrelated conversations
  isolated except for public system-prompt and template prefixes, so the
  short and four-request scenarios number each repeat's prompts differently
  to measure a cold prompt on both engines. The long-prompt requests send
  `cache_prompt: false`, which llama.cpp honours. TensorSharp still reused
  the 127-token preamble there: prefill rates count only uncached tokens,
  while the TTFT figures favour TensorSharp by about 3% of a 4K prompt.
- **Occasional checkpoint misses.** The GPU holding the most layers (16) runs
  close to full. In one of the three four-request repeats, and in one of the
  three 16K prompts, TensorSharp had reclaimed the template checkpoint under
  memory pressure and prefilled that request's whole prompt. These runs are
  included in the medians.
- **Not covered.** Tensor parallelism (`--tp`), speculative decoding, heavier
  CPU-offload profiles (more layers on the host than this fit required), and
  contexts above 16K tokens were not measured here. Thinking was disabled for
  every request.

## Robustness fixes found by these runs

- **Crash under memory pressure.** The first 4K prompt after a run of
  concurrent requests killed the server. ggml-cuda's kernel-scratch pool
  could not grow on a full GPU and ggml aborted. The executor now runs one
  full-size prefill at load (12.6 s) so the pool reaches its peak first; the
  same benchmark sequence then completed.
- **Shared checkpoints evicted by stale graph reservations.** One 16K-token
  prefill's 1.2 GiB graph made later concurrent requests evict the
  shared-prompt checkpoint. The admission reserve is now capped, and
  micro-batch halving applies only to the call that needed it.
- **Batched decode on host experts.** When two tokens of a batched step picked
  the same expert, ggml's CPU kernel lost its NUMA locality. A
  TensorSharp-owned kernel keeps the static row ownership and is bit-identical
  to ggml's: 3.5–3.8 ms per layer instead of 3.8–3.9 ms with 4 tokens.
- **Reference-count race in the managed core.** One of four full test runs
  failed in Muse Spark's per-op tensor-parallel loading with "object has
  already been destroyed". `RefCounted.Release` computed its new count from a
  second read of the field, so a concurrent AddRef/Release pair could make it
  store a count one too low. It also decided whether to destroy from a third
  read, so two concurrent final releases could both destroy. Release now
  derives both from its compare-and-swap. A new stress test failed 3 of 3
  times on the old code (in about 20 ms) and passes on the fix. The
  benchmarked server binary predates this fix; no server run hit the race.

## Reproducing

```bash
# Performance (same scenarios against each server)
python3 eng/muse-spark-bench.py --engine tensorsharp --url http://127.0.0.1:8083 \
  --out ts.json --scenarios short,long_4k,long_16k,parallel_4,cache_reuse,shared_prefix \
  --repeats 3 --decode-tokens 256
python3 eng/muse-spark-bench.py --engine llamacpp --url http://127.0.0.1:8082 \
  --out llama.json --scenarios short,long_4k,long_16k,parallel_4,cache_reuse,shared_prefix \
  --repeats 3 --decode-tokens 256

# Answer checks
python3 eng/muse-spark-validate.py --engine tensorsharp --url http://127.0.0.1:8083 \
  --model <served model id> --weights-id unsloth/Muse-Spark-Open-816B-A42B-GGUF/UD-Q3_K_XL \
  --profile <name> --output ts-validate.json --repeats 1 \
  --scenarios short,short_zh,decode,json,json_schema,json_unicode,long_8k,multi_turn,tool_round_trip,tool_string_arguments,agentic,parallel,cache_reuse,image,staggered_requests,image_chart

# Tables
python3 eng/muse-spark-report.py --ts-perf ts.json --llama-perf llama.json \
  --ts-quality ts-validate.json --llama-quality llama-validate.json --out report.md
```

The host-expert microbenchmark
[`eng/validation/muse-spark-cpu-moe-bench.cpp`](../../eng/validation/muse-spark-cpu-moe-bench.cpp)
compares ggml's `mul_mat_id` with the TensorSharp kernel on one layer.
`BENCH_SHARE=k` gives the tokens k shared experts.

## Test coverage of this build

On the VM, against the unchanged ggml checkout:

- Native-executor tests: 7 on the CPU lane and 5 on the CUDA lane, all
  passing. They cover ring wrap and rewinds, batched decode and prefill,
  checkpoint clones, speculative all-row verification, image rows, and the
  host-expert kernel against `mul_mat_id`.
- Shared-prefix rendering tests: 13 passing.
- Broader prefix-cache, chat-pipeline, Muse Spark and reference-count suites
  (711 tests) on the final source. On the CPU lane, two runs each passed 636
  and skipped 75. On the CUDA lane, 667 passed and 44 were skipped. None
  failed. An earlier run of the same suites, before the reference-count fix,
  had the one intermittent failure described above. Skipped tests need
  another GGML backend, more GPUs or model fixtures, and are not counted as
  passes.
