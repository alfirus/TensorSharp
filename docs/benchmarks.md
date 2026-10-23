# Benchmarks

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> Part of the [TensorSharp](../README.md) documentation. The README shows the headline table in [Benchmarks](../README.md#benchmarks).

## Historical laptop comparison vs llama.cpp

This historical snapshot records **TensorSharp `8a13852`**, .NET 10.0.204 and `llama.cpp` on **identical GGUF files, the same NVIDIA RTX 3080 Laptop GPU (16 GB), and one uniform OpenAI `/v1/chat/completions` surface** — with **both engines measured on their GGML CUDA and Vulkan builds**. Numbers are the **geomean speedup of TensorSharp over llama.cpp on the same backend** across `text_short`, `text_long` and `multi_turn` (single-stream, greedy, MTP off); **> 1.0× means TensorSharp is faster / lower-latency**. Full per-scenario tables: [archived report at `00c17e2f`](https://github.com/zhongkaifu/TensorSharp/blob/00c17e2fbb0882114301ad4589fa145dbada3753/docs/engine_comparison_report.md).

**Evidence limits.** The archived report does not record the llama.cpp revision, GGUF hashes, repeat counts or run-to-run spread, and the raw per-cell results are not committed. It documents that snapshot, not current-main performance or a fully specified reproduction. Prefill throughput is `prompt_tokens / streamed TTFT`, an HTTP latency proxy rather than an isolated kernel measurement. The [generated harness report](engine_comparison_report.md) is overwritten by later runs. Its [replacement snapshot at `7e720d79`](https://github.com/zhongkaifu/TensorSharp/blob/7e720d79d259ba4a6d2f0a57b172d2579a078250/docs/engine_comparison_report.md) records TensorSharp `a8d8f40` with only GGML CUDA and Direct CUDA results; it has no overlapping reference-engine cells and does not substantiate the table below.

| Model | Backend | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it (Q8_0, dense multimodal) | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | Vulkan | 1.02× | 0.85× | 0.84× |

Results vary by model and backend, including slower cells: Qwen 3.6 35B-A3B decode is 0.98× on CUDA and 0.87× on Vulkan, while Qwen 3.6 27B prefill is 0.96× on CUDA and 0.85× on Vulkan. The [`engine-comparison harness`](../benchmarks/engine_comparison) also supports tool-calling, structured-output, MTP on/off and parallel-request scenarios; this table covers only the three text scenarios above. Consult the archived report for the snapshot's actual coverage and output-quality comparisons.

Models too large for that 16 GB rig carry their own head-to-head in their card, measured the same way (both engines, same GGUF, same machine, back to back): [GLM-5.2 744B-A40B on 3x RTX PRO 6000](models/glm.md#performance) — TensorSharp leads prefill from ~1k prompt tokens up (pp2048 **1.20×**, pp4096 **1.21×**) and decode by 1.04×, with llama.cpp a few percent ahead on short prefills. The non-Flash [GLM-5.3](models/glm.md#glm-53-glm-dsa) has its own, on 8× A40 46 GB without NVLink (UD-Q2_K_XL, 10,531-token prompt, 300 decode tokens, median of 3, whole-layer placement): decode is a tie at **20.48** tok/s against llama.cpp's 20.28, TensorSharp prefills at 251.6 tok/s and loads the 236.4 GiB checkpoint **2.9× faster** (264 s against 753 s), and the honest gap is time to first token — 41.9 s against 29.0 s, about **1.4× slower**. llama.cpp's prefill tok/s was not recorded for that cell. Full method and per-repeat numbers: `docs/validation/cross-engine-2026-09/README.md` (local validation evidence, not committed). llama.cpp is a valid reference engine for `glm-dsa`, but not for `glm5next` (GLM-5.3-Flash).

## Muse Spark 816B on ten A40 GPUs

[Muse Spark](models/muse-spark.md) (Muse-Spark-Open-816B-A42B, UD-IQ3_XXS, 317 GiB in eight shards) is the largest model with a recorded head-to-head, measured in both multi-GPU modes each engine offers. TensorSharp ran its native whole-model executor on `ggml_cuda` with every expert in GPU memory, with `--layer-split 10` against llama.cpp's `-sm layer` and with `--tp 10` against its `-sm tensor`. llama.cpp `7170cad` ran unmodified on the same files and machine (ten A40 46 GB GPUs whose advertised peer-to-peer access does not work), built without NCCL because its NCCL build aborts on ten GPUs; TensorSharp used unmodified upstream ggml `da8e2a5`. Both used F16 KV, flash attention and a 32,768-token context with room for four concurrent requests, with thinking disabled. Decode, prefill and first-token figures are medians of three repeats (some of llama.cpp's slow tensor-split rows, of two), and the fixtures ran once; above 1.00× means TensorSharp is faster or lower-latency.

Layer split (`--layer-split 10` against `-sm layer`):

| Scenario | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| Decode, one request | 16.69 tok/s | 16.79 tok/s | 0.99× |
| Time to first token, short prompt | 0.64 s | 0.75 s | **1.17×** |
| Prefill, 4K / 16K-token prompt | 999.7 / 1346.8 tok/s | 483.2 / 826.1 tok/s | **2.07× / 1.63×** |
| Four concurrent requests, total output | 33.65 tok/s | 33.04 tok/s | **1.02×** |
| Second turn of a 2.3K-token conversation, first token | 0.45 s | 0.91 s | **2.00×** |
| New conversation sharing a 2.3K-token system prompt, first token | 0.48 s | 0.64 s | **1.34×** |
| Load time | 6.0 min | 14.4 min | **2.4×** |
| Answer-checking fixtures (`eng/muse-spark-validate.py`) | 13 / 14 | 12 / 14 | |

Tensor parallelism (`--tp 10` against `-sm tensor`):

| Scenario | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| Decode, one request | 22.63 tok/s | 16.19 tok/s | **1.40×** |
| Time to first token, short prompt | 0.58 s | 1.29 s | **2.22×** |
| Prefill, 4K-token prompt | 294.4 tok/s | 75.9 tok/s | **3.88×** |
| Prefill, 16K-token prompt | 295.2 tok/s | [report](perf/muse-spark-vs-llama-cpp.md) | |
| Four concurrent requests, total output | 45.73 tok/s | [report](perf/muse-spark-vs-llama-cpp.md) | |
| Second turn of a 2.3K-token conversation, first token | 0.46 s | [report](perf/muse-spark-vs-llama-cpp.md) | |
| New conversation sharing a 2.3K-token system prompt, first token | 0.51 s | [report](perf/muse-spark-vs-llama-cpp.md) | |
| Load time | 6.4 min | 14.7 min | **2.3×** |
| Answer-checking fixtures (`eng/muse-spark-validate.py`) | 14 / 14 | [report](perf/muse-spark-vs-llama-cpp.md) | |

Layer split is the mode for long prompts: TensorSharp pipelines one prompt's prefill across the ten GPUs, with micro-batches sized per call, while single-request decode, one token passing through the GPUs in turn, is level. Tensor parallelism is the mode for generation. Both engines split each layer the same way; the difference is the all-reduce, which TensorSharp runs through NCCL over shared memory and llama.cpp's NCCL-free build through its peer-copy butterfly. On teacher-forced wikitext-2, TensorSharp's layer-split next-token distributions differ from llama.cpp's by a mean KL divergence of 0.0026, less than the 0.0028 between two llama.cpp runs that differ only in micro-batch size.

All numbers come from one machine without working P2P or NVLink; with either, tensor-parallel results would be very different for both engines. llama.cpp's tensor split ran without NCCL. Images, speculative decoding, CPU expert offload, contexts above 16K tokens, other GPU counts and Vulkan were not measured. This comparison replaces an earlier six-A40 one, which measured the UD-Q3_K_XL checkpoint with expert stacks offloaded to host memory on an older TensorSharp build. Setup, launch commands, method and every limit: [Muse Spark vs llama.cpp](perf/muse-spark-vs-llama-cpp.md).
