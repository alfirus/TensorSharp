# Benchmarks

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> Part of the [TensorSharp](../README.md) documentation. The README shows the headline table in [Benchmarks](../README.md#benchmarks).

## Head-to-head vs llama.cpp (engine comparison)

TensorSharp’s .NET runtime and native GGML execution are compared with `llama.cpp` on **identical GGUF files, the same NVIDIA RTX 3080 Laptop GPU (16 GB), and one uniform OpenAI `/v1/chat/completions` surface** — with **both engines measured on their GGML CUDA and Vulkan builds**. Numbers are the **geomean speedup of TensorSharp over llama.cpp on the same backend** (single-stream, greedy, MTP off); **> 1.0× means TensorSharp is faster / lower-latency**. Full per-scenario tables: [`docs/engine_comparison_report.md`](engine_comparison_report.md).

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

TensorSharp pulls clearly ahead on CUDA prefill / first-token latency (multi-turn prefill wins on **every** model, up to **1.49×**), holds decode parity-or-better on CUDA, and wins Vulkan decode on the dense 12B (up to **1.32×** on long context) — even at 2-bit IQ2_XXS quantization. The remaining sub-1.0× cells are active optimization targets. The harness also covers tool-calling, structured-output, MTP on/off, and parallel-request scenarios you can run yourself via [`benchmarks/engine_comparison`](../benchmarks/engine_comparison). Every cell is in the [full report](engine_comparison_report.md).

Models too large for that 16 GB rig carry their own head-to-head in their card, measured the same way (both engines, same GGUF, same machine, back to back): [GLM-5.2 744B-A40B on 3x RTX PRO 6000](models/glm.md#performance) — TensorSharp leads prefill from ~1k prompt tokens up (pp2048 **1.20×**, pp4096 **1.21×**) and decode by 1.04×, with llama.cpp a few percent ahead on short prefills. The non-Flash [GLM-5.3](models/glm.md#glm-53-glm-dsa) has its own, on 8× A40 46 GB without NVLink (UD-Q2_K_XL, 10,531-token prompt, 300 decode tokens, median of 3, whole-layer placement): decode is a tie at **20.48** tok/s against llama.cpp's 20.28, TensorSharp prefills at 251.6 tok/s and loads the 236.4 GiB checkpoint **2.9× faster** (264 s against 753 s), and the honest gap is time to first token — 41.9 s against 29.0 s, about **1.4× slower**. llama.cpp's prefill tok/s was not recorded for that cell. Full method and per-repeat numbers: `docs/validation/cross-engine-2026-09/README.md` (local validation evidence, not committed). llama.cpp is a valid reference engine for `glm-dsa`, but not for `glm5next` (GLM-5.3-Flash).

## Muse Spark 816B on six A40 GPUs

[Muse Spark](models/muse-spark.md) (Muse-Spark-Open-816B-A42B, UD-Q3_K_XL, 374 GiB in ten shards) is the largest model with a recorded head-to-head. TensorSharp ran its native whole-model executor on `ggml_cuda` with `--layer-split 6`, keeping 240.4 GiB of weights on six A40 46 GB GPUs and 131.8 GiB of expert stacks in NUMA-local host memory. llama.cpp `0a1ef309c` ran unmodified on the same files and machine (two Xeon sockets, 280 GiB container memory limit, 48-CPU quota) with `-c 32768 -np 4 --kv-unified`. Both used F16 KV and a 32,768-token context, with thinking disabled. Each number is the median of three repeats; above 1.00× means TensorSharp is faster.

| Scenario | TensorSharp | llama.cpp | Ratio |
|---|---:|---:|---:|
| Decode, one request | 11.05 tok/s | 9.92 tok/s | **1.11×** |
| Time to first token, short prompt | 1.11 s | 2.60 s | **2.35×** |
| Prefill, 4K / 16K-token prompt | 162 / 161 tok/s | 76 / 84 tok/s | **2.12× / 1.92×** |
| Four concurrent requests, total output | 19.35 tok/s | 17.66 tok/s | **1.10×** |
| Second turn of a 2.3K-token conversation, first token | 0.86 s | 1.60 s | **1.86×** |
| New conversation sharing a 2.3K-token system prompt, first token | 0.84 s | 1.14 s | **1.36×** |
| Load time | about 5 min | about 21.5 min (`--load-mode none`) | **4.3×** |
| Answer-checking fixtures (`eng/muse-spark-validate.py`) | 15 / 16 | 13 / 16 | |

The host is shared, so repeats varied by several percent, and no confidence intervals are claimed. With NUMA-balanced host memory (`GGML_CUDA_NO_PINNED=1 --numa distribute`), llama.cpp's single-request decode came within 3% of TensorSharp (10.72 tok/s), but its prefill fell to 35 tok/s. Tensor parallelism, speculative decoding, heavier CPU-offload profiles and contexts above 16K tokens were not measured. Setup, launch commands and every limit: [Muse Spark vs llama.cpp](perf/muse-spark-vs-llama-cpp.md).
