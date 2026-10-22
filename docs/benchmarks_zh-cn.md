# 性能数据

[English](benchmarks.md) | [中文](benchmarks_zh-cn.md)

> [TensorSharp](../README_zh-cn.md) 文档的一部分。README 的[性能数据](../README_zh-cn.md#性能数据)一节给出了核心表格。

## 对比 llama.cpp 的同台评测（引擎对比）

TensorSharp 的 .NET 运行时与原生 GGML 执行路径对比 `llama.cpp`：**相同的 GGUF 文件、相同的 NVIDIA RTX 3080 Laptop GPU（16 GB）、统一的 OpenAI `/v1/chat/completions` 接口**，**两个引擎均分别在 GGML CUDA 与 Vulkan 构建上测量**。下表为 **在相同后端上，TensorSharp 相对 llama.cpp 的几何平均加速比**（单流、贪心采样、关闭 MTP）；**> 1.0× 表示 TensorSharp 更快 / 延迟更低**。完整表格见 [`docs/engine_comparison_report.md`](engine_comparison_report.md)。

| 模型 | 后端 | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it（Q8_0，dense 多模态） | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it（Q8_0，dense 多模态） | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | Vulkan | 1.02× | 0.85× | 0.84× |

TensorSharp 在 CUDA 的 prefill / 首 token 延迟上明显领先（多轮 prefill **每个模型**都获胜，最高 **1.49×**），CUDA decode 保持持平或更快，Vulkan 上 dense 12B 的 decode 明显胜出（长上下文最高 **1.32×**）——即便在 2-bit IQ2_XXS 量化下亦然。剩余低于 1.0× 的项仍是正在优化的目标。该框架还提供工具调用、结构化输出、MTP 开/关与并发场景，可通过 [`benchmarks/engine_comparison`](../benchmarks/engine_comparison) 在你自己的硬件上运行。完整报告见 [此处](engine_comparison_report.md)。

放不进这台 16 GB 机器的模型，会在各自的卡片里给出同样方式测得的正面对比（两个引擎、同一份 GGUF、同一台机器、背靠背）：[GLM-5.2 744B-A40B，3× RTX PRO 6000](models/glm_zh-cn.md#性能) —— 从约 1k prompt token 起 TensorSharp 的 prefill 领先（pp2048 **1.20×**、pp4096 **1.21×**），decode 领先 1.04×，短 prefill 上则是 llama.cpp 快几个百分点。非 Flash 的 [GLM-5.3](models/glm_zh-cn.md#glm-53glm-dsa) 另有一份自己的对比，测于 8 张 A40 46 GB（无 NVLink，UD-Q2_K_XL，10,531 token 提示，300 个 decode token，3 次取中位数，整层放置）：decode 打平，**20.48** tok/s 对 llama.cpp 的 20.28；TensorSharp 的 prefill 为 251.6 tok/s，加载这份 236.4 GiB 的 checkpoint **快 2.9×**（264 秒对 753 秒）；真正的差距在首 token 延迟——41.9 秒对 29.0 秒，约**慢 1.4×**。该组数据没有记录 llama.cpp 的 prefill tok/s。完整方法与逐次数据见 `docs/validation/cross-engine-2026-09/README.md`（本地验证记录，未提交到 Git）。llama.cpp 可以作为 `glm-dsa` 的参照引擎，但不能作为 `glm5next`（GLM-5.3-Flash）的参照引擎。

## 六块 A40 上的 Muse Spark 816B

[Muse Spark](models/muse-spark_zh-cn.md)（Muse-Spark-Open-816B-A42B，UD-Q3_K_XL，10 个分片共 374 GiB）是目前有正面对比记录的最大模型。TensorSharp 在 `ggml_cuda` 上以 `--layer-split 6` 运行原生整模型执行器，把 240.4 GiB 权重放在六块 A40 46 GB GPU 上，131.8 GiB 专家栈放在 NUMA 本地的主机内存中。llama.cpp `0a1ef309c` 未作修改，在同一份文件与同一台机器上运行（两路 Xeon，容器内存上限 280 GiB，48 核 CPU 配额），参数为 `-c 32768 -np 4 --kv-unified`。两者都使用 F16 KV 与 32,768 token 上下文，并关闭思维链。每个数字都是三次重复的中位数；大于 1.00× 表示 TensorSharp 更快。

| 场景 | TensorSharp | llama.cpp | 比值 |
|---|---:|---:|---:|
| 单请求 decode | 11.05 tok/s | 9.92 tok/s | **1.11×** |
| 短提示的首 token 延迟 | 1.11 秒 | 2.60 秒 | **2.35×** |
| 4K / 16K token 提示的 prefill | 162 / 161 tok/s | 76 / 84 tok/s | **2.12× / 1.92×** |
| 四个并发请求的总输出 | 19.35 tok/s | 17.66 tok/s | **1.10×** |
| 2.3K token 对话的第二轮首 token | 0.86 秒 | 1.60 秒 | **1.86×** |
| 共享 2.3K token 系统提示的新对话首 token | 0.84 秒 | 1.14 秒 | **1.36×** |
| 加载时间 | 约 5 分钟 | 约 21.5 分钟（`--load-mode none`） | **4.3×** |
| 答案校验用例（`eng/muse-spark-validate.py`） | 15 / 16 | 13 / 16 | |

测试主机与其他租户共享，重复测量之间相差几个百分点，因此不给出置信区间。若让 llama.cpp 的主机内存按 NUMA 均衡分布（`GGML_CUDA_NO_PINNED=1 --numa distribute`），它的单请求 decode 与 TensorSharp 相差不到 3%（10.72 tok/s），但 prefill 降到 35 tok/s。张量并行、投机解码、更重的 CPU 卸载配置以及 16K 以上的上下文都未测量。测试环境、启动命令与全部局限见 [Muse Spark 对比 llama.cpp](perf/muse-spark-vs-llama-cpp.md)（英文）。
