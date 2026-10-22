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

## 十块 A40 上的 Muse Spark 816B

[Muse Spark](models/muse-spark_zh-cn.md)（Muse-Spark-Open-816B-A42B，UD-IQ3_XXS，8 个分片共 317 GiB）是目前有正面对比记录的最大模型，两个引擎各自的两种多 GPU 模式都做了测量。TensorSharp 在 `ggml_cuda` 上运行原生整模型执行器，全部专家都在显存中：以 `--layer-split 10` 对照 llama.cpp 的 `-sm layer`，以 `--tp 10` 对照它的 `-sm tensor`。llama.cpp `7170cad` 未作修改，在同一份文件与同一台机器上运行（十块 A40 46 GB GPU，驱动声明支持 P2P 但实际不可用）；它的 NCCL 构建在十块 GPU 上会中止，因此使用不带 NCCL 的构建。TensorSharp 使用未修改的上游 ggml `da8e2a5`。两者都使用 F16 KV、flash attention 与 32,768 token 上下文（可容纳四个并发请求），并关闭思维链。decode、prefill 与首 token 数字都是三次重复的中位数（llama.cpp 张量并行中较慢的部分场景为两次），校验用例各运行一次；大于 1.00× 表示 TensorSharp 更快或延迟更低。

按层切分（`--layer-split 10` 对照 `-sm layer`）：

| 场景 | TensorSharp | llama.cpp | 比值 |
|---|---:|---:|---:|
| 单请求 decode | 16.69 tok/s | 16.79 tok/s | 0.99× |
| 短提示的首 token 延迟 | 0.64 秒 | 0.75 秒 | **1.17×** |
| 4K / 16K token 提示的 prefill | 999.7 / 1346.8 tok/s | 483.2 / 826.1 tok/s | **2.07× / 1.63×** |
| 四个并发请求的总输出 | 33.65 tok/s | 33.04 tok/s | **1.02×** |
| 2.3K token 对话的第二轮首 token | 0.45 秒 | 0.91 秒 | **2.00×** |
| 共享 2.3K token 系统提示的新对话首 token | 0.48 秒 | 0.64 秒 | **1.34×** |
| 加载时间 | 6.0 分钟 | 14.4 分钟 | **2.4×** |
| 答案校验用例（`eng/muse-spark-validate.py`） | 13 / 14 | 12 / 14 | |

张量并行（`--tp 10` 对照 `-sm tensor`）：

| 场景 | TensorSharp | llama.cpp | 比值 |
|---|---:|---:|---:|
| 单请求 decode | 22.63 tok/s | 16.19 tok/s | **1.40×** |
| 短提示的首 token 延迟 | 0.58 秒 | 1.29 秒 | **2.22×** |
| 4K token 提示的 prefill | 294.4 tok/s | 75.9 tok/s | **3.88×** |
| 16K token 提示的 prefill | 295.2 tok/s | [报告](perf/muse-spark-vs-llama-cpp.md) | |
| 四个并发请求的总输出 | 45.73 tok/s | [报告](perf/muse-spark-vs-llama-cpp.md) | |
| 2.3K token 对话的第二轮首 token | 0.46 秒 | [报告](perf/muse-spark-vs-llama-cpp.md) | |
| 共享 2.3K token 系统提示的新对话首 token | 0.51 秒 | [报告](perf/muse-spark-vs-llama-cpp.md) | |
| 加载时间 | 6.4 分钟 | 14.7 分钟 | **2.3×** |
| 答案校验用例（`eng/muse-spark-validate.py`） | 14 / 14 | [报告](perf/muse-spark-vs-llama-cpp.md) | |

长提示适合按层切分：TensorSharp 把一个提示的 prefill 以按调用确定大小的微批流水线化地分布到十块 GPU 上；单请求 decode 时一个 token 依次经过各 GPU，两者持平。生成适合张量并行：两个引擎对每层的切分方式相同，差别在 all-reduce——TensorSharp 通过共享内存上的 NCCL 完成，llama.cpp 的无 NCCL 构建则用它的 peer-copy butterfly。在 teacher-forced 的 wikitext-2 上，TensorSharp 按层切分的下一 token 分布与 llama.cpp 的平均 KL 散度为 0.0026，小于 llama.cpp 只改变微批大小时两次运行之间的 0.0028。

所有数字都来自同一台没有可用 P2P 或 NVLink 的机器；在有这些互联的机器上，两个引擎的张量并行结果都会很不一样。llama.cpp 的张量并行未使用 NCCL。图像、投机解码、CPU 专家卸载、16K 以上的上下文、其他 GPU 数量与 Vulkan 都未测量。本对比取代了早先的六块 A40 对比，那次测的是专家栈卸载到主机内存的 UD-Q3_K_XL checkpoint，TensorSharp 也是较早的构建。测试环境、启动命令、方法与全部局限见 [Muse Spark 对比 llama.cpp](perf/muse-spark-vs-llama-cpp.md)（英文）。
