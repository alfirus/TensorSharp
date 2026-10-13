# 个人多模态知识库

[English](knowledge-system.md) | [中文](knowledge-system_zh-cn.md)

`TensorSharp.Knowledge` 保存原文件，以及可搜索、可引用的内容表示。
TensorAgent 提供集合管理、上传、已保存聊天导入、搜索和逐聊天选择。
独立 TensorSharp 服务端与智能体工具使用同一个存储和检索协议。
这些能力在 `v2026.10.10` 之后加入；请使用包含相关改动的构建。

默认使用本地 BM25 文本检索，不需要嵌入模型。可选 GGUF 嵌入提供语义检索；
已加载的兼容媒体模型可从图片、音频和视频采样帧提取描述或转录。
TensorAgent 在 Windows/Linux 核心宿主上按需准备本地 PDF OCR 工具；
聊天与嵌入模型仍由用户显式选择。设计依据见[研究说明](knowledge-system-research.md)。
本页提供中文操作和接口指南；完整实现约束见[英文参考](knowledge-system.md)。

## 在 TensorAgent 中使用

1. 打开**菜单 → 知识库**，命名集合并创建。
2. 浏览集合，选择**上传文件**或**上传文件夹**。文件夹导入包含所有子目录，
   保留相对路径；不同文件夹中的同名文件是不同来源。应用使用系统文件夹选择器，
   浏览器使用目录选择器。清理聊天或上传缓存不会清除知识库中的原文件。
3. 使用**添加聊天**搜索已保存聊天，跨页选择最多 32 个，再执行**添加并索引所选聊天**。
   导入包括用户与助手的公开消息、可用附件和生成文件，不包含私有推理。
   缺失文件会明确报告；未完整导入的聊天保留选择以便重试。
   再次导入会更新稳定路径上的快照，不重复保存未变化的来源。
   原聊天后续变化不会自动同步到集合。
4. 检查提取状态、警告和每个文件的失败原因。利用文件名/路径、类型和排序筛选，
   分页浏览资料；重命名集合不会改变它的身份或聊天中的选择。
   搜索后可读取周边内容，打开来源预览或下载原文件。
   PDF 预览先显示已索引文本；原生应用通过系统文件查看器展示原件。
5. 显式勾选**在此聊天中使用该集合**，或在集合目录中选择对应复选框。
   浏览、导入或重新索引不会自动授权聊天使用该集合。
   输入框上方显示已选集合；最多选择 32 个。
   新聊天默认不选集合，重新打开保存的聊天会恢复其选择。
6. 提问或执行智能体任务。宿主先检索相关证据；支持工具的模型还能查找来源、
   调整搜索、展开上下文，并将完整原文件带入私有工作区做分析。

**检索到的来源**只显示最终回答实际引用的资料；未使用的命中或相邻片段不会列出。
回答中的来源链接与来源列表打开相同预览，并在可用时定位到引用片段。
引用身份由宿主核验，但这不保证回答中的每句话都得到证据支持。
删除或过期来源会报错，不会悄悄打开新版本。
图片和视频来源可在检索结果、展开内容和引用中显示缩略图；
受支持的视频预览跳转到采样时间。音频与无法预览的类型显示媒体图标。

文件夹上传分批处理，显示进度和逐文件错误。原生导入报告符号链接，
不会跟随链接离开所选目录；空文件夹没有可索引内容。

## 提取范围与来源位置

| 输入 | 可搜索内容与位置 | 限制 |
| --- | --- | --- |
| 文本、Markdown、代码、日志、XML、YAML | 文本段、行范围及可用的 Markdown 章节 | 严格解码并识别 BOM；拒绝二进制数据 |
| CSV/TSV | 带表头的记录与行号，支持带引号的多行字段 | 可搜索记录，不是数据库查询引擎 |
| JSON 数组、JSONL/NDJSON | 数组条目或逐行记录、行号和条目标签 | 有界解析；其他 JSON 值按文本处理 |
| HTML | 去掉 script/style/noscript 的文本 | 不执行脚本，不渲染或加载远程资源 |
| SRT/VTT | 字幕及起止时间 | 仅索引提供的字幕 |
| PDF | 可选择文本和扫描/含图片页面的整页 OCR、原页码 | OCR 可能误读，不还原精确布局 |
| DOCX | 段落、表格、被引用的注释 | 部分提取；没有渲染分页、嵌入图片或页面布局 |
| PPTX | 按演示顺序提取幻灯片文本 | 部分提取；不解释图片/图表、备注或布局 |
| XLSX | 工作表行、单元格坐标、缓存值 | 部分提取；不重算公式，不解释图表/图片 |
| 图片 | 兼容视觉模型及投影器生成的描述与可见文字 | 部分提取；可能遗漏/误读，没有原生图像嵌入 |
| 音频 | 兼容音频塔或显式转录伴随服务生成的转录 | 部分提取；无说话人分离、逐词时间或非语音事件 |
| 视频 | 最多八个采样帧的描述及采样时间 | 部分提取；不索引声音，不覆盖帧间事件；可变帧率时间近似 |
| 不支持的格式或缺少媒体能力 | 原文件与可搜索文件名 | 仅元数据；没有理解文件内容 |

`Ready` 表示提取器产生了内容且没有声明覆盖警告；`Partial` 表示存在提取限制；
`MetadataOnly` 只索引文件名。生成的描述和转录始终属于部分提取。
文档记录包含提取器身份、警告、内容哈希、版本，以及适用时的嵌入身份。
加载兼容媒体模型或改变嵌入模型后，可对保留的原件执行**重新索引**。

支持的媒体 MIME 类型也可识别没有已知扩展名的上传。解码器读取带已知后缀的
私有临时副本，原文件名和字节保持不变；扩展名本身不能证明编解码器或模型支持。
图片限 16,777,216 像素。本地音频支持最长 120 秒 PCM WAV；其他格式需要
兼容的 `TS_JEV_TRANSCRIPTION_URL` 服务或先转换。视频需要兼容编解码器和视觉模型，
限十分钟、36,000 帧，每帧不超过 16,777,216 像素。

## PDF OCR

提取保留 PDF 文本层，并对含栅格图片的页面执行整页 OCR，避免页眉掩盖扫描正文。
渲染包含完整裁切框与页面旋转；OCR 结果保留原一基页码并标为 `ocr`。
默认最多处理 128 页，每页渲染不超过 16,777,216 像素；Apple 渲染上限 200 DPI，
Poppler/Tesseract 请求 300 DPI 并按像素预算降低。每页 OCR 有 60 秒取消期限。
未识别出文本、缺少 OCR、超时或资源超限均在警告中报告。

| 宿主 | 默认 OCR | 要求与验证边界 |
| --- | --- | --- |
| macOS 13+、iOS 16+、Mac Catalyst 16+ | CoreGraphics + Apple Vision | 系统框架；iOS 不启动桌面工具。TensorAgent 应用本身要求 iOS 17+ |
| TensorAgent Windows x64 / Windows 11 ARM64 | 应用管理的 Poppler + Tesseract | 首次按需下载；ARM64 使用 Windows 11 x64 仿真，运行验证需要 Windows 设备或 runner |
| TensorAgent 核心宿主 Linux x64 / ARM64 | 应用管理的 Poppler + Tesseract | glibc；没有 Linux MAUI 应用头，musl 需要宿主提供后端 |
| 独立 Windows/Linux 宿主 | Poppler + Tesseract | 配置本地工具，或显式传入 `PdfOcrProvisioner` 启用托管安装 |
| 浏览器 UI | 服务端宿主的提取器 | OCR 在宿主运行，不在浏览器 JavaScript 中运行 |
| Android / 浏览器 WebAssembly SDK 宿主 | 没有打包的提取器 | 本仓库没有对应应用头；宿主可实现 `IKnowledgePdfPageExtractor` |

TensorAgent 的默认托管安装下载固定版本的 Poppler、Tesseract、依赖及英语、
简体/繁体中文和中文竖排语言包，验证文件大小和 SHA-256。
首次设置需要网络；之后复用本地安装离线识别，文档字节不会因本地 OCR 离开宿主。
纯文本 PDF 不触发下载。设置失败时不提交导入或重索引，保留已有文档与引用。
进度可在 UI 中查看，失败后可以重试。
工具、字体和语言数据各有上游许可证；维护与离线安装说明见 [OCR 工具指南](../eng/ocr/README.md)。

## 存储、检索与引用

原文件按 SHA-256 保存，文档清单引用不可变字节；启动时重建内存索引。
存储使用单写入者锁，是嵌入式个人存储，不是分布式或多写入者服务。
同一集合内的文件名（文件夹导入包含相对路径）标识文档。
同名且同字节的导入是幂等的；字节变化或重索引递增版本。
提取或嵌入失败时保留此前已提交版本。
只保留当前版本；旧引用读取返回过期错误。

片段引用形式为 `kb://{collectionId}/{documentId}@{version}/{chunkIndex}`，
保留位置、模态、文档和版本。周边读取只展开同一文档与版本。
宿主为实际交付的证据发放 `[K1]` 等引用标记，并在集合和版本仍有效时解析为规范链接。
未知、未发放或过期标记不会产生来源链接。

集合筛选先于排序。BM25 使用倒排索引和 Unicode/CJK 词项；可选嵌入使用
匹配模型身份与维度的归一化余弦检索，再用倒数排名融合合并文本与语义结果。
当前没有近似最近邻索引。宽泛问题可减少重复页面，明确的图片/音频/视频查找优先返回
具有匹配查询锚点的对应媒体原件；PDF OCR 不会因此被当作图片来源。
分数是相对排序信号，不能跨查询比较，也不是相关性概率。

加载聊天模型时，宿主默认可对前八个候选执行一次有界相关性排序；
`TS_KNOWLEDGE_RERANK=0` 关闭。少于三个候选或已批准完整文档快照时跳过。
每个请求树（包括子智能体）最多一次；超时、无效输出或模型不可用时保留原检索顺序。
排序不能发现候选池之外的证据，也不保证答案质量。
改变嵌入模型或前缀不会混用向量空间；不兼容文档仍可文本检索，语义检索需重新索引。

默认限额：单文件 64 MiB、原件逻辑存储 2 GiB、100 个集合、10,000 个文档，
每文档 2,000,000 个提取字符与 10,000 个片段，全库最多 100,000 个片段。
片段为 1,200 字符、重叠 120 字符。SDK 可用 `KnowledgeOptions` 调整限额。
索引和临时文件还需要额外磁盘与内存。

## 推理与智能体工具

支持范围包括 TensorAgent 与已配置独立服务端的 `POST /api/chat`，以及独立服务端的
`POST /v1/chat/completions`、`POST /v1/responses` 和 `POST /api/chat/ollama`。
`POST /api/generate` 支持检索和引用，但没有智能体工具循环。
Ollama 聊天路由为 `/api/chat/ollama`；`/api/chat` 是 Web UI 路由。
这些能力不会自动添加到每个 `/v1` 端点，CLI 也没有知识集合选择选项。

支持的推理请求使用 `knowledgeCollections` 显式授予本次请求的检索范围。
未提供或为空则不提供知识；无效或已删除集合会被拒绝。
每轮重新检索；自动插入的片段不会作为聊天历史保存。
旧助手回答仍可能复述先前资料，保存的来源元数据本身不会授权访问未选择的集合。

| 工具 | 功能 |
| --- | --- |
| `knowledge_list` | 分页发现已选来源，每页最多 20 条 |
| `knowledge_search` | 在所选范围内检索有界、可引用的证据 |
| `knowledge_read` | 展开准确引用，两侧各零至两个相邻片段 |
| `knowledge_open` | 获取带版本的原文件链接；有代码工作区时复制完整原件供分析 |

工具不能修改知识库或扩大范围，子智能体继承相同范围。
完整数据计算应读取原件，不能把排名靠前的几行当作整个数据集。
普通探索在连续三次无新证据、16 次知识调用或累计证据预算达到限制时结束，
之后使用有界证据快照回答；代码与产物工具仍可用于任务。
若单一来源满足大小限制及实际模型 tokenizer 上下文检查，宿主可提供完整的
已索引文本快照；这不证明原件提取完整或回答正确。具体预算见[英文参考](knowledge-system.md#inference-and-agent-behavior)。

OpenAI/Ollama 非流式响应提供 `knowledge_sources`、`knowledge_citation_repairs` 和
有界 `knowledge_tool_steps`。流式元数据位于最终 Chat Completions chunk、
`response.completed.response` 或 Ollama 的 `done: true` 行。
选中知识后，兼容适配器会等待最终文本完成引用规范化，再发送只追加的答案增量；
因此首个答案增量可能更晚，推理或工具活动可以更早到达。
结构化输出保持其内容，规范来源另放在元数据中；`tool_choice: "none"` 禁用工具但保留检索。

## 宿主设置与 HTTP API

TensorAgent 在 `AgentPaths.DataRoot/knowledge` 创建资料库，通过已认证的 loopback
服务管理。独立服务端需设置 `TS_KNOWLEDGE_DIRECTORY`；其浏览器聊天没有
TensorAgent 的知识管理菜单，使用 `/api/knowledge` 管理集合。
集合 ID 是范围标识，不是访问凭证；远程部署需由应用负责认证和授权。
存储不提供多用户权限模型。

| 环境变量 | 用途 |
| --- | --- |
| `TS_KNOWLEDGE_DIRECTORY` | 独立服务端资料库目录；不设置则不启用 |
| `TS_KNOWLEDGE_EMBEDDING_MODEL` | 可选句向量 GGUF；不设置则仅使用文本检索 |
| `TS_KNOWLEDGE_EMBEDDING_BACKEND` | 编码器后端，默认 `CPU` |
| `TS_KNOWLEDGE_QUERY_PREFIX` / `TS_KNOWLEDGE_DOCUMENT_PREFIX` | 模型要求的查询/文档前缀，默认空 |
| `TS_KNOWLEDGE_RERANK` | `0` 关闭有界模型相关性排序 |
| `TS_KNOWLEDGE_PDF_OCR` | `auto`、`apple`、`tesseract` 或 `none`；`none` 保留可选择文本 |
| `TS_KNOWLEDGE_PDFTOPPM_PATH` / `TS_KNOWLEDGE_TESSERACT_PATH` | 自定义工具绝对路径 |
| `TS_KNOWLEDGE_TESSDATA_DIRECTORY` | 自定义 Tesseract 语言数据目录 |
| `TS_KNOWLEDGE_OCR_LANGUAGES` | 托管默认 `eng+chi_sim`；繁体可用 `eng+chi_tra` |

显式设置工具路径或语言数据目录会退出托管安装。
更改提取配置后重启宿主并重新索引已有来源。
独立宿主默认不自动创建 TensorAgent 的托管安装器；可提供本地工具，
或在 SDK 中将 `PdfOcrProvisioner` 传给 `KnowledgeRuntime.CreateStore`。

以下路径相对于 `/api/knowledge`：

| 方法与路径 | 请求或响应 |
| --- | --- |
| `GET /collections` | `{ collections: [...] }` |
| `POST /collections` | `{ name, description? }` → `{ collection }` |
| `DELETE /collections/{collection}` | 删除集合与其来源 |
| `GET /collections/{collection}/documents?offset=0&limit=100` | `{ documents, total, nextOffset }`；每页最多 200 条 |
| `POST /collections/{collection}/documents` | Multipart 上传 → `{ documents }` |
| `DELETE /collections/{collection}/documents/{document}` | 删除来源 |
| `POST /collections/{collection}/documents/{document}/reindex` | 新版本 `{ document }` |
| `GET /collections/{collection}/documents/{document}/source?version=1` | 原件下载，版本过期返回 409 |
| `POST /search` | `{ query, collectionIds, limit?, maxCharacters? }` → `{ results }` |
| `POST /read` | `{ reference, collectionIds, adjacentChunks?, maxCharacters? }` → `{ results }` |

独立端点每请求一个文件。TensorAgent 每批最多 16 个文件及 128 MiB，单文件限制仍适用，
并返回逐文件 `errors`。搜索/读取用 `collectionIds`，推理用 `knowledgeCollections`。
普通格式错误返回 400，来源不存在 404，版本过期 409，限额超出 413。

TensorAgent 另有重命名、带版本的来源预览、安全媒体预览、文档筛选，以及
`GET /chats?q=&offset=0&limit=100` 和
`POST /collections/{collection}/chats`（`{ conversationIds: [...] }`，1–32 个 ID）。
聊天导入分别返回 `documents`、`importedChats` 和 `errors`。
这些扩展不是独立服务端端点；完整字段见[英文接口参考](knowledge-system.md#host-setup-and-http-api)。
可运行的集合创建、上传和推理示例见[服务端 API 示例](../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md)。

## 嵌入 .NET 应用与验证边界

核心 `KnowledgeStore` 不依赖 TensorAgent 或运行中的推理模型。
应用可提供 `IKnowledgeExtractor`、`IKnowledgeEmbeddingProvider` 和
`IKnowledgePdfPageExtractor`，并负责调用者权限。
核心提取器不隐式加载平台 OCR；宿主层负责接入。示例见[英文 SDK 指南](knowledge-system.md#embedding-in-a-net-application)。

可复用测试位于 `TensorSharp.Knowledge.Tests`、`InferenceWeb.Tests` 和
`TensorAgent/tests/TensorAgent.Tests`，覆盖提取、位置、持久化、版本、范围、工具、
传输、聊天来源元数据与 UI 流程。实时模型测试需要显式模型配置，缺失时必须报告未运行。
[浏览器布局探针](../eng/validation/tensoragent-knowledge-layout.py)使用真实页面与 API fixture，
不证明原生键盘、MAUI 设备或模型推理已验证。

[知识验证指南](../eng/validation/knowledge/README.md)与
[基准工具](../eng/validation/KnowledgeBench/README.md)区分文本检索、真实嵌入、
HTTP 协议、完整 CSV 工具计算、Apple OCR 和 Poppler/Tesseract 测量。
Windows/Linux 工作流的配置不代表它已经运行；未执行、依赖缺失或设备不可用均不算通过。
所有生成日志、报告与截图保存在忽略的 `docs/validation/` 或 `artifacts/`，不加入 Git。
