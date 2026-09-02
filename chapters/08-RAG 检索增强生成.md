## 8. RAG 检索增强生成

### 8.0 本章知识地图

| 节号 | 主题 | 面试权重 |
|---|---|---|
| 8.1 | RAG 全景：动机、范式演进、与微调选型 | ★★★★ |
| 8.2 | 文档解析：PDF/Word/扫描件/表格的坑 | ★★★ |
| 8.3 | Chunking：固定/递归/语义/Late/Small-to-Big | ★★★★★ |
| 8.4 | Embedding：从 Word2Vec 到 Qwen3-Embedding | ★★★★★ |
| 8.5 | 向量数据库：HNSW / 量化 / 各引擎对比 | ★★★★★ |
| 8.6 | 混合检索 + Rerank：BM25 / SPLADE / RRF / Cross-Encoder | ★★★★★ |
| 8.7 | Query 侧优化：HyDE / Multi-Query / Self-RAG / GraphRAG | ★★★★ |
| 8.8 | Agentic RAG：路由 / 工具 / 反思 / 与长上下文对比 | ★★★★ |
| 8.9 | RAG 评估：RAGAS / ARES / RGB / CRUD-RAG | ★★★★ |
| 8.10 | 工程落地：ES 混合索引 / 缓存 / 中文场景（重点） | ★★★★★ |
| 8.11 | 高频面试题 | ★★★★★ |
| 8.12 | 总结与延伸阅读 | ★★ |

> **本章主线问题**：RAG 凭什么能"低成本"地解决大模型幻觉、知识时效和私域可溯源？离线文档到在线回答这条链路上，哪些环节决定 RAG 的质量上限？工程上如何在延迟、召回、忠实度之间做权衡？

### 8.1 RAG 全景

#### 8.1.1 为什么需要 RAG

RAG（Retrieval-Augmented Generation，检索增强生成）由 Lewis et al. 2020 提出，核心思想是在 LLM 生成答案前，先用一个检索器从外部知识库中找出相关文档，把文档片段作为上下文拼进 prompt，再让 LLM 回答。

它解决了 LLM 的四个先天不足：

- **幻觉（Hallucination）**：参数化记忆里没有的，模型会"编造"。RAG 把事实性的回答建立在显式文档上，模型只要学会"照着写"就能拿到一个**可验证**的回答。
- **知识时效**：模型权重训练一次就定格，但世界一直在变。RAG 让你只更新索引就能"换脑子"，不用重训千亿参数。
- **私域数据**：企业内部的产品手册、合同模板、招投标文件根本不在公网预训练语料里。RAG 把这些"一次性喂给模型"成本变成"可持续更新的检索库"——这是它对企业最有价值的一点。
- **可溯源**：每条答案都能给出 chunk 级引用，用户可以反查"这答案到底从哪句话来的"。在金融、医疗、法律、招投标这种合规强相关场景，**引用是 RAG 的入场券，不是加分项**。

#### 8.1.2 RAG vs 微调：什么时候选谁

新人最容易问"我到底该微调还是该 RAG"。答案永远是**先看你的问题在解决谁**。

| 维度 | RAG | SFT / PEFT 微调 | 续训 / 长期记忆微调 |
|---|---|---|---|
| 解决问题 | 给模型**查字典** | 教模型**某种语气/格式/指令遵循** | 让模型**内化某领域知识** |
| 数据成本 | 文档本身 + 少量 QA 对 | 几千–几万条标注对 | 千万级领域语料 |
| 算力成本 | embedding + 检索（轻） | 单卡 LoRA 可起步 | 多卡全参预训练 |
| 时效性 | 索引更新即可 | 每次更新都要重训 | 几乎不可更新 |
| 可溯源 | 强，chunk 级引用 | 弱，答案来自参数 | 弱 |
| 风险 | 检索漏召 → 答不出 | 过拟合/灾难性遗忘 | 学错知识难回滚 |
| 适合场景 | 私域知识问答、客服、合规检索 | 风格对齐、输出格式、任务特化 | 领域基座、行业大模型 |

**经验法则**：RAG 解决"我不知道这件事"，SFT 解决"我应该用这种方式回答"，二者经常组合——先 RAG 拿事实，再 SFT 让回答稳定成你想要的格式。

#### 8.1.3 RAG 的三种范式：Naive / Advanced / Modular

按 Gao et al. 2024 的综述（"Retrieval-Augmented Generation for Large Language Models: A Survey"），RAG 的演进可分三类范式：

- **Naive RAG**：最朴素的"离线索引 + 在线检索 + 拼接 prompt + 生成"。Indexing：文档分块 → embedding → 入向量库。Retrieval：query 向量化 → top-k 相似度召回 → 拼接 → LLM。问题很多：chunk 切不好、召回不准、prompt 拼接没策略、没引用、没法迭代。
- **Advanced RAG**：在 Naive 的每一步做优化。**Pre-Retrieval**（query 改写、HyDE、metadata 过滤、路由）；**Retrieval**（混合检索、small-to-big、子查询分解）；**Post-Retrieval**（rerank、压缩、引用标注、上下文裁剪）。
- **Modular RAG**：把 RAG 拆成可插拔的模块（Search、Memory、Routing、Predict、Task Adapter、Reflection），各模块自由组合；进一步演化成 Agentic RAG，由 LLM 自己做规划、路由、反思。

![rag_evolution_paradigms](images/rag_evolution_paradigms.png)

> **面试提醒**：被问到"RAG 的最新进展"时，按 Naive → Advanced → Modular → Agentic 的演化脉络回答最稳，能体现你不是只看 2023 年那篇 Lewis 论文。

#### 8.1.4 RAG 与长上下文：会不会被替代

Claude 3.5 / Gemini 1.5 / GPT-4o 已经支持 100K–2M token 上下文，那 RAG 是不是过时了？**没有**，原因有三：

1. **成本**：1M token 的输入按 0.01 美元/1K 算，每次请求 10 美元；RAG 检索 10 个 chunk ≈ 2K token ≈ 0.02 美元。差距**两到三个数量级**。
2. **延迟**：1M token 的预填充（prefill）在 A100 上需要 30–60s；RAG 全链路通常小于 1 秒。
3. **Lost-in-the-Middle**：Liu et al. 2023ost in the Middle"）实测发现，相关文档放在 context 中间位置时，模型准确率从–75% 跌到–53%，**信息位置决定生成质量**——这是架构性的位置偏置。RAG 通过检索 + rerank + top-k 拼接，天然让相关 chunk 排在首尾两侧。

后文 §8.8.4 会定量比较"塞进 1M context" vs "RAG 取 20 个 chunk"在成本、延迟、忠实度上的差异。

### 8.2 文档解析

文档解析（Document Parsing）是把 PDF、Word、HTML、扫描件变成"可切、可嵌、可检索"的结构化文本。它是 RAG 流水线的第一步，**这一步错了，后面所有环节都救不回来**。

#### 8.2.1 PDF 的坑

PDF 不是"纯文本+排版"，它是"画在白板上的指令"：文本对象、字体、坐标、图像、矢量图都可能在不同位置。常见坑：

- **文字按"行"保存但顺序乱**：双栏论文提取出来是左右混排的乱序字符串，必须做阅读顺序恢复（reading order recovery）。
- **表格是矢量线 + 单元格文本**：直接复制粘贴会得到一堆错位的格子，需要专门的表格识别。
- **公式是图片或 MathML**：要么 OCR，要么保留 LaTeX。
- **扫描件没文字层**：整页就是一张图，必须先 OCR 才能往下走。
- **页眉页脚 / 页码混进正文**：不做版面分析会把"第 3 页"当成答案的一部分。

#### 8.2.2 主流工具对比

| 工具 | 强项 | 弱项 | 适用场景 |
|---|---|---|---|
| **PyMuPDF (fitz)** | 速度快、能取原始文本对象与坐标 | 表格识别弱，对扫描件无能为力 | 数字 PDF、文本层完整的财报 |
| **pdfplumber** | 表格抽取相对友好 | 速度慢，复杂版面易乱序 | 简单表格为主的报告 |
| **MinerU (opendatalab)** | 国产开源，对中文版面 + 公式 + 表格综合能力强 | 模型较大，部署成本中 | 中文论文、报告、教材 |
| **TextIn xParse (合合)** | 国产 SaaS，复杂版面 + 手写 + 印章识别 | 商业 API，按量付费 | 招投标扫描件、合同 |
| **Unstructured** | 多模态分区，元素类型细（Title / NarrativeText / Table） | 对中文与复杂表格表现一般 | 英文文档、混合内容 |
| **Marker** | 高质量 MD 输出，速度快 | 仅英文最优 | 英文论文转 Markdown |
| **Docling (IBM)** | 学术风格 PDF 处理 + 公式识别 | 速度慢、显存大 | 科研论文、arXiv |
| **PaddleOCR / PP-Structure** | 国产 OCR + 版面 + 表格 | 配置较复杂 | 中文扫描件 |
| **Surya / DocTR** | 现代 OCR 模型 | 表格识别弱 | 纯 OCR 场景 |

#### 8.2.3 版面分析与阅读顺序

版面分析（Layout Analysis）= 把页面拆成"标题 / 正文 / 表格 / 图片 / 页眉 / 页脚"等区域。常用模型：DiT（Document Image Transformer，UniDoc 系列）、LayoutLMv3、PP-DocLayout。

阅读顺序（Reading Order）：把拆出来的区域按"自上而下、自左而右、跨栏时先读完一栏再换"的逻辑串成一段。**没有阅读顺序，再好的分块都会把"上一节的结论"和"下一节的开头"缝在一起**。

#### 8.2.4 表格与公式

- **表格**：能保留 HTML/Markdown 形式最好，存成 `| col1 | col2 |\n| --- | --- |\n| ...`；LLM 看到表格比看到"列1 列2 列3\n值1 值2 值3"好得多。
- **公式**：优先保留 LaTeX（用 pix2tex / Nougat / Docling）；实在保不住就保留图像并在元数据里记"该 chunk 包含未识别的数学公式"。

#### 8.2.5 解析质量的下游影响

研究（Larson et al. 2024, "Evaluating the Impact of Source Documents on LLM-based RAG Systems"）表明：**文档解析错误是 RAG 系统最大单一失败源**。一个被错误切碎成 4 块的表格，比一个错误切碎成 4 段的叙述段落，**对回答的伤害大 2–3 倍**——因为表格是结构化信息，结构一断就全错。

![rag_doc_parsing_pipeline](images/rag_doc_parsing_pipeline.png)

> 上图示意了文档解析流水线的核心环节：原始 PDF/Word/扫描件 → 版面分析（标题 / 正文 / 表格 / 图片）→ 阅读顺序恢复 → OCR（扫描件）/ 公式识别 → 结构化输出（Markdown + HTML 表格 + LaTeX 公式）。**任何一个环节失误都会向下游传播**，因此这一阶段的容错率最低、工程投入最大。

### 8.3 切分 Chunking

**"Garbage in, garbage out" 在 RAG 里最具体的体现就是 chunking**。一个把"投标人资格要求"和"评标办法"硬切到一起的 chunk，会让 LLM 把两个独立规则混着答。

#### 8.3.1 固定长度切分

按字符数 / token 数一刀切。实现最简单，但致命：可能把"评分标准"四个字和"满分 100 分"切到两个 chunk。

```
chunk_1 = "本项目采用综合评估法，评标委员会根据投标人的..."
chunk_2 = "...资质、报价、技术方案综合打分，满分 100 分。"
```

**几乎从不单独使用**，仅作为 baseline 或最后兜底。

#### 8.3.2 递归字符切分（RecursiveCharacterTextSplitter）

LangChain 里的事实标准，按一个**优先级递减的分隔符列表**递归切分：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    length_function=len,
)
```

- 优先按段落切（`"\n\n"`），段落太长就按句子（`"\n"` / `"。"`），句子还长就按空格，再不行就按字符。
- **重叠率**经验值：10%–20%（chunk_size=512 → overlap=50–100）。重叠让"上一块的尾部"和"下一块的头部"有共同 token，减少边界信息丢失。
- 中文场景：分隔符里一定要有 `"。"`、`"！"`、`"？"`、`"；"`，否则会按英文习惯按空格切，效率极差。

#### 8.3.3 语义切分（Semantic Chunking）

用 embedding 模型**算相邻句子的相似度**，相似度低于阈值就切断。代表：LangChain 的 `SemanticChunker`。

变体：

- **百分位法**：算所有相邻句对相似度的分布，取 25 分位 / 50 分位作为阈值。
- **标准差法**：均值减去 n 倍标准差。
- **LumberChunker (Zhang et al. 2024)**：用 LLM 给每段打一个"分段是否合适"的决策，迭代合并直到某段被判为"不连续"。

代价：要多次 embedding 推理，索引构建慢 2–5 倍；阈值要调；中文标点/口语化文本相似度区分度差，**生产上不如结构化切分稳**。

#### 8.3.4 基于文档结构（Markdown 标题树 / 父子块）

对 Markdown / HTML 文档，按标题层级切：每个 H1 一块，每个 H2 一块……

进阶是**父子块（Parent-Document Retriever）**：

- **检索时用小块（small chunk）**——细粒度、命中率高、embedding 准。
- **生成时返回父块（parent chunk）**——上下文完整、LLM 看得多。

ES 里通常用两层索引：父块（段落/小节）存 BM25 字段；子块（句子级）存向量字段；检索命中子块后通过 `parent_id` 拉回父块。

#### 8.3.5 Late Chunking（Jina AI 2024）

Jina AI 团队（Günther et al. 2024, "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"）发现：传统"先切后 embed"会让"它的""该城市"这类指代在 chunk 中**丢失上下文**。Late Chunking 翻转这个顺序：

- **先让长上下文 embedding 模型对整篇文档做一次完整 forward**，得到每个 token 的 contextualized embedding；
- **再按边界把 token 序列切块**；
- **最后对每个块内的 token embedding 做 mean pooling**，得到该块的向量。

这样**每个 chunk 向量都保留了"它见过整篇文档"的信息**。在 Wikipedia / 学术搜索数据集上，Late Chunking + 简单固定边界，比"先切 + 语义边界"还要好。

代价：需要**支持 8K+ 上下文的 embedding 模型**（jina-embeddings-v3 = 8K，BGE-M3 = 8K，Qwen3-Embedding-0.6B = 32K）；单篇文档必须能塞进上下文窗口。

#### 8.3.6 Small-to-Big 与句子窗口检索

思路与 Late Chunking 类似：检索用细粒度（句子级），返回时给 LLM 更大的窗口（段落级）做上下文。

```
[小窗口检索]
  sentence_42: "投标人须具备 ISO 9001 认证。"
         ↓
  parent = paragraph_12 (含 5 个句子)
         ↓
[送入 LLM 的 prompt 上下文]
  段落全文 = 包含该句及其前后 4 句的完整段落
```

BGE-M3 内部就有"dense + sparse + 多向量（ColBERT 风格）"三种粒度输出，天然适合这种"细查粗用"。

#### 8.3.7 块大小怎么选

经验区间（**中文 RAG 为主**）：

![rag_chunking_strategies](images/rag_chunking_strategies.png)

| 任务 | 推荐 chunk_size | overlap | 备注 |
|---|---|---|---|
| 短答 FAQ | 128–256 | 0–32 | 强语义对齐 |
| 中等文档问答 | 256–512 | 32–64 | 最常见 |
| 长文档综述 / 报告 | 512–1024 | 64–128 | 配合 small-to-big |
| 代码检索 | 256–512 | 32 | 按函数级切 |
| 表格为主的文档 | 单表格 1 chunk | 0 | 不要切表格 |

没有"全局最优"，**必须在自己的业务数据上做 sweep**。指标看 RAGAS 的 Context Precision / Recall。

#### 8.3.8 切分评估

三个角度：

1. **块大小分布**：画个直方图，看有没有 90% 的块都恰好被切到上限的"尾巴效应"（说明分隔符设错了）。
2. **边界合理性**：抽 50–100 个块，人工评"这个块是否语义完整"。
3. **下游指标**：用 RAGAS / 检索 Recall@k 跑一遍，调 chunk_size + overlap 网格。

### 8.4 Embedding 模型

Embedding 是 RAG 的"语义编码器"，把文本变成稠密向量。**RAG 系统的检索上限 ≈ embedding 模型对 query 与 chunk 的语义区分能力**。

#### 8.4.1 演进时间线

- **2013 Word2Vec (Mikolov)**：第一个让"词向量算术（king − man + woman ≈ queen）"成立的模型。上下文无关、一个词一个向量。
- **2018–2019 BERT (Devlin)**：上下文相关的 token 向量。直接用 [CLS] 做句向量质量很差。
- **2019 Sentence-BERT (Reimers & Gurevych)**：**孪生网络**（Siamese Network）+ **均值池化**（mean pooling）让句向量在 STS 任务上比 BERT 原生用法高 10+ 点。CLS 不如 mean——这是几乎所有后续 sentence embedding 的默认设置。
- **2019–2021 对比学习**：SimCSE (Gao 2021) 用 dropout 做"无监督对比学习"——同一句话过两次 encoder 得到两个不同 embedding 当正样本；SupSimCSE 用 NLI 数据（蕴含/矛盾）当有监督正负例。
- **2022 BGE / M3E / GTE / E5**：中文 / 多语言 embedding 模型密集开源，**MTEB 榜单成了行业基准**。
- **2023–2025 指令感知 embedding**：BGE-M3、Qwen3-Embedding 支持**asymmetric retrieval**——query 加 `为这个句子生成表示以用于检索相关文章：` 前缀，doc 加 `为这个句子生成表示以用于检索相关文章：` 不同前缀。模型用同一编码器但在 prompt 模板上对齐 query/doc 的语义空间。
- **2024–2025 LLM 当 embedding 底座**：Qwen3-Embedding 把 Qwen3 系列的 0.6B / 4B / 8B decoder-only 模型作为底座，加对比学习头；MTEB Multilingual 榜单（2025-06-05 数据）Qwen3-Embedding-8B 拿到 **70.58 分位列第一**（来源：qwenlm.github.io/blog/qwen3-embedding/）。

#### 8.4.2 Sentence-BERT 原理（必须吃透）

```
        ┌─ Transformer(θ) ─┐
text_a ─┤                 ├── h_a (mean pooling over tokens)
        └─────────────────┘
        ┌─ Transformer(θ) ─┐
text_b ─┤                 ├── h_b (mean pooling over tokens)
        └─────────────────┘
                ↓
        cosine(h_a, h_b) → softmax with label
```

- **孪生网络**：两路 encoder 共享权重。
- **CLS vs mean pooling**：CLS 是预训练时的"分类头"，对句级语义"无感"；mean pooling 把所有 token 向量取平均（去掉 padding），更适合句级任务。
- **训练目标**：NLI 三分类（entailment / neutral / contradiction）当有监督信号，或者用 (anchor, positive, negative) 三元组做 triplet loss。

#### 8.4.3 主流模型速查（2025-2026）

| 模型 | 参数量 | 向量维度 | 上下文 | 特点 / 用途 |
|---|---|---|---|---|
| **text-embedding-3-small** (OpenAI) | 闭源 | 1536（Matryoshka） | 8192 | 便宜、商用 SOTA |
| **text-embedding-3-large** (OpenAI) | 闭源 | 3072（Matryoshka） | 8192 | 多语种仍可用但 C-MTEB 不一定最优 |
| **BGE-M3** (BAAI) | 568M | 1024 dense + sparse + ColBERT | 8192 | 中文 SOTA，**一模型支持 dense/sparse/多向量** |
| **BGE-large-zh-v1.5** | 326M | 1024 | 512 | 纯中文场景经典 |
| **bge-reranker-v2-m3** | 568M | 标量分 | 8192 | 重排阶段首选 |
| **Qwen3-Embedding-0.6B** | 0.6B | 1024 (MRL 32–1024) | 32768 | 性价比之王，**中文 RAG 首选** |
| **Qwen3-Embedding-4B** | 4B | 2560 (MRL 32–2560) | 32768 | 高质量 |
| **Qwen3-Embedding-8B** | 8B | 4096 (MRL 32–4096) | 32768 | MTEB Multilingual #1 (70.58) |
| **Qwen3-Reranker-0.6B / 4B / 8B** | 0.6B–8B | 标量 | 32768 | 2025 年开源 SOTA 重排器 |
| **mE5 / multilingual-e5-large** | 560M | 1024 | 512 | 微软多语言，指令前缀 |
| **nomic-embed-text-v1.5** | 137M | 768 (Matryoshka 64–768) | 8192 | 极致轻量，CPU 可跑 |
| **mxbai-embed-large** | 335M | 1024 | 512 | 英文 MTEB 64.68 |
| **jina-embeddings-v3** | 570M | 1024 (MRL 32–1024) | 8192 | 任务 LoRA + Matryoshka，**支持 Late Chunking** |
| **cohere embed-v4** | 闭源 | 1024 (MRL) | 128K | 商用，长上下文，**但闭源 + 按量** |
| **voyage-4** | 闭源 | 1024/2048 (MRL) | 32K | 商用，金融/法律领域偏强 |

来源：qwenlm.github.io/blog/qwen3-embedding/、morphllm.com/ollama-embedding-models、Hugging Face 模型卡。

#### 8.4.4 MTEB / C-MTEB 榜单怎么看

**MTEB (Massive Text Embedding Benchmark)** 由 Hugging Face 维护，覆盖 100+ 任务，分成 8 大类：Retrieval、Classification、Clustering、PairClassification、Reranking、Retrieval、STS、Summarization。中文有专门的 **C-MTEB** 子集。

**看榜单的正确姿势**：

- **看子任务，不要看总平均**：你是做检索就盯 Retrieval 子集的平均分；做聚类就盯 Clustering。
- **看语言**：英文 MTEB v2 跟 C-MTEB 不是一回事；多语言模型（MTEB Multilingual）覆盖 250+ 语言，但要平衡。
- **看维度**：分数差不多时，**维度低的更省存储、检索更快**（Matryoshka 截断后两全）。
- **看上下文长度**：长文档 RAG 必须用支持 8K+ 的模型。
- **看许可证**：商用项目避开 CC-BY-NC 类（jina 系列部分版本、bge 早期部分模型）。

#### 8.4.5 Matryoshka 套娃表示学习

Kusupati et al. 2022 (NeurIPS) 提出的训练方法：让模型在训练时**同时优化多个维度的子向量**——前 32 维、前 64 维、前 128 维……前 d 维都接近最优。代价：单个训练目标增加 5%–15% 推理开销。好处：存 1024 维时检索精度高；存 256 维时存储省 4×、检索快 4×。

**工程意义**：

- 上线初期用 1024 维保证效果；
- 数据量起来后切 256 维，存储 / 延迟 / 成本同比例降。

Qwen3-Embedding 全系、BGE-M3（需 Matryoshka-aware 截断）、nomic-embed-v1.5 都支持。

#### 8.4.6 指令前缀（asymmetric retrieval）

早期 embedding 模型对 query 和 doc 喂**同一种文本**——这对短 query / 长 doc 场景效果差（向量空间被强行拉伸）。

**解决**：query 用一种 prompt 模板，doc 用另一种 prompt 模板，模型在训练时就学会"把这两类文本映射到同一空间的不同区域"。

```
# BGE 中文检索
query_prompt = "为这个句子生成表示以用于检索相关文章："
doc_prefix = ""  # BGE 中文场景 doc 不加前缀

# bge-m3 (多语言)
query_prefix = ""  # bge-m3 不区分 prefix
```

OpenAI text-embedding-3 也用类似机制（内部不可见）。**实操时一定要用模型对应的官方模板**，否则会掉 1–3 点。

#### 8.4.7 向量维度、归一化与距离

| 距离 | 公式 | 何时用 |
|---|---|---|
| **余弦相似度** | `cos(a,b) = dot(a,b) / (‖a‖·‖b‖)` | 文本长度差异大、多数场景 |
| **点积（内积）** | `dot(a,b) = Σ a_i·b_i` | 向量已 L2 归一化（等价余弦）、速度优先 |
| **欧氏距离（L2）** | `‖a-b‖` | 空间结构重要时（少用于 RAG） |

- 大多数 embedding 训练时是按余弦相似度优化；**embedding 输出后做 L2 归一化**，点积就等于余弦。
- OpenAI、BGE、Qwen3-Embedding 默认输出已归一化，**直接用点积**。
- ES 的 `dense_vector` 用 `cosine` / `dot_product` / `l2_norm` 三选一；FAISS 用 `IndexFlatIP`（点积）或 `IndexFlatL2`（欧氏）。

**向量维度怎么选**：

- 768 / 1024 是 2025 年的事实标准（BGE、Qwen3-Embedding-0.6B、E5）；
- 3072 / 4096 留给"追求极致召回"且**存储不敏感**的场景（OpenAI large / Qwen3-8B）；
- 256 以下除非有强存储约束，否则不推荐——会让长文档语义被压平。

![rag_embedding_evolution](images/rag_embedding_evolution.png)

> 上图（参考 Hugging Face 博客"32_1b_sentence_embeddings"系列图）示意了文本表示从 2013 Word2Vec 的静态词向量到 2019 Sentence-BERT 的孪生网络句向量，再到 2024 指令感知 embedding 的演进路径。每个阶段的洞察都源自上一阶段的失败：Word2Vec 无上下文 → BERT 的 [CLS] 不如 mean pooling → 不区分 query/doc 的 embedding 拉跨了异构检索。

**选型口诀**：中文 + 开源 + 性价比 → Qwen3-Embedding-0.6B（1024 维 / 32K context）；中文 + 一模型多模态 → BGE-M3；商业 + 极致质量 → text-embedding-3-large / voyage-4 / cohere v4；离线/边缘 → nomic-embed-v1.5 或 mxbai。

### 8.5 向量数据库

向量数据库（Vector Database）= 在亿级、十亿级向量上做**亚秒级相似度检索**的专用系统。RAG 系统的检索延迟、并发、运维复杂度，**70% 取决于这块**。

#### 8.5.1 ANN 三大类索引

暴力（brute-force）= `O(N·d)`，百万级向量还行，到亿级就要分钟级不可用。必须用**近似最近邻（Approximate Nearest Neighbor, ANN）**，在工程可接受的精度损失下把复杂度压到亚线性。

| 家族 | 代表 | 思想 | 复杂度 | 现状 |
|---|---|---|---|---|
| **哈希** | LSH（局部敏感哈希） | 设计哈希函数让"近的点碰撞概率高" | 亚线性 | 历史地位，召回低，被图/PQ 取代 |
| **树** | KD-tree、Ball-tree、Annoy | 递归切空间 | 维数大于 20 时退化 | 几乎不用作主索引 |
| **图** | **HNSW**、NSG、Vamana | 多层可导航小世界图 | 查询 `O(log N)`、插入 `O(log N)` | **2025 年事实标准** |
| **量化** | PQ、SQ、IVF-PQ | 把向量压缩到几十字节 | 距离计算近似 | **跟 HNSW 组合用** |
| **倒排** | IVF（Inverted File） | 先聚类分桶、查最近的几桶 | `O(√N)` | 大规模场景 |

HNSW + 量化（PQ 或 SQ）是当前所有生产级向量库的"标准武器"。

#### 8.5.2 HNSW 详解（Hierarchical Navigable Small World）

Malkov & Yashunin 2018（TPAMI, arXiv:1603.09320）提出。直觉来自两个观察：

- **小世界网络**（six degrees of separation）：真实世界的人际网络有"短路径"——任意两个节点都能在几步内到达。
- **跳表（Skip List）**：多层链表让"跳跃"和"细查"分离开。

HNSW 把两者结合：

```
层 3:  ●────────────●                          ← 顶：稀疏、长程边
层 2:  ●─────●──────●──●                       ← 中：稍密
层 1:  ●──●─●──●─●──●──●─●                     ← 底：密集
层 0:  ●●─●●●─●●─●●●─●●●●─●●─●●●●─●●●       ← 底层：所有节点 + 短边
       ↑ 入点
```

**插入**：每个新节点按几何分布随机选一个最大层 l（`l ~ floor(-ln(rand) · mL)`，`mL = 1/ln(M)`），从顶层开始贪心下到第 l 层，再从第 l 层下到底层 0，每层连到最近的 M 个邻居（多样性剪枝避免 clique）。

**搜索**：从顶层入口贪心下降到层 0（每层只走 1 步），到层 0 后用宽度为 `efSearch` 的 beam search 展开。

**三个核心参数**：

| 参数 | 作用 | 典型值 | 调参效果 |
|---|---|---|---|
| `M` | 每节点最大双向边数（层 0 = 2M） | 16（默认）–64 | 越大 recall 越高、内存越大 |
| `efConstruction` | 构建时 beam 宽度 | 100–200 | 越大图质量越高、构建越慢 |
| `efSearch` | 查询时 beam 宽度 | 50–500 | 越大 recall 越高、查询越慢 |

复杂度：

- 构建 `O(N log N)`、内存 `O(N · M)`；
- 查询 `O(log N)`，常数小；
- 删除：原生支持但**会让图稀疏**，生产上常用"标记删除 + 定期重建"。

**为什么 HNSW 几乎统治了 2020 年后的向量库**：

- 单机单卡能跑 1M–10M 向量、亚毫秒延迟；
- 召回稳定 95%+（HNSW 调 efSearch 后常到 99%）；
- 实现简单，开源生态成熟（hnswlib、FAISS、Qdrant、Milvus、Weaviate、pgvector、ES、Redis 全都支持）。

**边界**：

- 内存大：1M × 1024 维 float32 = 4 GB，HNSW 边再加 100–200 MB；超过单机 RAM 就得 IVF-PQ 上 SSD。
- 删除与频繁更新会让 recall 漂移，需要周期重建。
- 图结构对单机 RAM 强依赖，分布式方案（如 Milvus 的 segmented HNSW）增加复杂度。

#### 8.5.3 量化（PQ / SQ / IVF-PQ）

当向量维度 × 数量超过单机内存时，必须压缩。

- **PQ（Product Quantization，乘积量化）**：把 d 维向量切成 m 段，每段独立做 k-means 得到 256 个 centroid。**每段用一个 8 bit 索引代替**，一个向量从 `4·d` 字节压到 `m` 字节（如 d=1024, m=64 → 64 字节，压缩 64×）。查时用 ADC（Asymmetric Distance Computation）近似距离。
- **SQ（Scalar Quantization）**：每维单独量化到 int8（每字节 1 个维度）。比 PQ 简单、精度更高，压缩 4×。
- **IVF-PQ**：先用 k-means 把全量向量分到 nlist 个桶（倒排），每桶内做 PQ。查时只搜最近的 nprobe 个桶，是**十亿级向量的标准武器**。

ES 8.x 默认量化策略：

| type | 压缩 | 精度 | 何时用 |
|---|---|---|---|
| `hnsw` | 无 | 最高 | 召回最优先（默认） |
| `int8_hnsw` | 4× | 几乎无损 | 内存受限时首选 |
| `int4_hnsw` | 8× | 轻微损失 | 极致内存压缩 |
| `bbq_hnsw` | 动态 | 较好 | 2024+ 新版，ES 自研二值化 |

#### 8.5.4 各向量库 / 引擎对比

| 引擎 | 部署复杂度 | 混合检索 | 标量过滤 | 分布式 | 成本 | 中文社区 |
|---|---|---|---|---|---|---|
| **FAISS** (Meta) | 库级（嵌入应用） | 否 | 否 | 需自建 | 开源免费 | 中 |
| **Milvus** (Zilliz) | 中（需 etcd + minio + 节点） | 弱 | 强 | **原生分布式** | 开源 + 云版 | **强** |
| **Qdrant** | **低**（单 Rust 二进制） | **强**（原生 dense+sparse+filter） | **强**（filter 嵌入 HNSW 遍历） | 集群版 | 开源 + 云 | 中 |
| **Weaviate** | 中 | **强**（模块化 hybrid） | 中 | 集群 | 开源 + 云 | 中 |
| **Chroma** | **低**（嵌入式） | 弱 | 弱 | 不适合大规模 | 开源 | 弱 |
| **Elasticsearch** | 已有集群时低 | **强**（BM25 + knn + RRF） | 强 | 强 | 商业 | **强** |
| **pgvector** | 极低（PG 扩展） | 中（需手动 SQL 拼） | **强**（PG 原生） | 弱（小于 5M 向量） | 开源 | 中 |
| **LanceDB** | 低（嵌入式 + 列存） | 弱 | 中 | 弱 | 开源 | 弱 |

**选型经验**（2025）：

- **已有 ES 集群 / 重 BM25 / 要强标量过滤** → **ES dense_vector + HNSW + RRF**（**RAG 主流方案**）；
- **全新搭建 / 高并发 / 大规模（亿级）** → Milvus 或 Qdrant；
- **轻量 / 单机 / 原型** → Chroma 或 LanceDB；
- **跟业务库同库** → pgvector；
- **想最稳 + 默认混合检索** → Qdrant。

> **真实项目选型参考**（招投标 RAG 知识库场景）：数据量小于 100 万 chunk + 强 BM25 关键词需求 + 已有运维栈 → 几乎都是 ES 双索引（BM25 索引 + 向量字段）。

![rag_vector_db_compare](images/rag_vector_db_compare.png)

### 8.6 混合检索与重排

单靠 dense retrieval 在生产上不够。**BM25 抓精确关键词、向量抓语义、cross-encoder 抓精细匹配**——三件套合起来才能稳。

#### 8.6.1 BM25（Best Matching 25）

Robertson & Zaragoza 2009，TF-IDF 家族的"集大成者"。公式：

$$
\mathrm{score}(D, Q) = \sum_{i=1}^{n} \mathrm{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\mathrm{avgdl}}\right)}
$$

每个符号：

| 符号 | 含义 |
|---|---|
| `q_i` | query 中第 i 个词项 |
| `f(q_i, D)` | `q_i` 在文档 D 中出现的次数（TF） |
| `|D|` | 文档 D 的长度（词数） |
| `avgdl` | 语料库平均文档长度 |
| `k1` | TF 饱和参数，**常用 1.2–2.0，默认 1.2** |
| `b` | 文档长度归一化，**常用 0.75**，0=不归一化，1=完全归一化 |
| `IDF(q_i)` | 逆文档频率，`log((N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1)` |

**为什么 BM25 至今没被淘汰**：

- **精确匹配强**：产品编号、错误码、专有名词、缩写——embedding 模型最容易栽的地方，BM25 不会。
- **零训练**：没有 embedding 模型就没有冷启动、版本管理、长尾词问题。
- **极快**：倒排索引，毫秒级。

**代价**：

- 不理解语义（"汽车" ≠ "vehicle"）；
- 对中文要分词（IK / jieba / HanLP）；
- 跟向量分数不可直接比较（见 §8.6.3 RRF）。

#### 8.6.2 SPLADE（学出来的稀疏向量）

Formal et al. 2021：用 BERT 模型**学出"哪些词重要"**——对每个词项输出一个权重，类似于"学出来的 BM25 词权重"。

```
[query = "新能源汽车补贴"]
                  ↓
[SPLADE encoder]
                  ↓
[term weights]
  "新能源": 1.42
  "汽车":   0.89
  "补贴":   1.78
  "电动":   0.61   ← 语义扩展
  "购车":   0.34   ← 语义扩展
  "的":     0.02
                  ↓
[稀疏向量，仍走倒排索引]
```

**优点**：稀疏（可走倒排索引）+ 语义扩展（"新能源汽车"自动补"电动"）；**代价**：比 BM25 慢、需要 GPU、模型要训练。

适合：冷门术语 + 行业黑话多的场景（如医疗 ICD 码、法律条款引用）。

#### 8.6.3 RRF：倒数排序融合（Reciprocal Rank Fusion）

Cormack, Clarke, Büttcher 2009 SIGIR 提出。**只关心名次，不关心分数**——这是它最大的优点，因为 BM25 分数和 cosine 分数根本不在一个量纲上。

$$
\mathrm{RRF\_score}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}
$$

- `k = 60` 是原论文实验得到的事实标准（k=40–80 都接近最优）；
- `rank_r(d)` = 文档 d 在第 r 个 retriever 结果里的名次（1-indexed）；
- 文档不在某个结果里 = 贡献 0。

**为什么不用分数加和**：

BM25 分数可以 0–50，cosine 分数 0–1；如果直接加，BM25 主导整个结果。即使做 min-max 归一化，**归一化窗口小（top-10）时会把一个低分 doc 拉到接近 1**，污染融合。RRF 用 rank 天然规避。

**为什么 k 取 60**：

| 排名 | RRF 贡献（k=60） |
|---|---|
| 1 | `1/61 ≈ 0.01639` |
| 10 | `1/70 ≈ 0.01429` |
| 100 | `1/160 = 0.00625` |

排名 1 比 100 多 2.6×；排名 1 比 10 多–15%。**温和的衰减曲线**让"两个检索器都认可"的中等排名比"只有一个第一"更重要。

**ES 8.8+ 原生 RRF 语法**：

```json
GET /bidding_docs/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        { "standard": { "query": { "multi_match": { "query": "投标人资质要求", "fields": ["title^3", "content"] } } } },
        { "knn": { "field": "embedding", "query_vector": [0.12, -0.34, ...], "k": 20, "num_candidates": 100 } }
      ],
      "rank_window_size": 50,
      "rank_constant": 60
    }
  }
}
```

**加权 RRF**：`sum(w_r / (k + rank_r(d)))`，对偏信某一检索器的场景手动加权。

#### 8.6.4 Rerank 重排

第一阶段检索（BM25 + dense + RRF）负责"召回 50–200 个候选"，**精度不够**——前 5 个里可能第 3、4 个是误召。第二阶段 Rerank 用更精细的模型对候选**两两精排**。

**Bi-Encoder vs Cross-Encoder**：

| 架构 | 推理方式 | 速度 | 精度 | 用法 |
|---|---|---|---|---|
| **Bi-Encoder**（第一阶段） | query 和 doc **分开编码**为向量 | 快（向量预计算好） | 中 | 海量候选初筛 |
| **Cross-Encoder**（第二阶段） | query 和 doc **拼一起**过一次 transformer | **慢**（每对都要过一遍） | **高** | top-50–200 精排 |

为什么 Cross-Encoder 更准：query 和 doc 在每一层 self-attention 里**充分交互**（"投标人"和"资质"之间的关系被精确建模），而 Bi-Encoder 只能"分别编码后做点积"——信息瓶颈在两段向量拼接处。

代价：每对 query-doc 都要过一次模型，**Rerank 50 个候选 ≈ 50 次推理**。对延迟敏感时要控制候选数（20–50 起步）。

**典型 Rerank 模型**（2025 选型参考）：

| 模型 | 规模 | 上下文 | 速度 | 适合 |
|---|---|---|---|---|
| **bge-reranker-v2-m3** | 568M | 8192 | 中 | 通用首选，**多语言 + 中文** |
| **BGE-reranker-large** | 560M | 512 | 中 | 英文经典 |
| **Cohere Rerank v3 / v4** | 闭源 API | 4K–8K | 快（API） | 商用，**质量 SOTA**，按量付费 |
| **Qwen3-Reranker-0.6B / 4B / 8B** | 0.6B–8B | 32K | 中 | **2025 开源 SOTA**（MTEB-R 0.6B 已 65.80，8B 69.02） |
| **Jina Reranker v2** | 0.3B | 8K | 中 | 多语言，质量与速度平衡 |
| **ColBERT / ColBERTv2** | 110M+ | 任意 | 中 | 晚交互（late interaction），**精度接近 Cross-Encoder、速度快** |
| **text-embedding-3 + self-rerank** | 闭源 | 8K | - | OpenAI 用户省事方案 |

**两阶段延迟预算**（典型云端）：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| query embedding | 30–80 ms | 5% |
| BM25 + knn + RRF（双路） | 50–150 ms | 20% |
| Cross-Encoder Rerank（top-50） | 200–500 ms | 60% |
| LLM 生成（流式首字） | 200–800 ms | 余 |
| **端到端首字** | **500–1500 ms** | 100% |

**Rerank 经验值**：候选数 30–50（再大边际收益递减、延迟上升）；**Rerank 后 RAGAS Faithfulness 通常涨 5–15 个点**。

![rag_rerank_pipeline](images/rag_rerank_pipeline.png)

### 8.7 Query 侧优化

用户输入的 query 经常是"碎片、口语、含错字、指向不明"的。**Query 优化是 RAG 系统里最容易被忽视的提效杠杆**。

#### 8.7.1 Query 改写（Rewrite）

把口语化 / 残缺 query 改写成更"检索友好"的版本：

![rag_query_optimization](images/rag_query_optimization.png)

```
[user]  "那个关于 XX 项目的预算文件在哪"
        ↓
[rewrite] "XX 项目 预算 文件 文档名称"
```

实现：用一个轻量 LLM（甚至规则）改写；或用同义词表扩展。

#### 8.7.2 HyDE（Hypothetical Document Embeddings）

Gao et al. 2023 ACL。**反直觉**：不让 LLM 改写 query，而是让它**生成一个"假设性的答案段落"**——然后用这个段落去检索。

```
query: "光伏板清洗频次"
        ↓
[LLM 生成]
hypothetical_doc: "光伏板每季度清洗一次，遇到沙尘暴后立即清洗。清洗时..."
        ↓
[embed(hypothetical_doc)]  →  向量检索
```

**为什么有效**：真实文档的向量空间和短 query 的向量空间分布差异大（短 query 偏"问句"分布，文档偏"陈述"分布），让 LLM 生成一段"假文档"对齐分布，召回更准。

**代价**：每次检索多一次 LLM 推理（+200–500 ms）；Anthropic 自己也提过有用户实测 HyDE **在某些场景反而拉低 RAGAS 分数**——Anthropic 报告 HyDE 在他们的实验上不一致；适用与否要在自己数据上验证。

#### 8.7.3 Multi-Query 多路查询

用一个 LLM 把原 query 拆成 3–5 个**不同角度**的子 query，分别检索后融合（RRF）。

```
query: "本项目是否允许联合体投标"
        ↓
[multi-query]
  q1: "联合体投标 资格要求"
  q2: "本项目 是否接受联合体"
  q3: "投标人组成形式 联合体"
        ↓
[3 路 RRF 融合]
```

适用于：用户 query 简短、文档量大、单一 query 召回不全。

#### 8.7.4 Step-Back Prompting

把具体问题抽象一步：

```
[user] "Q2 营收增长 3% 来自哪些产品线"
[step-back] "Q2 公司营收按产品线的拆分"
```

抽象后检索范围更广，更容易命中"按产品线拆分"的总览段落，再让 LLM 在答案里细化到 Q2。

#### 8.7.5 子查询分解（Sub-Query Decomposition）

复杂多跳问题拆成子问题，**串行检索**（前一个答案作为后一个的上下文）：

```
Q: "本项目投标人需具备哪些 ISO 认证？这些认证有效期如何？"
        ↓
[分解]
  Q1: "投标人 ISO 认证要求"
  Q2: "Q1 检索到的认证各自的有效期"
        ↓
[串行检索 + 累积上下文]
```

配合 Self-Ask（Press et al. 2022）或 ReAct 框架使用。

#### 8.7.6 Self-RAG（Asai et al. 2023 ICLR）

让 LLM **自己决定是否需要检索 + 自己评估检索质量**。训练时引入四种**反思 token**（Reflection tokens）：

| Token | 含义 | 取值 |
|---|---|---|
| `[Retrieve]` | 是否需要检索 | `{yes, no, continue}` |
| `[IsRel]` | 检索到的段落是否相关 | `{relevant, irrelevant}` |
| `[IsSup]` | 生成内容是否被段落支持 | `{fully, partially, no}` |
| `[IsUse]` | 生成内容整体有用性 | 1–5 |

推理流程：

```
[user query]
   ↓
[决定是否需要检索]  →  [Retrieve: no] 直接回答
   ↓ yes
[检索 top-k]
   ↓
[对每段评估相关性]  →  [IsRel] 过滤不相关
   ↓
[生成 + 标注支持度]  →  [IsSup] 检查事实
   ↓
[整体有用性]  →  [IsUse] 决定是否继续
   ↓
[最终答案]
```

Self-RAG 的代价：要训一个带 reflection token 的模型；开源版本有 `self-rag` (7B/13B)，但相对少见。生产上更常用 Corrective RAG（CRAG）这种"用现成 LLM + 评分 prompt"的方式。

#### 8.7.7 Corrective RAG（CRAG, Yan et al. 2024）

把 Self-RAG 的"反思"思路做成 pipeline：

```
[user query]
   ↓
[retriever 召回 top-k]
   ↓
[relevance grader:  LLM 评每段是否相关]
   ↓
├ 相关 → 直接用
├ 部分相关 → 走 web search 补全
└ 不相关 → 触发 query rewrite + 重新检索
   ↓
[最终上下文 → LLM 生成]
```

**比 Self-RAG 轻**：用现成 LLM 评分 prompt 即可实现 reflection，不需要训新模型。

#### 8.7.8 GraphRAG（Microsoft 2024）

Edge et al. 2024, "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"。**全局查询的杀手锏**——比如"本公司所有产品中前五大客户是谁"，传统向量 RAG 召回 N 个 chunk 也答不全。

**核心思路**：

1. **离线索引**：
   - 用 LLM 从每个 chunk 抽 **实体 + 关系**（entity-relation extraction）；
   - 构建知识图谱（实体为节点，关系为边）；
   - 用 Leiden 算法做**社区检测**（community detection）；
   - 用 LLM 对每个社区生成**摘要**（community summary / report）。

2. **在线查询**：
   - **Local Search**：从 query 出发找相关实体，扩展邻域 → 适合"X 是什么 / 跟 Y 关系"。
   - **Global Search**：从所有社区摘要走 Map-Reduce —— 每个社区摘要生成"部分答案 + 重要性分"，再 reduce 出总答案。**适合"数据中前 N 大 / 跨文档主题 / 时间趋势"**。

**代价**：

- 索引阶段两次 LLM pass（实体抽取 + 社区摘要），**对 1M token 语料用 GPT-4-turbo 大约 281 分钟 + 数百到上千美元**；
- 增量更新：图结构变化时必须重新检测社区。

**LazyGraphRAG**（Microsoft, 2024-11）：用 NLP（noun phrase）替代 LLM 做实体抽取、把摘要延后到查询时——**索引成本 = 向量 RAG，全局查询质量 = GraphRAG 99%、查询成本仅 4%**。是当前更工程化的选择。

**LightRAG**（Guo et al. 2024, HKUDS, arXiv:2410.05779）：图结构 + 双层（low-level / high-level）检索 + 增量更新。**比 GraphRAG 轻 2–5×**，适合中小规模知识库。

![rag_graphrag_pipeline](images/rag_graphrag_pipeline.png)

#### 8.7.9 各方法选型速查

| 场景 | 推荐方法 |
|---|---|
| query 简短 / 文档多 | Multi-Query + RRF |
| 短 query → 长文档 | HyDE |
| 用户 query 残缺 / 口语 | Query Rewrite |
| 多跳推理 | 子查询分解 + ReAct |
| 单一答案找不到 | Step-Back |
| 全局主题 / 跨文档汇总 | GraphRAG / LazyGraphRAG |
| 严格事实 / 高合规 | Self-RAG / CRAG |
| 实体关系密集（如招投标、企业） | LightRAG |

### 8.8 Agentic RAG

Agentic RAG（代理式 RAG）= 把 RAG 从固定 pipeline 变成**由 LLM 驱动的 agent**：自己决定检索什么、用哪个索引、查几次、怎么组合。

#### 8.8.1 从 Pipeline 到 Agent

传统 RAG 是直线：query → retriever → rerank → LLM。**真实用户的问题常常需要**：

- 问"招标时间"——走 FAQ 索引；
- 问"近 5 年公司中标的项目"——走 SQL 工具；
- 问"最新政策"——走 Web 搜索；
- 问"二者对比"——多轮检索 + 比较。

Agent 把这些决策交给 LLM：

```
[user query]
   ↓
[LLM 思考]
   ├ "这是个时间 + 实体问题"  → 选工具 A：SQL 查数据库
   ├ "还需要最新政策"        → 选工具 B：Web 搜索
   └ "给出对比"              → 选工具 C：之前检索到的内容
   ↓
[ReAct loop: 思考 → 行动 → 观察 → 反思]
   ↓
[最终答案]
```

#### 8.8.2 ReAct 在 RAG 中的落法

ReAct（Reason + Act, Yao et al. 2022）让 LLM 显式输出"思考 + 工具调用 + 观察"的循环：

```
Thought 1: 用户问投标人资格，需要先查标书原始条款。
Action 1:  retriever.search(query="投标人 资格要求", top_k=10)
Observation 1: 找到了 5 段相关条款...

Thought 2: 还需要确认联合体投标的额外要求。
Action 2:  retriever.search(query="联合体 投标 资格")
Observation 2: 找到 2 段...

Thought 3: 信息足够，可以回答了。
Action 3:  Answer(...)
```

**生产上要注意**：

- **最大步数**（通常 3–5 步）：防止死循环；
- **Token 预算**：每步都要把历史塞回 context，4 步后接近 8K+；
- **工具描述质量**：工具清单是 prompt 的一部分，**写好工具描述等于 50% 的成功率**。

#### 8.8.3 关键能力清单

Agentic RAG 需要的不只是 LLM 聪明：

| 能力 | 作用 | 实现 |
|---|---|---|
| **路由（Routing）** | 根据 query 选工具 | LLM 分类 / few-shot prompt / 轻量分类器 |
| **查询规划（Planning）** | 多跳问题拆解 | CoT / ReAct / Plan-and-Execute |
| **工具调用（Tool Use）** | SQL / Web / 计算器 | Function calling（OpenAI / Anthropic / Qwen3 工具调用） |
| **反思（Reflection）** | 评估上一步结果 | CRAG / Self-RAG / Reflexion |
| **记忆（Memory）** | 多轮对话上下文 | 短期 context + 长期向量库 |
| **可控性** | 人能干预 | HITL（human-in-the-loop）审批工具调用 |

#### 8.8.4 长上下文能否替代 RAG

一句话：**不能，但场景在收窄**。

定量对比（2025 年生产环境典型数据）：

| 维度 | RAG（top-20） | 长上下文（200K–1M） |
|---|---|---|
| 单次输入 token |–2K–4K | 200K–1M |
| 单次成本 | 0.005–0.02 美元 | 1–10 美元（按主流闭源定价） |
| 首字延迟 | 0.5–1.5 s | 5–30 s（长 prefill） |
| 端到端 faithfulness | 0.7–0.9 | 0.6–0.85（受 lost-in-the-middle 拖累） |
| 知识更新 | 改索引（分钟级） | 改权重（不现实） |
| 引用 | 强（chunk 级） | 弱（无明确段落） |
| 适用知识库规模 | 百万–亿级文档 | 小于 500 页 |

**Lost-in-the-Middle 的实测量级**（Liu et al. 2023, GPT-3.5-Turbo, 20 篇文档）：

| 答案位置 | 准确率 |
|---|---|
| 第 1 篇 |–75% |
| 第 5 篇 |–70% |
| 第 10 篇（中间） |–53% |
| 第 15 篇 |–65% |
| 第 20 篇 |–66% |

> 注意：U 形曲线对 decoder-only 普遍存在；encoder-decoder 与新代际 decoder-only 较平，但**未消失**。

**实用策略**：

- 小于 200K token 知识库 → **长上下文 + 提示词技巧**（Q+Docs+Q 模式把"针"放两边）更省事；
- 200K–1M token → 长上下文做"二阶段粗筛"+ RAG 做"精查"；
- 超过 1M token、高频更新、强引用需求 → **RAG 仍然是唯一合理选择**。

#### 8.8.5 Agentic RAG 的代价

- **延迟**：每次 LLM 决策–300–800 ms，多步 1–3 s 起步；
- **成本**：单次回答 4–10 次 LLM 调用；
- **可控性**：同一个 query 不同次可能走不同路径，A/B 评估困难；
- **可靠性**：必须做异常处理（工具失败、超时、循环）——用 LangGraph / LlamaIndex Workflows / DSPy 这类框架比裸 prompt 稳得多。

**生产建议**：

- 大部分内部知识库**不需要 Agentic**——一条 Advanced RAG pipeline 解决 80% 问题；
- 当 query 类型多、需要外部工具、或者必须做"边查边反思"时，引入 Agentic；
- 上 Agentic 前先把离线 RAGAS 跑到 0.75+ 再说——baseline 都没立稳就上 agent 是常见踩坑。

### 8.9 RAG 评估体系

不评估的 RAG 就是在盲调参。**评估体系是 RAG 系统的"测试工程师"**。

![rag_eval_pipeline](images/rag_eval_pipeline.png)

#### 8.9.1 评估三件套

RAG 系统评估有三大类指标：

- **检索侧指标**：Recall@k、MRR、NDCG、Hit Rate、MAP——只看 retriever 召回准不准。
- **生成侧指标**：Faithfulness、Answer Relevancy——只看 LLM 答得好不好。
- **端到端指标**：Context Precision、Context Recall、Answer Correctness——综合"检索+生成"。

#### 8.9.2 检索侧指标

假设 ground truth 相关文档集合 = `G`，模型召回 top-k 集合 = `R_k`。

| 指标 | 公式 | 含义 | 何时用 |
|---|---|---|---|
| **Recall@k** | `|G ∩ R_k| / |G|` | ground truth 中有多少被召回了 | 关心"漏没漏" |
| **Precision@k** | `|G ∩ R_k| / k` | 召回了多少是相关的 | 关心"准不准" |
| **Hit Rate** | `1[recall_at_k 大于 0]` | top-k 至少命中一个 | 1 个正样本时退化 |
| **MRR** | `mean(1/rank_of_first_relevant)` | 第一个相关文档的倒数排名 | 关心"第一的位置" |
| **NDCG@k** | 按相关性等级加权 | 位置 + 等级 | 多级相关性（完美/部分/无关） |
| **MAP** | 各 query AP 平均 | 平均精度均值 | 多相关文档 |

**RAG 系统里最常用**：Recall@10（端到端）+ NDCG@10（rerank 前后比较）。`Recall@10 大于 0.9` 通常是工程目标。

#### 8.9.3 RAGAS 四大指标

RAGAS（RAG Assessment, Es et al. 2023）= 用 LLM-as-judge 自动评估 RAG 系统的开源框架，**不需要人工标注 ground truth**。核心 4 个指标：

##### 1. Faithfulness（忠实度）

**回答中的事实声称是否被检索到的上下文支持**——这是防 hallucination 的核心指标。

$$
\mathrm{Faithfulness} = \frac{\text{被上下文支持的声称数}}{\text{回答中总声称数}}
$$

计算步骤（LLM-as-judge）：

1. 把回答拆成 N 个 atomic 声称（"投标人需具备 ISO 9001 认证"）；
2. 对每个声称，让 LLM 判定"能否从 retrieved context 推出"；
3. 计数 = 支持数 / 总声称数。

**RAG 系统的"硬指标"**——`Faithfulness` 小于 0.7 通常意味着系统在"自说自话"。

##### 2. Answer Relevancy（答案相关性）

**回答是否切题**——即使内容真实，如果不切题也是失败。

$$
\mathrm{AnswerRelevancy} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{cos}(\vec{q}, \vec{q_i^{artificial}})
$$

计算步骤：

1. 用 LLM 从 answer 反向生成 N 个"假想原 question"；
2. 算这 N 个 fake question 与原 query 的 embedding 余弦相似度；
3. 取平均。

直观：能反推出原 query 的 answer 才是切题的。

##### 3. Context Precision（上下文精度）

**检索到的 context 中相关 chunk 占多少**——衡量 retriever 的"信噪比"。

$$
\mathrm{ContextPrecision} = \frac{\sum_{k=1}^{K} (\mathrm{Precision@k}) \cdot v_k}{\sum_{k=1}^{K} v_k}
$$

`v_k ∈ {0, 1}` 表示第 k 个 chunk 是否相关；带 precision 加权 = MAP 的变体。

低 Context Precision 的代价：喂给 LLM 大量噪声 → 容易 hallucinate，且增加 token 成本。

##### 4. Context Recall（上下文召回率）

**ground truth 答案所需的全部信息是否被检索到**——需要 reference answer。

$$
\mathrm{ContextRecall} = \frac{|\text{GT 中能从 context 推出的句子数}|}{|\text{GT 总句子数}|}
$$

低 Context Recall = retriever 漏召，LLM 答不全。

**RAGAS 五大指标 = 上述 4 + Answer Correctness**（与 ground truth 答案做 F1 + 语义相似度加权）。

#### 8.9.4 ARES（Automated REE-val System, Saad-Falcon et al. 2024）

RAGAS 依赖 LLM-as-judge，**评估 LLM 的偏见会传递**。ARES 进一步：

- 用 LLM 生成合成 QA 对（用 domain corpus）；
- 用一个**轻量分类器**（LM-based judge）训练去打分；
- 优点：成本低、可重复、对 judge-LLM 依赖小；
- 缺点：仍需要"领域语料"做起点。

**适用**：CI/CD 集成大规模回归；不适合早期 prototype。

#### 8.9.5 RGB 基准（Chen et al. 2024）

RGB（Retrieval-Augmented Generation Benchmark）把 RAG 能力拆成四个维度：

| 能力 | 测试什么 | 失败模式 |
|---|---|---|
| **噪声鲁棒性** | 检索到的段落里混大量无关段 | 答非所问 / 被噪声带偏 |
| **否定拒绝** | 文档里没有答案时模型应拒绝 | "幻觉"地编造答案 |
| **信息整合** | 需要从多段拼接才能答 | 漏掉某段 / 答不全 |
| **反事实鲁棒性** | 检索文档里包含与参数化知识矛盾的"反事实" | 模型相信参数知识，忽略 context |

**反事实鲁棒性**是 2024 年起新强调的能力——许多模型"嘴上说"会按文档答，但碰到"地球是方的（反事实）"会强行答"对"或"不对"。

#### 8.9.6 CRUD-RAG（Lyons et al. 2024）

按 CRUD 四象限评估：

| | Create | Read | Update | Delete |
|---|---|---|---|---|
| 测试什么 | 创造性生成 | 事实问答 | 矛盾信息更新 | 信息遗忘 |

工程意义：很多 RAG 系统只测了 Read（事实问答），忽略 Update（同一事实多版本）和 Delete（旧版本过滤）。

#### 8.9.7 评测集怎么造

没有 ground truth 时三种方法：

1. **人工标注**：50–500 条 QA 对，金标准。代价高但最可信。
2. **LLM 合成**：用 GPT-4 / Claude / Qwen3 从文档生成 QA 对，**人工抽 20% 校验**。成本低 10×。
3. **Self-Instruct / Evol-Instruct**：从少量种子 QA 演化出大量变体（加约束、改问法、反向问）。

**关键技巧**：

- QA 对要**覆盖所有检索失败模式**（含错字、跨章节、表格、否定）；
- ground truth 答案要 **cite 到具体 chunk**（不然 Context Recall 算不出来）；
- 评估时**用与生成不同的 LLM**做 judge（用 Qwen3 当生成、用 GPT-4 当 judge 减少自评偏差）。

### 8.10 工程落地

> **本节结合实际招投标 RAG 知识库项目**（FastAPI + 阿里云百炼 embedding + ES 双索引：文档级检索 + 问答 chunk 索引）。每一条都是项目里踩过或避开的坑。

![rag_es_hybrid_arch](images/rag_es_hybrid_arch.png)

> 上图展示了 RAG 系统的典型工程架构：左侧"离线索引"链路（解析 → chunking → embedding → 入库）和右侧"在线服务"链路（query 理解 → 混合检索 → rerank → 生成）通过 ES 双索引 + Redis 缓存 + LLM 网关衔接。红色为瓶颈点（embedding API、LLM 网关），需要重点限流与重试。

#### 8.10.1 ES 混合索引设计

**核心思路**：**两个独立的物理索引，各自管一种检索**。

```
index_a (bm25_chunks)
  ├── body: text (IK 分词)
  ├── title: text (boost 3x)
  ├── doc_id: keyword
  ├── chunk_id: keyword
  └── metadata: nested (department, time, doc_type)

index_b (qa_chunks)
  ├── body: text
  ├── body_vector: dense_vector (1024 维, hnsw, m=16, ef_construction=100)
  ├── doc_id: keyword
  ├── chunk_id: keyword
  └── metadata: nested
```

**为什么不放一个 index**：BM25 字段和 dense_vector 字段的倒排 / 图索引结构差异大，分开物理索引可独立调参、独立扩缩容。**查询时用 cross-index search + RRF 融合**。

**ES dense_vector 索引语法**（8.x）：

```json
PUT /qa_chunks
{
  "mappings": {
    "properties": {
      "body": { "type": "text", "analyzer": "ik_max_word" },
      "body_vector": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine",
        "index_options": {
          "type": "int8_hnsw",
          "m": 16,
          "ef_construction": 100
        }
      },
      "doc_id": { "type": "keyword" },
      "chunk_id": { "type": "keyword" },
      "department": { "type": "keyword" },
      "create_time": { "type": "date" }
    }
  }
}
```

- `int8_hnsw` 是 **ES 8.12+ 默认**的量化方案，4× 内存节省 + 几乎无精度损失。
- `similarity` 三选一：cosine / dot_product / l2_norm；中文 embedding 一般用 cosine。

**BM25 字段调优**（中文）：

```json
"settings": {
  "analysis": {
    "analyzer": {
      "ik_smart": { "type": "ik_smart" },
      "ik_max_word": { "type": "ik_max_word" }
    }
  }
}
```

- 索引时用 `ik_max_word`（细粒度，召回多）；
- 查询时用 `ik_smart`（粗粒度，精度高）；
- 标点、停用词走 IK 自身词表。

**RRF 融合查询**：

```json
GET /bidding_docs/_search
{
  "retriever": {
    "rrf": {
      "retrievers": [
        {
          "standard": {
            "query": {
              "bool": {
                "must": [
                  { "multi_match": { "query": "投标人 资质 要求", "fields": ["title^3", "body"] } }
                ],
                "filter": [
                  { "term": { "department": "招标办" } }
                ]
              }
            }
          }
        },
        {
          "knn": {
            "field": "body_vector",
            "query_vector": [0.12, -0.34, ...],
            "k": 20,
            "num_candidates": 100,
            "filter": [
              { "term": { "department": "招标办" } }
            ]
          }
        }
      ],
      "rank_window_size": 50,
      "rank_constant": 60
    }
  },
  "size": 10,
  "_source": ["title", "body", "doc_id", "chunk_id"]
}
```

`rank_window_size` = 每路保留多少候选；`rank_constant` = RRF 的 k。

#### 8.10.2 metadata 过滤与分区

**多租户 + 权限隔离必备**：

- `department`（部门）、`permission_group`（权限组）、`doc_type`（文件类型）、`project_id`（项目 ID）、`create_time`（入库时间）；
- 在 knn 和 BM25 查询里都加 `filter` 子句；
- ES 的 filter 是**预计算 + 缓存 bitset**，几乎无开销。

**分区策略**：

- **按 tenant 分索引**（`bidding_tenant_a`、`bidding_tenant_b`）：隔离最强，但索引数爆炸；
- **共享索引 + tenant filter**：99% 场景选这个；
- **按时间分索引**（滚动索引 `bidding-2026-08`、`bidding-2026-09`）：适合时间序列数据。

#### 8.10.3 增量更新与索引版本管理

**三种粒度**：

| 粒度 | 实现 | 代价 | 何时用 |
|---|---|---|---|
| **整文档重索引** | 文档变化时删旧写新 | 重算 embedding，O(全文档大小) | 文档小 / 改动频繁 |
| **chunk 级别** | 只对变化的 chunk 重建 | 需要 chunk 稳定 ID | 文档大、改动局部 |
| **混合** | 文档级 fingerprint 判定是否变化 | 100% 准确、O(变化部分) | **生产推荐** |

**embedding 换版管理**：

```
v1 (bge-large-zh-v1.5, 1024d) → v2 (Qwen3-Embedding-0.6B, 1024d)
  ↓
新字段 body_vector_v2
  ↓
双写期：同时存在 v1 / v2
  ↓
读切换：query 走 v2 索引
  ↓
下架 v1
```

**关键**：embedding 模型换版**必须全量重算**——维度、归一化、语义空间都可能变，老向量不能直接搬到新模型下。

#### 8.10.4 批量 embedding 的限流与重试

**阿里云百炼 / OpenAI / Cohere** 等云端 embedding API 都有 QPS 限速。生产经验：

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingClient:
    def __init__(self, qps_limit=50, batch_size=10):
        self.semaphore = asyncio.Semaphore(qps_limit)
        self.batch_size = batch_size

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry_error_callback=lambda r: r.result()
    )
    async def embed_batch(self, texts):
        async with self.semaphore:
            resp = await call_api(texts)  # QPS 限流由 semaphore 保证
            if resp.status_code == 429:
                raise RateLimitError()
            return resp.json()["vectors"]
```

**核心实践**：

- batch_size：8–32（太小浪费、太大单次失败成本高）；
- QPS：按服务商限制的 80% 设置（留 20% 退避余量）；
- 429 退避：指数退避（2s → 4s → 8s → 16s → 32s → 60s 上限）；
- 超时：单次请求 30s 上限，3 次失败后落 dead-letter queue，**不要无限重试阻塞流水线**；
- 进度：写 checkpoint，崩溃后可断点续跑（按 doc_id 已经成功的跳过）。

#### 8.10.5 缓存

三层缓存：

| 层 | 缓存什么 | 命中率 | 实现 |
|---|---|---|---|
| **embedding 缓存** | 文本 → 向量 | 高（重复文本） | Redis / 内存 LRU |
| **query 缓存** | query → 完整回答 | 中 | Redis + TTL |
| **语义缓存** | 相似 query → 同一回答 | 中 | GPTCache / 自建向量检索 |

![rag_caching_layer](images/rag_caching_layer.png)

**GPTCache**（Zilliz 开源）：把 query 也 embedding 进向量库，相似的 query 命中时直接返回缓存的回答。**对重复/近似 query 多的场景**（客服、FAQ）非常有效。

```python
from gptcache import Cache
from gptcache.adapter.api import init_similar_from_data

cache = Cache()
cache.init(
    pre_embedding_func=embedding_for_cache,  # 用便宜的小模型
    data_manager=cache_data_manager,
    similarity_threshold=0.85,  # 0.85+ 视为同一问题
)
```

**风险**：语义缓存可能把"看似相同但实际不同"的 query 错误合并；**阈值要严格 + 加关键词白名单**。

#### 8.10.6 中文场景特有问题

- **分词**：必须用 IK / HanLP / Jieba；ES 默认 standard analyzer 对中文是逐字切。IK 的 `ik_max_word` 索引 / `ik_smart` 查询组合是事实标准。
- **专有名词**：招标术语、药名、缩写必须加自定义词表；IK 的 `ext_dict` / `ext_stopwords` 维护。
- **中英混排**："SQL 数据库优化"、"API 接口设计"——IK 能识别英文连续串，但注意大小写。
- **同义词**："中标" ↔ "得标" ↔ "成交"、"招标人" ↔ "采购人"——加 synonym token filter。
- **繁简**："資料" vs "资料"——加 `stconvert` 插件。
- **数字 / 单位**："100 万" vs "1000000" vs "一百万"——做归一化预处理。
- **多音字**："行长"（háng vs háng）——靠上下文 embedding 解决，BM25 可能误召。
- **全角 / 半角**："（" vs "("、"１" vs "1"——做归一化。

#### 8.10.7 引用溯源与可解释性

合规场景（金融、医疗、法律、招投标）的**强需求**。每条 chunk 必须能反查回原文档。

```json
{
  "chunk_id": "c_001",
  "doc_id": "招标公告_2025_xxx",
  "doc_title": "XX 项目招标公告",
  "doc_url": "https://...",
  "doc_page": 5,
  "section": "第 3 章 投标人资格要求",
  "body": "投标人须具备 ISO 9001 认证...",
  "score": 0.89
}
```

前端展示：

```
[答案] 投标人须具备 ISO 9001 质量管理体系认证。

引用：
[1] 招标公告_2025_xxx · 第 3 章 · 第 5 页 (相关度 0.89)
[2] ...
```

**Anthropic Contextual Retrieval**（2024-09）报告：**Contextual Embeddings 让 top-20 失败率从 5.7% 降到 3.7%（降 35%）；加 Contextual BM25 降到 2.9%（49%）；再加 Rerank 降到 1.9%（67%）**。代价：每个 chunk 多一次 LLM 推理，但配合 prompt caching 成本可控制在 **$1.02 / 百万文档 token**（数据来源：anthropic.com/engineering/contextual-retrieval）。

#### 8.10.8 监控指标

上线后必看的运行时指标：

![rag_monitoring_dashboard](images/rag_monitoring_dashboard.png)

| 指标 | 目标 | 告警阈值 |
|---|---|---|
| **query QPS** | - | 突增 3x 排查 |
| **embedding P99 延迟** | 小于 200 毫秒 | 大于 500 毫秒 |
| **ES 检索 P99** | 小于 300 毫秒 | 大于 1 秒 |
| **Rerank P99** | 小于 500 毫秒 | 大于 1.5 秒 |
| **端到端 P99** | 小于 3 秒 | 大于 8 秒 |
| **Faithfulness（在线抽样）** | 大于 0.8 | 小于 0.65 |
| **空召回率** | 小于 5% | 大于 15% |
| **Token 单次平均** | 小于 2K | 大于 5K |
| **embedding API 429 比例** | 小于 0.5% | 大于 5% |

#### 8.10.9 性能与成本基准（2025 年典型）

| 规模 | 文档数 | chunk 数 | 索引存储 | 单次检索成本 | 单次回答成本 |
|---|---|---|---|---|---|
| 小 | 1K | 5K | 小于 100 MB | 小于 0.001 美元 | 小于 0.01 美元 |
| 中 | 100K | 1M |–10 GB | 0.001–0.005 美元 | 0.01–0.05 美元 |
| 大 | 10M | 100M |–1 TB（+ 量化） | 0.005–0.02 美元 | 0.05–0.3 美元 |
| 超大 | 1B+ | 10B+ | 数十 TB 与 IVF-PQ | 0.02 美元以上 | 走专用硬件 |

### 8.11 高频面试题

> 25+ 题，每题 3–8 行答案。面试按重要性选背。

**Q1. 什么是 RAG？为什么需要 RAG 而不是只用 LLM？**
A. RAG = 检索增强生成。先用 retriever 从外部知识库找相关文档，把 chunk 拼进 prompt 再让 LLM 生成。原因：减少幻觉、知识可更新（不用重训）、私域数据可用、强可溯源。

**Q2. RAG vs 微调怎么选？**
A. RAG 解决"我不知道这件事"（注入事实）；SFT 解决"我应该这样回答"（风格/格式/指令）。先 RAG 再视情况加 SFT。

**Q3. Naive / Advanced / Modular / Agentic RAG 的区别？**
A. Naive = 朴素 pipeline；Advanced = 每一步做优化（query 改写、混合检索、rerank）；Modular = 模块化可拼装；Agentic = LLM 自己决定路由、检索、反思。演进方向：自动化程度递增、灵活性递增。

**Q4. 文档解析的难点有哪些？**
A. PDF 是"画在白板上的指令"——文字乱序、表格是矢量、公式是图、扫描件要 OCR。复杂版面要"版面分析 + 阅读顺序恢复"。表格和公式要保留结构，HTML/Markdown 形式最佳。**解析错误是 RAG 最大单一失败源**。

**Q5. 为什么说 chunking 决定 RAG 上限？**
A. 召回靠 embedding 与 query 的相似度，**如果 chunk 不完整、错位、含噪声**，再好的 embedding 也召不回正确信息。常见错法：固定切分切碎表格、递归切分跨章节、语义切分阈值难调。

**Q6. RecursiveCharacterTextSplitter 的核心思想？**
A. 按优先级递减的分隔符列表递归切——优先按段落，再按句子，再按空格，再按字符。中文必须把 `"。"`、`"！"`、`"？"`、`"；"` 加进 separators。

**Q7. Late Chunking 是什么？为什么有效？**
A. Jina 2024。**先 embed 整篇文档（用长上下文模型），再按边界切块、pooling**。这样每个 chunk 的向量都"见过整篇文档"，解决"代词 / 指代丢失上下文"问题。需要 8K+ 上下文 embedding 模型。

**Q8. Small-to-Big 是什么？**
A. 检索用细粒度（句子级，命中率高），生成用粗粒度（段落级，LLM 看得全）。BGE-M3 天然支持 dense + sparse + ColBERT 三种粒度。

**Q9. Sentence-BERT 比 BERT 直接用 [CLS] 强在哪？**
A. 孪生网络 + mean pooling。CLS 是分类头，**对句级语义"无感"**；mean pooling 把所有 token 向量取平均（去 padding），更适合句级任务。STS 任务上提升 10+ 点。

**Q10. 怎么选 embedding 模型？**
A. 中文 → Qwen3-Embedding-0.6B / BGE-M3；英文 → text-embedding-3-large / voyage-4 / cohere v4；离线 → nomic-embed-v1.5；看 MTEB 子任务分数而不是总平均；优先选带 Matryoshka（可截断维度）。

**Q11. MTEB 榜单怎么正确读？**
A. 关注你要做的子任务（retrieval / clustering / classification）、语言子集（英文 MTEB v2 / C-MTEB / Multilingual）、上下文长度、许可证（避 CC-BY-NC）、维度（低维省存储）。

**Q12. 什么是 Matryoshka Representation Learning？**
A. Kusupati 2022。训练时同时优化多个子维度（32/64/128/.../1024），让低维向量也接近最优。**一个模型可按存储预算截断**。

**Q13. cosine vs dot product vs L2 怎么选？**
A. 文本 RAG 默认 cosine，**L2 归一化后 dot = cosine**（更快）；dot 用于已归一化向量；L2 几乎不用。BGE / Qwen3 / OpenAI 输出已归一化，**直接用 dot**。

**Q14. HNSW 为什么是向量检索事实标准？**
A. 查询 / 插入 `O(log N)`、亚毫秒延迟、单机能跑 1M–10M、召回 95–99%。**边界**：内存大、频繁更新会降召回、需定期重建。算法来源：Malkov & Yashunin 2018。

**Q15. HNSW 三个核心参数？**
A. M（每节点最大边数，默认 16–32，越大 recall 越高、内存越大）；efConstruction（构建 beam 宽度，100–200，越大图质量越高、构建越慢）；efSearch（查询 beam 宽度，50–500，是查询时唯一可调的"recall 旋钮"）。

**Q16. 量化（PQ / SQ / IVF-PQ）什么时候用？**
A. 向量超过单机 RAM 时。**SQ (int8)** 4× 压缩、几乎无损、最简单；**PQ** 几十倍压缩、损失中等；**IVF-PQ** 是十亿级向量的标配。ES 8.12+ 默认 `int8_hnsw`。

**Q17. 主流向量库怎么选？**
A. 已有 ES / 重 BM25 → ES + RRF；高并发 / 亿级 → Milvus / Qdrant；轻量 / 原型 → Chroma / LanceDB；跟业务库同库 → pgvector；想要开箱混合 → Qdrant。

**Q18. BM25 公式写一下？**
A. `score(D,Q) = Σ IDF(q_i) · f(q_i,D)·(k1+1) / (f(q_i,D) + k1·(1-b+b·|D|/avgdl))`。`k1=1.2`、`b=0.75` 是默认；`b=0` 不归一化、`b=1` 完全归一化。

**Q19. 为什么需要混合检索？**
A. 单一 dense 检索对精确关键词（产品编号、错误码、缩写）容易漏；单一 BM25 不理解语义。**混合 = 互相补漏**——dense 抓语义、BM25 抓精确。

**Q20. RRF 为什么不用分数加和？**
A. BM25 分数 0–50、cosine 0–1，**量纲不同**。即使做 min-max 归一化，归一化窗口小（top-10）时会污染。RRF 用 rank 天然规避——`RRF_score(d) = Σ 1/(k + rank_r(d))`，`k=60` 默认。

**Q21. Cross-Encoder 为什么比 Bi-Encoder 准？**
A. Bi-Encoder 分别编码后做点积，**信息瓶颈在两段向量拼接处**；Cross-Encoder 把 query+doc 拼一起过 transformer，每层 self-attention **充分交互**（"投标人"和"资质"之间的关系被精确建模）。代价：每对都要过一次模型，**慢**。所以用在 Rerank 阶段（top-50–200）。

**Q22. Self-RAG 的四种 reflection token？**
A. `[Retrieve]`（是否需要检索，yes/no/continue）、`[IsRel]`（检索段是否相关，relevant/irrelevant）、`[IsSup]`（生成内容是否被上下文支持，fully/partially/no）、`[IsUse]`（整体有用性，1–5）。

**Q23. GraphRAG 解决什么问题？代价是什么？**
A. 解决"跨文档主题 / 全局汇总"（如"前 5 大客户是谁"传统向量 RAG 答不全）。代价：**索引阶段两次 LLM pass + 社区摘要**，百万 token 语料用 GPT-4 数百到上千美元。**LazyGraphRAG** 把成本压到向量 RAG 同级（**0.1%**），适合中小规模。

**Q24. Lost in the Middle 现象是什么？**
A. Liu 2023。LLM 对 context 中间位置的内容**利用率显著低于首尾**——U 形曲线。20 篇文档时中间位置准确率从–75% 跌到–53%。**RAG 通过 top-k + Rerank 天然规避**（相关 chunk 排首尾）。

**Q25. 长上下文能否替代 RAG？**
A. 不能。三道坎：成本（1M token 输入 1–10 美元 vs RAG 约 0.02 美元）、延迟（prefill 30–60 秒 vs RAG 小于 1 秒）、lost-in-the-middle。小于 200K token 知识库 + 提示词技巧（Q+Docs+Q）可省 RAG；更大规模 RAG 仍是唯一选择。

**Q26. RAGAS 四个核心指标怎么算？**
A. Faithfulness = 被上下文支持的声称数 / 总声称数；Answer Relevancy = 反推问题与原 query 的平均 cos 相似度；Context Precision = 相关 chunk 的 MAP（precision@k 加权）；Context Recall = GT 中能从 context 推出的句子数 / GT 总句子数。

**Q27. 怎么构建 RAG 评测集？**
A. 三种方法：人工标注（金标准）、LLM 合成（GPT-4 / Claude 从文档生成 QA）+ 抽 20% 人工校验、Self-Instruct / Evol-Instruct 演化变体。**关键**：QA 要覆盖所有失败模式、答案要 cite 到具体 chunk、judge LLM 要与生成 LLM 不同。

**Q28. embedding 模型换版怎么迁移？**
A. 必须**全量重算**——维度、归一化、语义空间都可能变。步骤：建新字段 `body_vector_v2`；双写期同时存在；读切到 v2 索引；下架 v1。生产经验：换版期跑 A/B（新旧两路同时检索，人工评估哪个准）。

**Q29. 招投标 RAG 知识库项目里，最难的三件事？**
A. (1) **扫描件识别**：招标公告常含盖章签字扫描，需 PaddleOCR + 版面分析；(2) **表格还原**：评分办法、报价表常是表格，需 MinerU / TextIn 保留 HTML 结构；(3) **跨文档引用**：投标人对多个项目做对比，需文档级 metadata + parent_id 串联。

**Q30. 怎么排查 RAG 系统"答非所问"？**
A. 按 RAGAS 四指标定位：(1) Context Recall 低 → 检索漏召，调 BM25 / 换 embedding / 加 query 改写；(2) Context Precision 低 → 检索召回太多噪声，加强 filter / 加 Rerank；(3) Faithfulness 低 → LLM 忽略 context，调 prompt / 加 "只基于以下内容回答" / 换更好的 LLM；(4) Answer Relevancy 低 → 切题问题，常见于 prompt 模板设计差。

### 8.12 三句话总结

1. **RAG = 检索 + 生成**，解决 LLM 幻觉、时效、私域、可溯源四大问题；RAG 与 SFT 不是替代而是组合——RAG 给事实、SFT 给格式。**长上下文 ≠ 替代 RAG**——三道坎：成本（1M token 输入约 1–10 美元 vs RAG 约 0.02 美元）、延迟（prefill 数十秒 vs RAG 小于 1 秒）、lost-in-the-middle。
2. **决定 RAG 质量的是链路里最弱的一环**——文档解析、chunking、embedding 选型、检索策略、rerank、prompt 模板，**每一段都不能省**。经验优先级：解析质量 优先于 embedding 选型，再优先于 chunking，再优先于 检索策略，最后才是 prompt 技巧。
3. **未来 2–3 年关键趋势**：Qwen3-Embedding / 8B 级别 LLM 当 embedding 底座让 dense 检索逼近闭源 SOTA；Late Chunking 与 Contextual Retrieval 几乎"零成本"补齐 chunk 上下文损失；Agentic RAG 让复杂查询走 LLM 决策的图式 workflow，但 80% 场景仍是 Advanced RAG pipeline 解决。

### 8.13 延伸阅读

#### 论文 / 综述

- **Lewis et al., 2020**. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". NeurIPS 2020. （RAG 开山之作）
- **Gao et al., 2024**. "Retrieval-Augmented Generation for Large Language Models: A Survey". ACM Computing Surveys. （最权威综述，覆盖 Naive/Advanced/Modular/Agentic）
- **Karpukhin et al., 2020**. "Dense Passage Retrieval for Open-Domain Question Answering". （DPR，开创 dense retrieval）
- **Khattab & Zaharia, 2020**. "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT". （late interaction 奠基）
- **Malkov & Yashunin, 2018**. "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs". IEEE TPAMI. （HNSW 原始论文）
- **Günther et al., 2024**. "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models". （Late Chunking 论文）
- **Cormack, Clarke, Büttcher, 2009**. "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods". SIGIR 2009. （RRF 原始论文）
- **Liu et al., 2023**. "Lost in the Middle: How Language Models Use Long Contexts". TACL. （Lost-in-the-Middle 现象）
- **Edge et al., 2024**. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization". （GraphRAG 微软论文）
- **Asai et al., 2023**. "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection". ICLR 2024.
- **Yan et al., 2024**. "Corrective Retrieval Augmented Generation". （CRAG）
- **Kusupati et al., 2022**. "Matryoshka Representation Learning". NeurIPS 2022.
- **Es et al., 2023**. "RAGAS: Automated Evaluation of Retrieval Augmented Generation". （RAGAS 框架）
- **Anthropic, 2024**. "Introducing Contextual Retrieval". （Contextual Retrieval 报告，reduction 49% / 67%）

#### 博客 / 实战

- **Lilian Weng 博客** "LLM-powered Autonomous Agents"（agentic RAG 框架图）：lilianweng.github.io/posts/2023-06-23-agent/
- **Jay Alammar "The Illustrated Retrieval Transformer"**：jalammar.github.io/illustrated-retrieval-transformer/
- **Elasticsearch 官方文档 - kNN search**：elastic.co/guide/en/elasticsearch/reference/current/knn-search.html
- **Qwen3 Embedding 官方博客**（含 MTEB 分数表）：qwenlm.github.io/blog/qwen3-embedding/
- **Anthropic 官方"Contextual Retrieval"**：anthropic.com/engineering/contextual-retrieval
- **Microsoft Research "LazyGraphRAG"**：microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost
- **Hugging Face 文档 - RAG**：huggingface.co/docs/transformers/model_doc/rag

#### GitHub 仓库（可直接跑）

- **LangChain**（含 RAG 模板）：github.com/langchain-ai/langchain
- **LlamaIndex**（含 GraphRAG 集成）：github.com/run-llama/llama_index
- **RAGAS 评测框架**：github.com/explodinggradients/ragas
- **GraphRAG 微软官方**：github.com/microsoft/graphrag
- **LightRAG (HKUDS)**：github.com/HKUDS/LightRAG
- **Jina Late Chunking 示例**：github.com/jina-ai/late-chunking
- **BGE 官方仓库**（含 BGE-M3）：github.com/FlagOpen/FlagEmbedding
- **Qwen3-Embedding 官方**：github.com/QwenLM/Qwen3-Embedding
- **GPTCache（语义缓存）**：github.com/zilliztech/GPTCache
- **Verba（Weaviate 官方 RAG UI）**：github.com/weaviate/Verba

#### 中文资源

- **Jina AI 中文博客**："深入理解延迟分块：实质与误解"：jina.ai/zh-CN/news/what-late-chunking-really-is-and-what-its-not-part-ii
- **阿里云百炼 embedding 文档**：help.aliyun.com/zh/model-studio/developer-reference/embedding-api-details
- **腾讯云 VectorDB / 智能检索 文档**：cloud.tencent.com/product/vdb
- **Elasticsearch 中文社区**：elastic.co/cn/

#### 落地参考

- **FastAPI + ES + 阿里云百炼** 的招投标 RAG 实战（中型项目，–50 万 chunk）：推荐参照 LlamaIndex 的 `SubQuestionQueryEngine` + LangGraph 编排多步 agentic 工作流。
- **BGE-M3 + ES 8.12+ int8_hnsw** 是当前中文 RAG 性价比最高组合之一。
- **Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B**（同一基座）是 2025 年中文 RAG 默认选型。
