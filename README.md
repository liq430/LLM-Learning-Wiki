# 大模型技术知识 Wiki

一份**系统化、图文并茂**的中文大模型技术知识库，覆盖从数学基础、Transformer 架构、预训练、微调、对齐、量化，到推理部署、RAG、Agent、Prompt 工程、评估、工程落地与安全合规、面试冲刺、前沿动态的完整知识体系。

> 面向「算法工程师 + 产品经理 / 业务人员」双受众：既讲透原理与公式，也讲清落地与面试要点。

## ✨ 特色

- 🧠 **系统化**：16 章主章节 + 1 篇附录册，从数学到前沿逐层递进，可作为课程教学大纲使用
- 📖 **图文并茂**：82 张配图，含自绘架构图 + 经典论文图解（来源已在文中标注）
- 🧮 **公式完整**：用 Markdown + LaTeX 块级公式呈现手算与推导过程
- 🛠️ **工程视角**：每一章都兼顾原理与开源实现 / 部署建议
- 🎯 **面试友好**：单独一章「面试冲刺：手算与高频百问」可作为求职复习手册

## 📑 目录

| # | 章节 | 一句话简介 | 阅读量 |
|---|------|-----------|--------|
| 📍 全文导读<br>`00` | [00. 导读与学习路线图](chapters/00-导读与学习路线图.md) | 全书结构 + 按岗位选路 + 考前排期 + 缩写速查；先把这一章读完再决定其它章节的阅读顺序。 | ⏱️ 47 分钟 |
| 🧮 数学打底<br>`01` | [01. 数学基础（大模型面试够用版）](chapters/01-数学基础.md) | 线性代数、概率统计、信息论与优化理论的「面试最小子集」——含公式推导与手算示例。 | ⏱️ 1 小时 44 分钟 |
| 🏗️ 模型架构<br>`02` | [02. Transformer 与模型架构](chapters/02-Transformer%20与模型架构.md) | Attention / MHA / MQA / GQA / RoPE / RMSNorm / SwiGLU 等现代 Transformer 模块逐项拆解。 | ⏱️ 1 小时 21 分钟 |
| 🌱 预训练<br>`03` | [03. 预训练 Pre-Training](chapters/03-预训练%20Pre-Training.md) | 数据清洗、Tokenizer 与 BPE、训练目标（MLM / LM / 扩散）、分布式并行与 Zero 系列策略。 | ⏱️ 36 分钟 |
| 🎯 微调<br>`04` | [04. 微调与参数高效微调](chapters/04-微调与参数高效微调.md) | LoRA / QLoRA / Adapter / Prefix / Prompt Tuning，及其在 SFT 与跨任务迁移的工程实践。 | ⏱️ 29 分钟 |
| ⚖️ 对齐<br>`05` | [05. 对齐与强化学习](chapters/05-对齐与强化学习.md) | RLHF → DPO → GRPO 的奖励建模、偏好优化与策略约束，含在线/离线 RL 算法对比。 | ⏱️ 30 分钟 |
| 🪶 量化压缩<br>`06` | [06. 量化与模型压缩](chapters/06-量化与模型压缩.md) | PTQ / QAT / GPTQ / AWQ / SmoothQuant / KV-Cache 量化，含 4-bit/8-bit 部署的精度-速度权衡。 | ⏱️ 17 分钟 |
| 🚀 推理部署<br>`07` | [07. 推理与部署](chapters/07-推理与部署.md) | vLLM / SGLang / TensorRT-LLM / Continuous Batching / PagedAttention / Speculative Decoding。 | ⏱️ 2 小时 35 分钟 |
| 📚 RAG<br>`08` | [08. RAG 检索增强生成](chapters/08-RAG%20检索增强生成.md) | Embedding / Chunk / Retriever（BM25 / DPR / Hybrid）/ Re-Rank / Generator，含评估与索引设计。 | ⏱️ 1 小时 23 分钟 |
| 🤖 Agent<br>`09` | [09. 智能体](chapters/09-智能体.md) | ReAct / Plan-and-Execute / Reflection / Tool-use / Memory，含 LangGraph / AutoGen 等框架对比。 | ⏱️ 3 小时 1 分钟 |
| ✨ Prompt<br>`10` | [10. Prompt 工程与推理引导](chapters/10-Prompt%20工程与推理引导.md) | 基础提示、CoT、ToT、Self-Consistency、Function Calling、Structured Output 与工具编排。 | ⏱️ 1 小时 22 分钟 |
| 📏 评估<br>`11` | [11. 大模型评估体系](chapters/11-大模型评估体系.md) | 通用基准（MMLU / GSM8K / HumanEval）、领域基准、对齐评估、人类评估与 LLM-as-Judge。 | ⏱️ 1 小时 9 分钟 |
| 🛡️ 落地合规<br>`12` | [12. 工程落地与安全合规](chapters/12-工程落地与安全合规.md) | 模型选型、成本估算、A/B 灰度、Prompt 工程、安全合规（内容审核、隐私、越狱防护）。 | ⏱️ 1 小时 21 分钟 |
| 🎤 面试冲刺<br>`13` | [13. 面试冲刺：手算与高频百问](chapters/13-面试冲刺手算与高频百问.md) | 高频百问、手算题、推理 trick 速查、面试套话模板，含难度分级。 | ⏱️ 1 小时 22 分钟 |
| 🗺️ 资源进阶<br>`14` | [14. 学习资源与进阶路线](chapters/14-学习资源与进阶路线.md) | 论文清单、博客清单、课程清单、开源实现，按「入门 → 进阶 → 专题」分级。 | ⏱️ 47 分钟 |
| 🔭 前沿<br>`15` | [15. 前沿动态与趋势](chapters/15-前沿动态与趋势.md) | SOTA 模型、缩放定律新趋势、合成数据、长上下文、多模态、Agent 化方向。 | ⏱️ 1 小时 1 分钟 |
| 🧰 附录速查<br>`99` | [99. 附录](chapters/99-附录.md) | 公式表、缩写表、论文索引、参数默认值速查、评测基准表。 | ⏱️ 1 小时 3 分钟 |

> 阅读量按字数 + 500 字/分钟粗估，仅供参考。每章开头都有「本章目录」可跳转。

## 🌐 在线阅读版

### ① 文档站（GitHub Pages，可全文搜索）：https://liq430.github.io/LLM-Learning-Wiki/

- 支持**全文搜索**、侧边栏多级目录、暗色模式、代码复制、标题锚点
- 站点锚点与 GitHub **完全一致**，两边收藏的链接可以互相打开
- 推送到 `main` 分支后由 GitHub Actions 自动构建发布

### ② 腾讯文档版（公式与表格渲染更佳）

已发布到腾讯文档的在线版（含渲染后的公式与图片），入口：

**总导航：https://docs.qq.com/aio/DVXJXQnV1VGpMbGRN**

## 📦 本仓库结构

```
LLM-Learning-Wiki/
├── README.md                 # 本文件
├── mkdocs.yml                # MkDocs 配置（Material 主题）
├── mkdocs_hooks.py           # 让站点锚点与 GitHub 保持一致的 hook
├── requirements.txt          # 文档站构建依赖
├── docs/                     # MkDocs 文档源（index.md + 指向 chapters/images 的软链）
├── chapters/                 # 全部章节（每章一个 Markdown 文件）
├── images/                   # 章节引用的配图
└── .github/workflows/
    └── deploy-docs.yml       # GitHub Pages 自动部署
```

## 🔍 如何阅读

### 方式 ① 直接在 GitHub 浏览
- 直接打开 `chapters/` 下任意 `.md` 文件即可阅读
- 每章开头的「本章目录」可以点击跳转

### 方式 ② 直接打开文档站（推荐）

**https://liq430.github.io/LLM-Learning-Wiki/** —— 无需任何安装，支持全文搜索与多级目录。

### 方式 ③ 本地构建可搜索的文档站
```bash
# 安装依赖（仅首次）
pip install -r requirements.txt

# 本地预览（默认 http://localhost:8000）
mkdocs serve

# 构建静态站点（输出到 site/）
mkdocs build
```

### 方式 ④ GitHub Pages 自行部署
仓库已配 `.github/workflows/deploy-docs.yml`，推送到 `main` 分支后自动构建并发布。
在仓库 **Settings → Pages → Source** 选择 **GitHub Actions** 即可启用。

## 📋 知识地图（俯瞰）

| # | 章节 | 一句话简介 |
|---|------|-----------|
| `00` | [00. 导读与学习路线图](chapters/00-导读与学习路线图.md) | 全书结构 + 按岗位选路 + 考前排期 + 缩写速查；先把这一章读完再决定其它章节的阅读顺序。 |
| `01` | [01. 数学基础（大模型面试够用版）](chapters/01-数学基础.md) | 线性代数、概率统计、信息论与优化理论的「面试最小子集」——含公式推导与手算示例。 |
| `02` | [02. Transformer 与模型架构](chapters/02-Transformer%20与模型架构.md) | Attention / MHA / MQA / GQA / RoPE / RMSNorm / SwiGLU 等现代 Transformer 模块逐项拆解。 |
| `03` | [03. 预训练 Pre-Training](chapters/03-预训练%20Pre-Training.md) | 数据清洗、Tokenizer 与 BPE、训练目标（MLM / LM / 扩散）、分布式并行与 Zero 系列策略。 |
| `04` | [04. 微调与参数高效微调](chapters/04-微调与参数高效微调.md) | LoRA / QLoRA / Adapter / Prefix / Prompt Tuning，及其在 SFT 与跨任务迁移的工程实践。 |
| `05` | [05. 对齐与强化学习](chapters/05-对齐与强化学习.md) | RLHF → DPO → GRPO 的奖励建模、偏好优化与策略约束，含在线/离线 RL 算法对比。 |
| `06` | [06. 量化与模型压缩](chapters/06-量化与模型压缩.md) | PTQ / QAT / GPTQ / AWQ / SmoothQuant / KV-Cache 量化，含 4-bit/8-bit 部署的精度-速度权衡。 |
| `07` | [07. 推理与部署](chapters/07-推理与部署.md) | vLLM / SGLang / TensorRT-LLM / Continuous Batching / PagedAttention / Speculative Decoding。 |
| `08` | [08. RAG 检索增强生成](chapters/08-RAG%20检索增强生成.md) | Embedding / Chunk / Retriever（BM25 / DPR / Hybrid）/ Re-Rank / Generator，含评估与索引设计。 |
| `09` | [09. 智能体](chapters/09-智能体.md) | ReAct / Plan-and-Execute / Reflection / Tool-use / Memory，含 LangGraph / AutoGen 等框架对比。 |
| `10` | [10. Prompt 工程与推理引导](chapters/10-Prompt%20工程与推理引导.md) | 基础提示、CoT、ToT、Self-Consistency、Function Calling、Structured Output 与工具编排。 |
| `11` | [11. 大模型评估体系](chapters/11-大模型评估体系.md) | 通用基准（MMLU / GSM8K / HumanEval）、领域基准、对齐评估、人类评估与 LLM-as-Judge。 |
| `12` | [12. 工程落地与安全合规](chapters/12-工程落地与安全合规.md) | 模型选型、成本估算、A/B 灰度、Prompt 工程、安全合规（内容审核、隐私、越狱防护）。 |
| `13` | [13. 面试冲刺：手算与高频百问](chapters/13-面试冲刺手算与高频百问.md) | 高频百问、手算题、推理 trick 速查、面试套话模板，含难度分级。 |
| `14` | [14. 学习资源与进阶路线](chapters/14-学习资源与进阶路线.md) | 论文清单、博客清单、课程清单、开源实现，按「入门 → 进阶 → 专题」分级。 |
| `15` | [15. 前沿动态与趋势](chapters/15-前沿动态与趋势.md) | SOTA 模型、缩放定律新趋势、合成数据、长上下文、多模态、Agent 化方向。 |
| `99` | [99. 附录](chapters/99-附录.md) | 公式表、缩写表、论文索引、参数默认值速查、评测基准表。 |

## 关于

- 内容为系统学习大模型技术的知识整理，配图部分来自公开技术博客（已在文中标注出处），部分为本地绘制。
- 在线版与本地版内容一致；腾讯文档在线版的公式与表格排版更佳，文档站的搜索体验更佳。
- 章节标题锚点统一按 GitHub 的 `github-slugger` 规则生成（968 个标题已逐条校验，0 差异），
  因此在 GitHub 复制的标题链接可以直接粘到文档站，反之亦然。
- 配套脚本：`prepare_github.py`（从 `parts/*.md` 原始素材生成本仓库结构）。
- 最近更新：2026-09-02 · 本次修复对应 GitHub commit [`2c6184a`](https://github.com/liq430/LLM-Learning-Wiki/commit/2c6184a)
