#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复「定界符错乱」：反引号与 $ 交叉配对（如 `tools/list$ ... $tools/call`）。

成因
----
早前的批量转换脚本按 `` `([^`]+?)` `` 全局正则处理，遇到同一行内已有 `$...$`
公式时会把定界符切错，留下「反引号开头、$ 结尾」的畸形对。后果是 MathJax
与 Markdown 各自吞掉半段，正文被当成公式、公式被当成代码。

本脚本用精确字符串匹配（不是正则）逐条修复，替换前强制校验原串存在且唯一。

  用法：python3 fix_delimiters.py          # 干跑
        python3 fix_delimiters.py --write  # 落盘
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# (文件, 原串, 新串) —— 全部为精确字面量
FIXES = [
    # ── 01 数学基础：π_ref 相关 ───────────────────────────────
    # 注：`$π_ref` 在 01 章出现 2 次（L1035 / L1689），`$K = vocab_size$` 在 01b
    #     出现 2 次（L230 / L507），故用带上下文的长串唯一锁定。
    ('chapters/01-数学基础.md', '，`q = π_ref$（参考模型）', '，$q = π_{\\mathrm{ref}}$（参考模型）'),
    ('chapters/01-数学基础.md', '参考模型 $π_ref` 概率低', '参考模型 $π_{\\mathrm{ref}}$ 概率低'),
    ('chapters/01-数学基础.md', '训练时 `p$ 是**数据分布**', '训练时 $p$ 是**数据分布**'),
    ('chapters/01-数学基础.md', '所以 $H(p)` 是一个**常数**', '所以 $H(p)$ 是一个**常数**'),
    ('chapters/01-数学基础.md',
     '若反向写成 `D_KL(π_ref ‖ π)$',
     '若反向写成 $D_{\\mathrm{KL}}(π_{\\mathrm{ref}} \\| π)$'),
    ('chapters/01-数学基础.md',
     '则要求 π 覆盖 $π_ref` 的所有模式', '则要求 π 覆盖 $π_{\\mathrm{ref}}$ 的所有模式'),

    # ── 01b 机器学习基础 ─────────────────────────────────────
    ('chapters/01b-机器学习与深度学习基础.md',
     '在 loss 加 `λ/2 ‖θ‖²$', '在 loss 加 $λ/2 ‖θ‖²$'),
    ('chapters/01b-机器学习与深度学习基础.md', '得 $λθ`', '得 $λθ$'),
    ('chapters/01b-机器学习与深度学习基础.md',
     '公式 `y = F(x) + x$', '公式 $y = F(x) + x$'),
    ('chapters/01b-机器学习与深度学习基础.md',
     '$∂y/∂x = ∂F/∂x + I`', '$∂y/∂x = ∂F/∂x + I$'),
    # 纯代码常量名：K 是数学符号，vocab_size 是代码常量，拆开处理
    ('chapters/01b-机器学习与深度学习基础.md',
     '它（$K = vocab_size$，`y` 是 next token id）', '它（$K =$ `vocab_size`，`y` 是 next token id）'),
    ('chapters/01b-机器学习与深度学习基础.md',
     '**LLM 预训练就是它**，$K = vocab_size$，$p_y$',
     '**LLM 预训练就是它**，$K =$ `vocab_size`，$p_y$'),

    # ── 05 对齐与强化学习 ────────────────────────────────────
    ('chapters/05-对齐与强化学习.md',
     '`σ(β · Δ) = 0.9$', '$σ(β · Δ) = 0.9$'),
    ('chapters/05-对齐与强化学习.md', '$Δ = 21.97`', '$Δ = 21.97$'),

    # ── 07 推理与部署 ────────────────────────────────────────
    ('chapters/07-推理与部署.md', 'FLOPs 约 `4 n^2 d$', 'FLOPs 约 $4 n^2 d$'),
    ('chapters/07-推理与部署.md', '每次 $2 n^2 d`', '每次 $2 n^2 d$'),
    ('chapters/07-推理与部署.md', '到 `γ = 20$ 时', '到 $γ = 20$ 时'),
    ('chapters/07-推理与部署.md', '右图固定 $γ = 5` 与', '右图固定 $γ = 5$ 与'),

    # ── 08 RAG：美元金额必须转义，否则被 MathJax 当公式 ──────
    ('chapters/08-RAG 检索增强生成.md',
     '**$1.02 / 百万文档 token**', '**\\$1.02 / 百万文档 token**'),

    # ── 09 智能体：MCP 方法名 / 占位符，一律回到反引号 ───────
    ('chapters/09-智能体.md', '、`#4$ 彼此无依赖', '、`#4` 彼此无依赖'),
    ('chapters/09-智能体.md', '；$#3` 等 `#1`', '；`#3` 等 `#1`'),
    ('chapters/09-智能体.md',
     '`tools/list$ 拉取工具清单', '`tools/list` 拉取工具清单'),
    ('chapters/09-智能体.md', '→ $tools/call$ 调用', '→ `tools/call` 调用'),
    ('chapters/09-智能体.md', '→ $resources/read$ 读资源', '→ `resources/read` 读资源'),
    ('chapters/09-智能体.md', '→ $ping$ 探活', '→ `ping` 探活'),
    ('chapters/09-智能体.md',
     'Server 可通过 $notifications/tools/list_changed` 通知',
     'Server 可通过 `notifications/tools/list_changed` 通知'),
    ('chapters/09-智能体.md',
     'Host 通过 MCP 的 `tools/list$ 拿到工具定义',
     'Host 通过 MCP 的 `tools/list` 拿到工具定义'),
    ('chapters/09-智能体.md',
     '（OpenAI 的 $tools` 数组', '（OpenAI 的 `tools` 数组'),
    ('chapters/09-智能体.md',
     '/ Anthropic 的 `tools$）', '/ Anthropic 的 `tools`）'),
    ('chapters/09-智能体.md',
     '→ 模型输出 $tool_calls$ →', '→ 模型输出 `tool_calls` →'),
    ('chapters/09-智能体.md',
     'Host 通过 MCP 的 $tools/call$ 去执行', 'Host 通过 MCP 的 `tools/call` 去执行'),
    ('chapters/09-智能体.md',
     '结果翻译回 $role="tool"` 消息回灌', '结果翻译回 `role="tool"` 消息回灌'),
    ('chapters/09-智能体.md', '用 `grep$ 定位关键词', '用 `grep` 定位关键词'),
    ('chapters/09-智能体.md', '（$limit` 200–500 行）', '（`limit` 200–500 行）'),
    ('chapters/09-智能体.md',
     'Host 调 MCP 的 `tools/list$ 拉清单', 'Host 调 MCP 的 `tools/list` 拉清单'),
    ('chapters/09-智能体.md',
     'Host 调 MCP 的 $tools/call` →', 'Host 调 MCP 的 `tools/call` →'),
    # 配置项，回滚为反引号
    ('chapters/09-智能体.md', '$requires_approval: true$', '`requires_approval: true`'),

    # ── 12 工程落地 ──────────────────────────────────────────
    ('chapters/12-工程落地与安全合规.md',
     '`C_per GPU$ 为单卡可承载并发', '$C_{\\mathrm{per\\ GPU}}$ 为单卡可承载并发'),
    ('chapters/12-工程落地与安全合规.md', '，$S` 为安全冗余系数', '，$S$ 为安全冗余系数'),
    ('chapters/12-工程落地与安全合规.md',
     '| $C_{\\mathrm{req}},avg$ |', '| $C_{\\mathrm{req,avg}}$ |'),

    # ── 01 数学基础：配置项回滚 ───────────────────────────────
    ('chapters/01-数学基础.md',
     '所有 2D 矩阵用 $weight_decay = λ$', '所有 2D 矩阵用 `weight_decay` = $λ$'),

    # ══════════════════════════════════════════════════════════
    # 二、含中文的公式：把为规避 Markdown 转义而用中文替代的符号改回真符号
    #     （作者原写「除以」「加」「大于」，是因为 Markdown 会吞掉 / + >）
    # ══════════════════════════════════════════════════════════
    ('chapters/01b-机器学习与深度学习基础.md',
     '`θ ← −η(Adam更新 + λθ)`',
     '$\\theta \\leftarrow -\\eta(\\text{Adam 更新} + \\lambda\\theta)$'),
    ('chapters/07-推理与部署.md', '`1 除以 n_heads`', '$1 / n_{\\mathrm{heads}}$'),
    ('chapters/07-推理与部署.md',
     '`n_groups 除以 n_heads`', '$n_{\\mathrm{groups}} / n_{\\mathrm{heads}}$'),
    ('chapters/07-推理与部署.md',
     '`L (d_c 加 d_h^R) p`', '$L (d_c + d_h^R) p$'),
    ('chapters/07-推理与部署.md',
     '`Θ(N^2 d^2 除以 M)`', '$\\Theta(N^2 d^2 / M)$'),
    ('chapters/07-推理与部署.md',
     '`128^2 × 2 字节 = 32 KB`', '$128^2 \\times 2$ 字节 $= 32\\,\\mathrm{KB}$'),
    ('chapters/07-推理与部署.md',
     '`E2E 延迟 = TTFT + TPOT × 输出长度`',
     '$\\text{E2E 延迟} = \\text{TTFT} + \\text{TPOT} \\times \\text{输出长度}$'),
    ('chapters/07-推理与部署.md',
     '`QPS × 输入长度 × 2N`',
     '$\\mathrm{QPS} \\times \\text{输入长度} \\times 2N$'),
    ('chapters/07-推理与部署.md',
     '`QPS × 输出长度`', '$\\mathrm{QPS} \\times \\text{输出长度}$'),
    ('chapters/08-RAG 检索增强生成.md',
     '`1[recall_at_k 大于 0]`',
     '$\\mathbf{1}[\\,\\mathrm{recall\\_at\\_k} > 0\\,]$'),
    ('chapters/12-工程落地与安全合规.md', '| `Q_月` |', '| $Q_{\\text{月}}$ |'),

    # ── 02 Transformer：^(ᵀ) 写法应为转置，规范化脚本无法自动处理 ──
    ('chapters/02-Transformer 与模型架构.md',
     '$QK^(^{\\top}) = XX^(^{\\top})$', '$QK^{\\top} = XX^{\\top}$'),

    # ── 01：反引号把一条完整公式截成两段，括号永远配不上 ──
    ('chapters/01-数学基础.md',
     '注意力 `O = softmax(QK^T` 除以 `sqrt(d)) V` 的梯度推导',
     '注意力 $O = \\mathrm{softmax}(QK^T / \\sqrt{d})\\,V$ 的梯度推导'),

    # ══════════════════════════════════════════════════════════
    # 三、渲染期才发现的问题（verify_local_render.py + diag_render.py 抓出）
    # ══════════════════════════════════════════════════════════
    # 07 章：W 已带 ^{UK}，再跟 ^T 触发 Double exponent，必须整体加括号
    ('chapters/07-推理与部署.md',
     '直接与 $W^{UK}_i^T$ 相乘', '直接与 $(W^{UK}_i)^{T}$ 相乘'),

    # 09 章：这行有 4 空格缩进 → 被解析成缩进代码块，MathJax 跳过 <pre><code>；
    # 同时纯 Unicode 公式（$λ$）不含 ^ _ { } \，也不会被 hook 的哨兵保护。
    # 两头都补：去缩进（与上方 $$ 块对齐）+ 希腊字母改 LaTeX 命令。
    # 注：去缩进与希腊字母转命令分两条规则写。长串规则负责去缩进 + Δt/λ，
    #     短串规则补 α——短串在两种缩进下都成立，幂等且不会被长串吃掉。
    ('chapters/09-智能体.md',
     '$\\lambda$ 控制衰减速度，$α$ 平衡语义相关性与新鲜度',
     '$\\lambda$ 控制衰减速度，$\\alpha$ 平衡语义相关性与新鲜度'),
]


def main():
    write = '--write' in sys.argv
    done, missing, dup, applied = 0, [], [], 0

    # 按文件分组后每个文件只读一次、写一次。
    # 早先按 FIXES 原顺序边走边切文件的写法，会让同一文件被反复读写，
    # 结果是「明明改成功了却报 done=0」，统计完全不可信。
    by_file = {}
    for rel, old, new in FIXES:
        by_file.setdefault(rel, []).append((old, new))

    for rel, rules in by_file.items():
        path = os.path.join(ROOT, rel)
        text = open(path, encoding='utf-8').read()
        touched = False
        for old, new in rules:
            cnt = text.count(old)
            if cnt == 0:
                # 幂等：新串已存在说明这条规则此前已应用过
                if text.count(new) > 0:
                    applied += 1
                    continue
                missing.append((rel, old))
                continue
            if cnt > 1:
                dup.append((rel, old, cnt))
                continue
            text = text.replace(old, new)
            done += 1
            touched = True
        if write and touched:
            open(path, 'w', encoding='utf-8').write(text)

    print(f'{"[已写入]" if write else "[干跑]"} 本次替换 {done} 处，此前已应用 {applied} 条，'
          f'规则总数 {len(FIXES)}')
    if missing:
        print(f'\n未找到原串（{len(missing)} 条，需人工确认）：')
        for rel, old in missing:
            print(f'   {os.path.basename(rel)}  {old!r}')
    if dup:
        print(f'\n原串不唯一（{len(dup)} 条，已跳过）：')
        for rel, old, c in dup:
            print(f'   {os.path.basename(rel)}  出现 {c} 次  {old!r}')
    return 1 if (missing or dup) else 0


if __name__ == '__main__':
    sys.exit(main())
