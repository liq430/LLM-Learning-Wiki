#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描 Wiki 全站中「用反引号包裹、但其实是数学公式」的行内代码跨度。

背景
----
MathJax 只识别 $...$ / $$...$$；反引号内的内容按行内代码原样输出，
希腊字母（π、θ）、上下标（x_t、T²）会以普通 Unicode 显示，公式不渲染。
本脚本用于找出这些"伪装成代码的公式"，供 convert_inline_math.py 批量转换。

关键设计（两个已踩过的坑）
--------------------------
1. 行内代码跨度必须按「同长度反引号定界符」逐行配对解析，
   不能用全局 `([^`]+?)` 正则 —— 后者会跨 $$ 公式块把两段文本错误连成一个匹配。
2. snake_case 英文标识符（log_softmax / n_kv_head / top_p）的下划线会被
   SUBS 规则误判为数学下标。规则：下划线 >=2 个，或下划线前缀长度 >1 → 判代码。
3. 强数学特征（LaTeX 命令 / 希腊字母 / 数学算符）优先级高于 code-like 排除，
   否则 O(n log n)、T² 会被 func(args) / 赋值规则误判成代码。
"""
import re
import os
import sys
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))

# 行内代码跨度：定界符为 n 个连续反引号，内容为不含 n 连反引号的最短串
CODE_SPAN = re.compile(r'(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)')

LATEX = (r'\\(?:log|ln|lg|exp|sqrt|frac|sum|prod|int|partial|theta|alpha|beta|gamma|delta'
         r'|lambda|sigma|mu|pi|epsilon|varepsilon|tau|phi|varphi|psi|omega|rho|nu|eta|zeta'
         r'|max|min|arg|lim|inf|sup|cdot|cdots|ldots|times|div|pm|leq|geq|neq|approx|sim'
         r'|propto|in|notin|subset|subseteq|cup|cap|rightarrow|leftarrow|Rightarrow'
         r'|Leftarrow|Leftrightarrow|mid|hat|bar|tilde|vec|mathbb|mathcal|mathrm|textbf'
         r'|text|begin|end|left|right|nabla|partial|top|perp|binom|det|dim|ker|rank'
         r'|argmax|argmin|softmax|logsumexp|operatorname)')
GREEK = r'[αβγδεζηθικλνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩϑϒΦ]'
MATHOP = r'[∑∏∫∮∝∞∂∇√∈∉∋∧∨∩∪⊂⊃⊆⊇≤≥≠≈≡≅∼≃→←⇒⇐⇔↔↦↑↓±∓×÷⋅∗∘∙‖°′″²³⁰¹⁴⁵⁶⁷⁸⁹⁻]'
SUPERS = r'\^[\w\{<\\(]'
SUBS = r'_[\w\{<\\(]'

# 纯 snake_case / camelCase ASCII 标识符
IDENT = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')

PY_KW = re.compile(r'\b(?:def|class|import|from|return|if|else|elif|for|while|try|except'
                   r'|with|as|in|is|not|and|or|None|True|False|self|print|lambda|yield'
                   r'|assert|raise|global|pass|break|continue|del|async|await|elif)\b')
LIB = re.compile(r'\b(?:torch|np|numpy|pd|pandas|tf|transformers|F|nn|os|sys|json|re|math'
                 r'|random|time|sklearn|matplotlib|plt|jax|trl|peft|accelerate|deepspeed'
                 r'|vllm|flash_attn|datasets|tokenizers|sentence_transformers|faiss'
                 r'|transformer|AutoModel|AutoTokenizer|BitsAndBytes)\s*[.\[]')
# 配置赋值：左边是小写 snake_case 标识符（>=3 字符），右边是单个值（不含空格）
#   命中：weight_decay = λ / kl_coef = 0 / temperature = 0.7
#   不命中：C_t = f_t · C_{t-1} + i_t · C̃_t（大写开头，且右边含空格）
#   不命中：η = 0.1 / τ = 1.0（左边是希腊字母，属数学）
CFG = re.compile(r'^[a-z][a-z0-9_]{2,}\s*=\s*\S+$')


# 组合附加符号（v̂ 的 ̂）与修饰符上下标（ᵀ / ₀ / ⁻¹）：MathJax 吃不下，必须翻成 LaTeX
COMBINING_RE = re.compile(r'[\u0300-\u036f]')
MODIFIER_RE = re.compile(r'[\u1d40-\u1daa\u1d2c-\u1d6a\u02b0-\u02e0\u2070-\u209f]')


def strong_math(s):
    """强数学特征：LaTeX 命令 / 希腊字母 / 数学算符 / 组合符 / 修饰符上下标
    —— 优先级最高，能压过所有 code-like 排除规则"""
    return bool(re.search(LATEX, s) or re.search(GREEK, s) or re.search(MATHOP, s)
                or COMBINING_RE.search(s) or MODIFIER_RE.search(s))


def math_sub_sup(s):
    """上下标是否属于数学形态（而非 snake_case 标识符）"""
    m = re.search(SUBS, s) or re.search(SUPERS, s)
    if not m:
        return False
    if IDENT.match(s) and not re.search(GREEK + '|' + MATHOP, s):
        # 纯 ASCII 标识符：下划线 >=2 个（n_kv_head）或前缀长度 >1（top_p）都判代码
        parts = re.split(r'[_\^]', s)
        if s.count('_') >= 2:
            return False
        if len(parts[0]) > 1:
            return False
    return True


def has_math(s):
    if strong_math(s):
        return True
    if math_sub_sup(s):
        return True
    if re.search(r'\bO\s*\(', s):
        return True
    return False


# 编程函数（区别于 max / min / exp / log 这类数学算子）
PROG_FUNC = re.compile(
    r'\b(?:round|int|float|str|bool|len|list|dict|set|tuple|range|zip|enumerate'
    r'|format|print|append|split|join|replace|sorted|hash|clamp|concat|reshape'
    r'|view|detach|cpu|cuda|item|numpy|torch)\s*\(')
# 路径形式 a/b/c（MCP 方法名 notifications/tools/list_changed）
PATH_SLASH = re.compile(r'^[A-Za-z_][\w]*(?:/[A-Za-z_][\w]*)+$')
# 配置项布尔字面量 requires_approval: true
KV_BOOL = re.compile(r':\s*(?:true|false|null)\s*$', re.I)
# 无空格的 k=v 出现两次以上：temperature=0.7 + top_p=0.9
# （有空格的 `d_k = d_v = ...` 是数学关系，不受影响）
MULTI_KV = re.compile(r'[A-Za-z_]\w*=\S')


def is_code_like(s):
    # 强数学特征优先，避免 O(n log n)、T² 被下面的 func(args) / 赋值规则误判
    if strong_math(s):
        return False
    if PY_KW.search(s) or LIB.search(s) or CFG.match(s):
        return True
    t = s.strip()
    if t.startswith(('{', '[')) and t.endswith(('}', ']')):
        return True
    if re.match(r'^[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)+$', s):      # a.b.c 路径
        return True
    # func(args)：但参数里带组合附加符号 / 希腊字母的是数学（sqrt(v̂)、f(θ)）
    if re.match(r'^[A-Za-z_][\w]*\([^)]*\)$', s) \
            and not re.search(r'[\u0300-\u036f]|' + GREEK, s):
        return True
    if re.search(r'\.(py|md|json|yaml|yml|txt|csv|sh|ipynb|bin|safetensors|gguf|parquet|js|ts|lock)$', s):
        return True
    if re.search(r'(?:^|\s)--?[A-Za-z][\w-]*$', s):                 # CLI flag
        return True
    if re.search(r'^\|[\s:|-]+\|$', s):
        return True
    if PATH_SLASH.match(s) or KV_BOOL.search(s) or PROG_FUNC.search(s):
        return True
    if len(MULTI_KV.findall(s)) >= 2:          # temperature=0.7 + top_p=0.9
        return True
    if IDENT.match(s) and not math_sub_sup(s):                      # 纯标识符且非数学下标
        return True
    return False


def strip_combining(s):
    """剥离组合附加符号，只留基字符——用于按"视觉长度"判断"""
    return ''.join(c for c in s if not (0x0300 <= ord(c) <= 0x036f))


def safe_to_convert(s):
    bare = strip_combining(s)               # v̂ 视觉长度算 1，不是 2
    if not (1 <= len(bare) <= 200):
        return False
    if '$' in s or '\n' in s or '`' in s:
        return False
    if re.search(r'^[A-Za-z]+\s*=\s*(?:None|True|False)', s):
        return False
    if len(bare) == 1:
        # 单字符只放行希腊字母与数学算符：它们不可能是代码，
        # 但等宽正体显示很突兀（λ vs λ）。ASCII 单字母（w / d / x）
        # 与伪代码、CLI 参数高度混叠，一律保留反引号。
        if re.fullmatch(GREEK + '|' + MATHOP, bare):
            return True
        # 带组合附加符号的单字符（v̂ 二阶动量 / X̃ 增广矩阵 / m̂ 一阶动量）
        # 一定不是代码标识符，放行
        if COMBINING_RE.search(s):
            return True
        return False
    if len(bare) == 2 and not strong_math(s):
        return False
    return True


def iter_code_spans(path):
    """yield (lineno, span_text)，自动跳过围栏代码块"""
    in_fence = False
    with open(path, encoding='utf-8') as fh:
        for i, ln in enumerate(fh, 1):
            st = ln.strip()
            if st.startswith('```') or st.startswith('~~~'):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for m in CODE_SPAN.finditer(ln):
                yield i, m.group(2)


def main():
    files = sorted(
        [os.path.join('chapters', f) for f in os.listdir(os.path.join(ROOT, 'chapters')) if f.endswith('.md')]
    )
    if os.path.isdir(os.path.join(ROOT, 'docs')):
        files += sorted(os.path.join('docs', f) for f in os.listdir(os.path.join(ROOT, 'docs'))
                        if f.endswith('.md'))

    odd_lines, cand, all_spans = [], [], 0
    per_file = collections.Counter()
    for rel in files:
        path = os.path.join(ROOT, rel)
        with open(path, encoding='utf-8') as fh:
            lines = fh.read().split('\n')
        in_fence = False
        for i, ln in enumerate(lines, 1):
            st = ln.strip()
            if st.startswith('```') or st.startswith('~~~'):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if ln.count('`') % 2:
                odd_lines.append((rel, i, ln[:130]))
        for i, seg in iter_code_spans(path):
            all_spans += 1
            if is_code_like(seg):
                continue
            if has_math(seg) and safe_to_convert(seg):
                cand.append((rel, i, seg))
                per_file[rel] += 1

    print(f'扫描文件数      : {len(files)}')
    print(f'行内代码跨度总数: {all_spans}')
    print(f'奇数反引号行    : {len(odd_lines)}')
    for rel, i, txt in odd_lines:
        print(f'   [奇数] {rel}:L{i}  {txt}')
    print(f'\n【候选：反引号包裹的数学公式】{len(cand)} 处')
    for rel, c in per_file.most_common():
        print(f'   {c:4d}  {rel}')
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    print(f'\n--- 候选明细（前 {limit} 条）---')
    for rel, i, s in cand[:limit]:
        print(f'   {os.path.basename(rel)}:L{i}  {s!r}')
    return cand


if __name__ == '__main__':
    main()
