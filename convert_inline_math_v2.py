#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二轮：把「用反引号包裹、其实是数学公式」的行内代码跨度转成 $...$。

与第一轮（convert_inline_math.py）的区别
----------------------------------------
1. 行内代码跨度按「同长度反引号定界符」逐行配对解析，不再用全局 `([^`]+?)` 正则
   —— 后者会跨 $$ 公式块把两段文本错误连成一个匹配（第一轮因此漏改了大量公式）。
2. 收紧 snake_case 误判：下划线 >=2 个（n_kv_head）或下划线前缀长度 >1（top_p）
   一律判为代码标识符，不转。
3. 强数学特征（LaTeX 命令 / 希腊字母 / 数学算符）优先级高于 code-like 排除规则，
   否则 O(n log n)、T² 会被 func(args) / 赋值规则误判成代码。
4. 本轮按「桶」分阶段转换，用 --buckets 控制，便于逐桶验证：
      F  纯数学（默认，可直接转）
      B  含组合附加符号（m̂ / X̃）—— 需先折叠为 \hat{} \tilde{}
      C  含修饰符上下标（ᵀ / ₀ / ⁶）—— 需先转成 ^{} _{}
      A  含中文（需人工改写，脚本默认跳过并给出清单）
   被排除：D 代码特征（__sync_warpgroup / route(...) -> x）、E 标识符列表（W_hh, W_xh）

用法
----
    python3 convert_inline_math_v2.py            # 干跑，只报告
    python3 convert_inline_math_v2.py --write    # 落盘
    python3 convert_inline_math_v2.py --buckets F C B
"""
import re
import os
import sys
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_SPAN = re.compile(r'(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)')

CJK = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]')
COMBINING = re.compile(r'[\u0300-\u036f\u1ab0-\u1aff\u20d0-\u20f0]')
MODIFIER = re.compile(r'[\u1d40-\u1daa\u1d2c-\u1d6a\u02b0-\u02e0\u2070-\u209f]')
CODE_HINT = re.compile(r'(?:->|::|__\w|proj\b|module\b|tensor\b|dtype\b|device\b|shape\b|api\b|API\b|\.py\b|\.json\b)')
LIST_IDS = re.compile(r'^[A-Za-z_][\w]*(?:,\s+[A-Za-z_][\w]*|\s*,\s*\.\.\.)+')

sys.path.insert(0, ROOT)
from scan_backticks import (is_code_like, has_math, safe_to_convert,  # noqa: E402
                            CODE_SPAN as _CS)


# ── Unicode → LaTeX 规范化 ──────────────────────────────────────────
# MathJax 能直接吃大部分 Unicode 数学符号（× · − ≈ Σ θ …），但吃不下两类：
#   1) 组合附加符号（m + U+0302）—— 会拆成两个字符
#   2) 修饰符字母（ᵀ U+1D40）、下标数字（₀ U+2080）—— 不在数学字符表内，显示豆腐块
# 下面把它们显式翻译成 LaTeX 命令。
COMBINING_MAP = {
    '\u0302': 'hat',      # ̂  组合抑扬符
    '\u0303': 'tilde',    # ̃  组合波浪符
    '\u0304': 'bar',      # ̄  组合长音符
    '\u0307': 'dot',      # ̇  组合上点
    '\u0308': 'ddot',     # ̈  组合分音符
}
COMBINING_RE = re.compile(
    r'([A-Za-z\u0370-\u03ff\u0391-\u03a9\u03b1-\u03c9])'
    r'([\u0302\u0303\u0304\u0307\u0308])'
)

# 预组合字符（不是"基字符+组合符"，需单独映射）
PRECOMPOSED = {
    'ŷ': r'\hat{y}', 'ŝ': r'\hat{s}', 'ŵ': r'\hat{w}', 'ẑ': r'\hat{z}',
    'ĉ': r'\hat{c}', 'ĥ': r'\hat{h}', 'ĵ': r'\hat{j}', 'ŵ': r'\hat{w}',
    'ǹ': r'\dot{n}', 'à': r'\grave{a}',
}
# 修饰符/上下标字母数字
MODIFIER_MAP = {
    'ᵀ': r'^{\top}', 'ᵁ': r'^{U}', 'ᵂ': r'^{W}', 'ᵃ': r'^{a}', 'ᵇ': r'^{b}',
    'ᶜ': r'^{c}', 'ᵈ': r'^{d}', 'ᵉ': r'^{e}', 'ᵍ': r'^{g}', 'ʰ': r'^{h}',
    'ⁱ': r'^{i}', 'ʲ': r'^{j}', 'ᵏ': r'^{k}', 'ˡ': r'^{l}', 'ᵐ': r'^{m}',
    'ⁿ': r'^{n}', 'ᵒ': r'^{o}', 'ᵖ': r'^{p}', 'ʳ': r'^{r}', 'ˢ': r'^{s}',
    'ᵗ': r'^{t}', 'ᵘ': r'^{u}', 'ᵛ': r'^{v}', 'ˣ': r'^{x}', 'ʸ': r'^{y}',
    'ᶻ': r'^{z}', '⁺': r'^{+}', '⁻': r'^{-}', '⁼': r'^{=}', '⁽': r'^{(}',
    '⁾': r'^{)}', '⁰': r'^{0}', '¹': r'^{1}', '²': r'^{2}', '³': r'^{3}',
    '⁴': r'^{4}', '⁵': r'^{5}', '⁶': r'^{6}', '⁷': r'^{7}', '⁸': r'^{8}',
    '⁹': r'^{9}',
    '₀': r'_{0}', '₁': r'_{1}', '₂': r'_{2}', '₃': r'_{3}', '₄': r'_{4}',
    '₅': r'_{5}', '₆': r'_{6}', '₇': r'_{7}', '₈': r'_{8}', '₉': r'_{9}',
    '₊': r'_{+}', '₋': r'_{-}', '₍': r'_{(}', '₎': r'_{)}',
    '★': r'^{\star}', '☆': r'^{\star}', '∗': r'^{*}',
}
_MOD_KEYS = sorted(MODIFIER_MAP, key=len, reverse=True)
_MOD_RE = re.compile('|'.join(re.escape(k) for k in _MOD_KEYS))
_PRE_RE = re.compile('|'.join(re.escape(k) for k in sorted(PRECOMPOSED, key=len, reverse=True)))


# 上标数字 -> 普通数字，用于把 ⁻¹ 这类组合并成一个上标
SUP_DIGIT = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
             '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
SUP_SIGN_DIGITS = re.compile(r'([⁻⁺])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)')


def _apply_modifiers(body):
    """替换修饰符/上下标字符，并避免出现 `^^{}` `__{}` 这种重复记号。"""
    out, i, n = [], 0, len(body)
    while i < n:
        hit = None
        for k in _MOD_KEYS:                      # 长键优先
            if body.startswith(k, i):
                hit = k
                break
        if hit is None:
            out.append(body[i]); i += 1; continue
        tex = MODIFIER_MAP[hit]
        prev = out[-1] if out else ''
        # 前面已经是 ^ 或 _，只补花括号内容，去掉重复的 ^ / _
        if tex.startswith('^') and prev == '^':
            out.append('{' + tex[2:-1] + '}')
        elif tex.startswith('_') and prev == '_':
            out.append('{' + tex[2:-1] + '}')
        else:
            out.append(tex)
        i += len(hit)
    return ''.join(out)


def normalize(body):
    """把 math mode 里 MathJax 吃不下 / 会显示错的 Unicode 翻成 LaTeX 命令。"""
    # 0) ⁻¹ / ⁺² 这类"上标符号 + 上标数字"先合并成一个上标，否则会拆成 ^{-}^{1}
    body = SUP_SIGN_DIGITS.sub(
        lambda m: ('^{-' if m.group(1) == '⁻' else '^{+}')
                  + ''.join(SUP_DIGIT[c] for c in m.group(2)) + '}',
        body)
    # 1) 组合附加符号（可能连续多个，循环直到稳定）
    prev = None
    while prev != body:
        prev = body
        body = COMBINING_RE.sub(lambda m: f'\\{COMBINING_MAP[m.group(2)]}{{{m.group(1)}}}', body)
    # 2) 预组合字符
    body = _PRE_RE.sub(lambda m: PRECOMPOSED[m.group(0)], body)
    # 3) 修饰符 / 上下标字母数字
    body = _apply_modifiers(body)
    return body


def bucket_of(s):
    if CJK.search(s):
        return 'A'
    if COMBINING.search(s):
        return 'B'
    if MODIFIER.search(s):
        return 'C'
    if CODE_HINT.search(s):
        return 'D'
    if LIST_IDS.search(s):
        return 'E'
    return 'F'


def iter_file(path):
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    return text


def md_files():
    out = []
    for d in ('chapters', 'docs'):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith('.md'):
                out.append(os.path.join(d, f))
    return out


def convert_file(path, wanted, write):
    text = iter_file(path)
    lines = text.split('\n')
    in_fence = False
    changed = []
    skipped = collections.defaultdict(list)
    new_lines = []
    for i, ln in enumerate(lines, 1):
        st = ln.strip()
        if st.startswith('```') or st.startswith('~~~'):
            in_fence = not in_fence
            new_lines.append(ln)
            continue
        if in_fence:
            new_lines.append(ln)
            continue

        out, last = [], 0
        for m in CODE_SPAN.finditer(ln):
            body, delim = m.group(2), m.group(1)
            b = bucket_of(body)
            take = (b in wanted
                    and not is_code_like(body)
                    and has_math(body)
                    and safe_to_convert(body))
            if take:
                out.append(ln[last:m.start()])
                out.append('$' + normalize(body) + '$')
                changed.append((i, body, normalize(body)))
            else:
                out.append(ln[last:m.start()])
                out.append(m.group(0))
                if b in wanted:
                    skipped[b].append((i, body))
            last = m.end()
        out.append(ln[last:])
        new_lines.append(''.join(out))

    if write and changed:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(new_lines))
    return changed, skipped


def main():
    args = [a for a in sys.argv[1:]]
    write = '--write' in args
    wanted = set()
    if '--buckets' in args:
        idx = args.index('--buckets')
        for a in args[idx + 1:]:
            if a.startswith('-'):
                break
            wanted.add(a.upper())
    else:
        wanted = {'F'}

    total = 0
    per_file = collections.Counter()
    all_skipped = collections.defaultdict(list)
    for rel in md_files():
        ch, sk = convert_file(os.path.join(ROOT, rel), wanted, write)
        total += len(ch)
        per_file[rel] = len(ch)
        for b, items in sk.items():
            all_skipped[b] += [(rel, i, s) for i, s in items]

    mode = '已写入' if write else '干跑'
    print(f'[{mode}] 目标桶 = {sorted(wanted)}')
    print(f'转换处数: {total}')
    for rel, c in per_file.most_common():
        if c:
            print(f'   {c:4d}  {rel}')
    if '--show' in args:
        print('\n--- 转换明细（原文 -> 规范化后）---')
        for rel, c in per_file.most_common():
            if not c:
                continue
            ch, _ = convert_file(os.path.join(ROOT, rel), wanted, False)
            for i, old, new in ch:
                mark = '' if old == new else '  -> ' + repr(new)
                print(f'   {os.path.basename(rel)}:L{i}  {old!r}{mark}')
    for b in sorted(all_skipped):
        print(f'\n--- 桶 {b} 中被规则排除（{len(all_skipped[b])} 处）---')
        for rel, i, s in all_skipped[b]:
            print(f'   {os.path.basename(rel)}:L{i}  {s!r}')
    return total


if __name__ == '__main__':
    main()
