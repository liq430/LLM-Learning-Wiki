#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 math mode 中的「裸多字符下标」规范成 LaTeX 花括号形式。

问题
----
LaTeX 里 `n_kv` 只把**第一个字符**当下标，等价于 `n_k v` —— n 带下标 k，
后面的 v 掉回正文基线。要写 `n_{\mathrm{kv}}` 才是"n 下标 kv"。

站内有 53 种 / 135 处这类写法：
    D_KL        -> D_{\mathrm{KL}}
    n_kv        -> n_{\mathrm{kv}}
    T_warmup    -> T_{\mathrm{warmup}}
    π_ref       -> \pi_{\mathrm{ref}}
    η_max       -> \eta_{\max}
    C_req,avg   -> C_{\mathrm{req,avg}}

其中：
- 纯数学索引（ii / ij / jj / tt）保持斜体，不加 \mathrm：`a_{ii}`
- max / min 是算子，用 `\max` / `\min`：`η_{\max}`

安全边界
--------
- 围栏代码块、行内代码跨度先整体掩码，绝不动
- 已经写成 `_{...}` 的一律跳过
- 单字符下标（`d_h`、`x_t`）本来就正确，跳过

用法
----
    python3 fix_subscripts.py           # 干跑，打印将发生的替换
    python3 fix_subscripts.py --write   # 落盘
"""
import re
import os
import sys
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))

GREEK = 'αβγδεζηθικλνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'

# 围栏代码块
FENCE = re.compile(r'^[ \t]*(?:```|~~~).*?(?:^[ \t]*(?:```|~~~)[ \t]*$|$)', re.S | re.M)
# 行内代码跨度（定界符长度 1..3）
INLINE_CODE = re.compile(r'(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)')
# 数学区域：先匹配块级 $$...$$ 与 \[...\]，再匹配行内 $...$
MATH = re.compile(
    r'\$\$(.+?)\$\$'
    r'|\\\[(.+?)\\\]'
    r'|(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)',
    re.S,
)
# 裸下标：基符号（ASCII 字母 / 希腊字母 / \xxx 命令） + _ + 未加花括号的多字符
# 注意：word 只吃 \w，不能吃逗号 —— 否则 f(q_i,D) 会被误读成 q_{i,D}（实为两个参数）
# 注意 2：必须用 ASCII [A-Za-z0-9]，不能用 \w —— Python 的 \w 在 Unicode 模式下
# 会把上标 ²（U+00B2）算进来，导致 q_i² 被误读成 q_{i²}（实为 q_i 的平方）
BARE = re.compile(
    r'(?<![\w{\\])([A-Za-z' + GREEK + r']|\\[A-Za-z]+)'
    r'_((?![{\\])(?:[A-Za-z][A-Za-z0-9]*))'
)

# 纯数学索引：保持斜体，不加 \mathrm
MATH_INDEX = {'ii', 'ij', 'ji', 'jj', 'kk', 'tt', 'nn', 'mm', 'ss', 'rr'}
# 算子名：用 \max / \min
OPERATOR = {'max': r'\max', 'min': r'\min'}


def fix_body(body, stats):
    def rep(m):
        base, word = m.group(1), m.group(2)
        # 单字符下标（d_h / x_t / q_i）本来就正确，跳过
        if len(word) == 1:
            return m.group(0)
        if word in MATH_INDEX:
            out = f'{base}_{{{word}}}'
        elif word in OPERATOR:
            out = f'{base}_\\{word}'
        else:
            out = f'{base}_{{\\mathrm{{{word}}}}}'
        stats[f'{base}_{word}'] += 1
        return out
    return BARE.sub(rep, body)


def process(text, stats, where):
    code_store = []

    def stash(m):
        code_store.append(m.group(0))
        return f'\x00{len(code_store) - 1}\x00'

    # 1) 掩码围栏代码块，再掩码行内代码
    masked = FENCE.sub(stash, text)
    masked = INLINE_CODE.sub(stash, masked)

    # 2) 在剩余文本里定位数学区域并替换
    def math_rep(m):
        body = m.group(1)
        if body is None:
            body = m.group(2)
        if body is None:
            body = m.group(3)
        new = fix_body(body, stats)
        if m.group(1) is not None:
            return '$$' + new + '$$'
        if m.group(2) is not None:
            return '\\[' + new + '\\]'
        return '$' + new + '$'

    masked = MATH.sub(math_rep, masked)

    # 3) 还原
    return re.sub(r'\x00(\d+)\x00', lambda m: code_store[int(m.group(1))], masked)


def md_files():
    out = []
    for d in ('chapters', 'docs'):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            continue
        out += [os.path.join(d, f) for f in sorted(os.listdir(p)) if f.endswith('.md')]
    return out


def main():
    write = '--write' in sys.argv
    stats = collections.Counter()
    where = collections.Counter()
    total_files = 0
    for rel in md_files():
        path = os.path.join(ROOT, rel)
        src = open(path, encoding='utf-8').read()
        new = process(src, stats, where)
        if new != src:
            total_files += 1
            if write:
                open(path, 'w', encoding='utf-8').write(new)

    print(f'{"[已写入]" if write else "[干跑]"} 替换种类 {len(stats)}，'
          f'总处数 {sum(stats.values())}，涉及文件 {total_files}')
    for k, c in stats.most_common():
        print(f'   {c:3d}  {k}')
    return sum(stats.values())


if __name__ == '__main__':
    main()
