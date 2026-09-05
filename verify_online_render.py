#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
线上全量渲染验证（github.io + 系统代理）

与 verify_local_render.py 的区别
--------------------------------
本脚本**不**起本地服务，直接打 GitHub Pages 线上站点，是六层验证的最后一层。
（早前版本从本地脚本复制而来，文档串仍在讲「自带服务器 / mkdocs build --clean
/ 127.0.0.1:8899」，与实际实现完全不符，已连同残留的死代码一并清理。）

要点：
1. 目标站点是 `https://liq430.github.io/LLM-Learning-Wiki`，**不需要**本地 site/。
2. 本机访问 `*.github.io` 必须走系统代理 `http://127.0.0.1:7897`（环境自带的
   HTTP_PROXY 对 github.io 不通），通过 `chromium.launch(proxy=...)` 显式传入。

统计口径与 mkdocs_hooks.py 的保护逻辑一致（$$ 不要求独占一行），复查五件事：
  1) 源文件公式数  ==  页面渲染出的公式数
  2) MathJax 渲染失败数（merror）== 0
  3) 渲染后页面是否残留未处理的 $$ / \[ / \begin{
  4) 公式是否被 <em> 从中间切断
  5) 所有章节都真正跑完（verified == 章节总数）

关于第 5 条——这是「假 PASS」的防线。本类脚本的通用病害：
遍历 N 个对象，失败就 `continue`，最后比较总数。若全挂，累加器全是 0，
`gs == gt` 成立 → 打印 PASS。因此必须显式统计真正跑完的数量，
且 `verified == len(chapters)` 要进判定式，不能只靠 `not problems` 兜底。
"""
import os
import re
import sys
import pathlib
import urllib.parse

REPO = pathlib.Path(__file__).resolve().parent
# 本机访问 *.github.io 必须走系统代理（环境自带的 HTTP_PROXY 对 github.io 不通）
PROXY = "http://127.0.0.1:7897"

FENCE = re.compile(r"^(?P<fence>```+|~~~+).*?^(?P=fence)$", re.S | re.M)
CODE_INLINE = re.compile(r"`[^`\n]*`")
MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
MATH_INLINE_ALL = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)")


def src_count(p):
    """源端公式数：剔除围栏代码块与行内代码后统计（MathJax 同样会跳过它们）"""
    t = p.read_text(encoding="utf-8")
    t = FENCE.sub("", t)
    t = CODE_INLINE.sub("", t)
    blk = len(MATH_BLOCK.findall(t))
    inl = len(MATH_INLINE_ALL.findall(MATH_BLOCK.sub("", t)))
    return blk, inl


PROBE = r"""
() => {
  const root = document.querySelector('.md-content__inner') || document.body;
  const cs = Array.from(document.querySelectorAll('mjx-container'));
  const errs = cs.filter(c => c.querySelector('merror, mjx-merror'));
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const leftover = [];
  let n;
  while ((n = walker.nextNode())) {
    const pe = n.parentElement;
    if (!pe || pe.closest('code, pre, mjx-container')) continue;
    const t = n.nodeValue || '';
    if (t.includes('$$') || t.includes('\\[') || t.includes('\\begin{')) {
      leftover.push(t.trim().slice(0, 90));
    }
  }
  return {
    total: cs.length,
    failed: errs.length,
    samples: errs.slice(0, 3).map(e =>
      (e.getAttribute('aria-label') || e.innerText || '').slice(0, 130)),
    leftover,
    emInsideMath: (root.innerHTML.match(/\$\$[^$]{0,200}<em>[^$]{0,200}\$\$/g) || []).length,
  };
}
"""


def main():

    base = "https://liq430.github.io/LLM-Learning-Wiki"

    from playwright.sync_api import sync_playwright

    # 清掉环境自带的 HTTP_PROXY（它对 *.github.io 不通），改由下面的
    # launch(proxy=...) 显式指定系统代理 127.0.0.1:7897。
    # 注意与本地脚本相反：本地版要绕过代理直连 127.0.0.1，这里恰恰要强制走代理。
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)

    chapters = sorted((REPO / "chapters").glob("*.md"))
    print(f"待验证章节 {len(chapters)} 个，目标 {base}（经代理 {PROXY}）\n")
    print(f"  {'章节':38s} {'块':>5s} {'行内':>5s} | {'渲染':>5s} {'失败':>4s} {'残留':>4s}")
    print("  " + "-" * 74)

    gs = gt = gf = gl = gem = 0
    verified = 0                  # 真正取到渲染结果的章节数（假 PASS 防线）
    problems = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(proxy={"server": PROXY})
        ctx = b.new_context(viewport={"width": 1400, "height": 900})
        page = ctx.new_page()
        for p in chapters:
            blk, inl = src_count(p)
            nsrc = blk + inl
            url = base + "/chapters/" + urllib.parse.quote(p.stem) + "/"
            try:
                page.goto(url, wait_until="load", timeout=60000)
                page.wait_for_function(
                    "() => window.MathJax && window.MathJax.startup"
                    " && window.MathJax.startup.promise", timeout=30000)
                page.evaluate("() => window.MathJax.startup.promise")
            except Exception as e:
                problems.append(f"{p.name}: 加载异常 {str(e)[:80]}")
                print(f"  {p.stem[:36]:38s} {blk:5d} {inl:5d} | 加载异常")
                continue
            page.wait_for_timeout(500)
            r = page.evaluate(PROBE)
            verified += 1         # 放在所有 continue 之后
            gs += nsrc; gt += r["total"]; gf += r["failed"]
            gl += len(r["leftover"]); gem += r["emInsideMath"]
            marks = []
            if r["failed"]:
                marks.append(f"失败{r['failed']}")
            if r["total"] != nsrc:
                marks.append(f"数量源{nsrc}/页{r['total']}")
            if r["leftover"]:
                marks.append(f"残留{len(r['leftover'])}")
            if r["emInsideMath"]:
                marks.append(f"em切断{r['emInsideMath']}")
            flag = ("   <== " + "，".join(marks)) if marks else ""
            print(f"  {p.stem[:36]:38s} {blk:5d} {inl:5d} | "
                  f"{r['total']:5d} {r['failed']:4d} {len(r['leftover']):4d}{flag}")
            for s in r["samples"]:
                print(f"        ! {s}")
            for s in r["leftover"][:3]:
                print(f"        ~ {s}")
            if marks:
                problems.append(p.name)
        b.close()

    print("\n" + "=" * 74)
    print(f"  已验证章节 {verified}/{len(chapters)}")
    print(f"  源文件公式合计 {gs} 个 | 页面渲染 {gt} 个 | 差值 {gt - gs}")
    print(f"  MathJax 渲染失败 {gf} 个 | 未处理残留 {gl} 处 | 被 <em> 切断 {gem} 处")
    if problems:
        print(f"\n  问题明细（共 {len(problems)} 条）：")
        for x in problems[:20]:
            print(f"    - {x}")
        if len(problems) > 20:
            print(f"    ... 另有 {len(problems) - 20} 条未展示")
    # verified 必须等于章节总数：否则「一章都没打开」时 gs == gt == 0
    # 会让判定成立，报出一个彻头彻尾的假 PASS。
    ok = (gf == 0 and gl == 0 and gem == 0 and gs == gt
          and not problems and verified == len(chapters))
    if verified != len(chapters):
        print(f"  ===> FAIL：{len(chapters) - verified} 章未真正完成验证，"
              f"统计口径不成立（数字再漂亮也不能算通过）")
    elif ok:
        print("  ===> PASS：全站公式渲染完整且无失败")
    else:
        print("  ===> FAIL，见上方明细")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
