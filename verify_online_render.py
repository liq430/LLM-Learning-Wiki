#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
线上全量渲染验证（github.io + 系统代理）

为什么要自带服务器
------------------
1. `mkdocs build --clean` 会释放 site/ 的 inode，外部常驻的 http.server 会
   立刻失效并返回 502；本脚本在进程内起服务，生命周期与验证任务严格一致。
2. 环境变量里有 HTTP_PROXY（127.0.0.1:51527），Chromium 默认会走它访问
   127.0.0.1:8899 而失败；必须显式 `proxy={'server': 'direct://'}`。

统计口径与 mkdocs_hooks.py 的保护逻辑一致（$$ 不要求独占一行），复查四件事：
  1) 源文件公式数  ==  页面渲染出的公式数
  2) MathJax 渲染失败数（merror）== 0
  3) 渲染后页面是否残留未处理的 $$ / \[ / \begin{
  4) 公式是否被 <em> 从中间切断
"""
import os
import re
import sys
import pathlib
import functools
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

REPO = pathlib.Path(__file__).resolve().parent
SITE = REPO / "site"
PORT = 8979

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

    # 关键：环境里有 HTTP_PROXY，Chromium 会把 127.0.0.1 也送去代理而
    # ERR_PROXY_CONNECTION_FAILED。proxy={'server':'direct://'} 实测无效，
    # 必须靠 --no-proxy-server 启动参数 + 清掉进程内代理环境变量双保险。
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    chapters = sorted((REPO / "chapters").glob("*.md"))
    print(f"待验证章节 {len(chapters)} 个，服务 {base}\n")
    print(f"  {'章节':38s} {'块':>5s} {'行内':>5s} | {'渲染':>5s} {'失败':>4s} {'残留':>4s}")
    print("  " + "-" * 74)

    gs = gt = gf = gl = gem = 0
    problems = []
    with sync_playwright() as pw:
        b = pw.chromium.launch(proxy={"server": "http://127.0.0.1:7897"})
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
    print(f"  源文件公式合计 {gs} 个 | 页面渲染 {gt} 个 | 差值 {gt - gs}")
    print(f"  MathJax 渲染失败 {gf} 个 | 未处理残留 {gl} 处 | 被 <em> 切断 {gem} 处")
    if problems:
        print(f"\n  问题章节 {len(problems)} 个：")
        for x in problems[:20]:
            print(f"    - {x}")
    ok = (gf == 0 and gl == 0 and gem == 0 and gs == gt and not problems)
    print("  ===> " + ("PASS：全站公式渲染完整且无失败" if ok else "FAIL，见上方明细"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
