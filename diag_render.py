#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
定向诊断：只看指定章节，把 MathJax 报错公式的**源码**和渲染差异定位到行。

用法：python3 diag_render.py 07 09
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
PORT = 8978

FENCE = re.compile(r"^(?P<fence>```+|~~~+).*?^(?P=fence)$", re.S | re.M)
CODE_INLINE = re.compile(r"`[^`\n]*`")
MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
MATH_INLINE = re.compile(r"(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)")

PROBE = r"""
() => {
  const cs = Array.from(document.querySelectorAll('mjx-container'));
  const errs = cs.filter(c => c.querySelector('merror, mjx-merror'));
    return {
        total: cs.length,
        failed: errs.length,
        // 报错容器在文档中的序号 == 源端公式的序号（渲染顺序即文档顺序）
        failIdx: cs.map((c, i) => c.querySelector('merror, mjx-merror') ? i : -1)
                   .filter(i => i >= 0),
        samples: errs.slice(0, 10).map(e =>
            (e.getAttribute('aria-label') || e.innerText || '').slice(0, 200)),
    };
}
"""


def src_formulas(p):
    """返回 [(body, is_block), ...]，按出现顺序"""
    t = p.read_text(encoding="utf-8")
    t = FENCE.sub("", t)
    t = CODE_INLINE.sub("", t)
    out, last = [], 0
    for m in re.finditer(r"\$\$(.+?)\$\$|(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)", t, re.S):
        if m.group(1) is not None:
            out.append((m.group(1).strip(), True))
        else:
            out.append((m.group(2).strip(), False))
    return out


def main():
    keys = sys.argv[1:] or ["07", "09"]
    targets = []
    for p in sorted((REPO / "chapters").glob("*.md")):
        if any(k in p.name for k in keys):
            targets.append(p)
    if not targets:
        print("没匹配到章节")
        return 2

    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(SITE))
    handler.log_message = lambda *a, **k: None
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{PORT}"

    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--no-proxy-server"])
        page = b.new_context(viewport={"width": 1400, "height": 900}).new_page()
        for p in targets:
            url = base + "/chapters/" + urllib.parse.quote(p.stem) + "/"
            page.goto(url, wait_until="load", timeout=60000)
            page.wait_for_function(
                "() => window.MathJax && window.MathJax.startup && window.MathJax.startup.promise",
                timeout=30000)
            page.evaluate("() => window.MathJax.startup.promise")
            page.wait_for_timeout(600)
            r = page.evaluate(PROBE)

            srcs = src_formulas(p)
            print(f"\n{'=' * 72}\n{p.name}   源端 {len(srcs)} 条 | 页面 {r['total']} 条 | "
                  f"失败 {r['failed']}")
            if r["failed"]:
                print("  ── MathJax 报错（按序号对应源端公式）──")
                for i in r["failIdx"]:
                    body = srcs[i][0] if i < len(srcs) else '(越界)'
                    print(f"    [#{i}] {body[:300]}")
                for s in r["samples"]:
                    print(f"    ! {s.splitlines()[0] if s else s}")
            # 页面比源端少：从页面里逐条找不到的源端公式（用序号近似定位）
            if r["total"] < len(srcs):
                print(f"  ── 页面少渲染 {len(srcs) - r['total']} 条，源端公式清单前 60 条 ──")
                for i, (body, isblk) in enumerate(srcs[:60]):
                    print(f"    [{i}] {'$$' if isblk else '$'}{body[:110]}")
        b.close()
    httpd.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
