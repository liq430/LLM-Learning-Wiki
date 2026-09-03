/*
 * MathJax 主库加载器 —— 多源回退，优先国内 CDN，最后回退到仓库自托管副本。
 *
 * 为什么需要回退：
 *   jsdelivr / unpkg / cdnjs 在国内经常被墙或限速到超时，
 *   一旦主库没进浏览器，整页公式都会保持 $...$ 原样。
 *   而只自托管主库、ams 仍走外链，又会造成「部分公式失效」。
 *
 * 策略（按序尝试，前一个 onerror 才试下一个）：
 *   1) BootCDN    —— 国内，快
 *   2) 阿里云     —— 国内，备用
 *   3) 仓库自托管 —— GitHub Pages 同域，永远可达
 *
 * 自托管副本的站点根目录从 Material 的 bundle 脚本反推，
 * 因此本地 `mkdocs serve`（根=/）与 GitHub Pages（根=/LLM-Learning-Wiki/）都能自动适配。
 */
(function () {
  "use strict";

  var CDNS = [
    "https://cdn.bootcdn.net/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.js",
    "https://gw.alipayobjects.com/os/lib/mathjax/3.2.2/es5/tex-mml-chtml.js"
  ];

  // 从 Material 的 bundle 脚本 src 反推站点根，得到自托管主库的绝对路径
  function localMain() {
    var bundle = document.querySelector('script[src*="assets/javascripts/bundle"]');
    var src = bundle && bundle.getAttribute("src");
    var idx = src ? src.indexOf("assets/javascripts/") : -1;
    return idx >= 0 ? src.slice(0, idx) + "javascripts/tex-mml-chtml.js" : null;
  }

  var sources = CDNS.slice();
  var local = localMain();
  if (local) sources.push(local);

  function load(i) {
    if (i >= sources.length) {
      console.warn("[MathJax] 所有加载源均失败，公式将以源码形式显示");
      return;
    }
    var url = sources[i];
    var isLocal = url === local;

    if (isLocal) {
      // 明确告诉 MathJax 自托管根目录，ams 包会去 <root>/input/tex/extensions/ams.js 找
      // 注意：去掉尾部斜杠——MathJax 的路径/字体模板是 "[mathjax]/xxx" 形式，
      //       根目录若带尾斜杠会拼出 "javascripts//output/..."（双斜杠 → 字体 404）
      window.MathJax = window.MathJax || {};
      window.MathJax.loader = window.MathJax.loader || {};
      window.MathJax.loader.paths = window.MathJax.loader.paths || {};
      window.MathJax.loader.paths.mathjax = url.replace(/tex-mml-chtml\.js$/, "").replace(/\/$/, "");
    }

    var s = document.createElement("script");
    s.id = "MathJax-script";
    s.async = false;
    s.src = url;
    s.onerror = function () {
      if (s.parentNode) s.parentNode.removeChild(s);
      load(i + 1);
    };
    document.head.appendChild(s);
  }

  load(0);
})();
