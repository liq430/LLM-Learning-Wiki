/*
 * MathJax 3 配置 —— 必须在主库加载之前注入。
 *
 * 背景：先前公式「有的能渲染、有的不能」，根因是
 *       \text{}  \operatorname{}  \begin{cases}  \begin{align}  \begin{aligned}
 *   这些命令属于 ams 扩展包，而 MathJax 默认的 tex-mml-chtml bundle 不含 ams。
 *   一旦 ams 没加载成功，用到它的公式就会静默失败（其余公式照常渲染），
 *   表现为「部分公式显示成 $...$ 原文」。
 *
 * 修法：通过 loader.load + packages['[+]'] 显式挂载 ams。
 * 注意：不要在这里写 loader.paths —— 让 MathJax 从「主库自身的 src」反推根目录，
 *       这样 CDN 源就找 CDN 的 ams、自托管源就找自托管的 ams，两边都自洽。
 */
window.MathJax = {
  tex: {
    // 行内公式分隔符；$...$ 与 \(...\) 都支持
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    // 行间（块级）公式分隔符
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    // \$ 转义输出字面量 $，避免正文里的货币金额被当成公式
    processEscapes: true,
    // 关键：在默认包之外追加 ams
    packages: { "[+]": ["ams"] }
  },
  loader: {
    load: ["[tex]/ams"]
  },
  options: {
    // 跳过代码块与纯文本区域，防止 $ 被误判为公式起始
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code", "annotation"],
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process"
  },
  svg: {
    // 让公式字号随页面缩放，而不是固定 em
    fontCache: "global"
  }
};
