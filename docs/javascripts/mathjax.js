// 自定义 MathJax 配置：让 Material 加载 ams 包
// 必须在 Material 自带的 mathjax 脚本之前加载（mkdocs.yml 的 extra_javascript 顺序保证）
//
// 背景：Material for MkDocs 默认使用 MathJax 3 的 tex-mml-chtml.js，
// 该 bundle 不包含 ams 包，导致以下常用命令无法渲染：
//   \operatorname{...}  \begin{align}  \begin{cases}  \begin{aligned}
// 这会引发 "The following macros are not allowed: operatorname" 致命错误，
// 进一步导致同一页其它公式也全部不渲染。
//
// 修法：提前把 [tex]/ams 加入 loader 与 packages。
// 注意：必须在 window.MathJax 上设置，Material 的脚本会读取这个对象。
window.MathJax = {
  tex: {
    loader: { load: ["[tex]/ams"] },
    packages: { "[+]": ["ams"] }
  }
};
