r"""MkDocs hook：锚点与 GitHub 一致 + 保护数学公式不被 Markdown 破坏。

【功能一】toc 使用与 GitHub 一致的 slugify，保证两侧锚点互通。

关键差异点（相对 Python-Markdown 默认实现）：
- 连续空格不折叠：`PI / NTK` -> `pi--ntk`（每个空格一个连字符）
- ² ³ ① ② ③ · 等字符需显式移除（Python 的 \\w 会保留它们，但 GitHub 会删）
- 保留 CJK 字符（Python-Markdown 默认会把中文全删掉，锚点退化成纯数字）

实现要点：
- MkDocs 校验配置后，`config['markdown_extensions']` 只剩扩展名字符串列表，
  扩展参数单独放在 `config['mdx_configs']`（dict: 扩展名 -> 参数 dict）。
  因此自定义 slugify 必须写进 `mdx_configs['toc']['slugify']`，
  直接塞回 markdown_extensions 会触发
  `TypeError: Extension "builtins.dict" must be of type: "markdown.extensions.Extension"`。
- MkDocs 每渲染一页都会新建一个 markdown.Markdown 实例，但 slugify 闭包是共享的，
  重复标题的去重表必须按页重置（与 github-slugger 的「按文档重置」语义一致），
  否则第 2 页里出现的同名标题会被错误地加上 `-1` 后缀。

【功能二】保护数学公式。
这是文档站「部分公式不渲染」的另一半根因（另一半是 ams 包未自托管）。
Markdown 会把 _..._ 解析成斜体，于是公式里的下标被插入 <em> 标签拦腰截断，例如：

    \mathbb{E}_{p}[\log q_\theta]
      -> \mathbb{E}<em>}[\log q</em>\theta]      # MathJax 收到残片，无法识别

GitHub 的渲染器内置了对公式的保护，所以同一个 md 文件在 GitHub 上正常、
在 MkDocs 站点上却坏掉，现象具有很强的迷惑性。

修法：在 Markdown 解析之前把公式整体替换成哨兵 token（纯 ASCII 字母数字，
Markdown 不会改动它），解析生成 HTML 之后再原样还原。
哨兵按页隔离存放，避免并行/串行渲染时互相污染。
"""
import html as _html
import re

GH_EXTRA = re.compile(
    "["
    "\u00AB-\u00BF"      # «¬­®¯°±²³´µ¶·¸¹º»¼½¾¿
    "\u00D7\u00F7"       # × ÷
    "\u02C2-\u02C5" "\u02D2-\u02DF" "\u02E5-\u02EB"
    "\u0300-\u036F"      # 组合音标
    "\u2010-\u2027"      # 连字符/破折号/引号
    "\u2030-\u205E"
    "\u2100-\u214F" "\u2150-\u218B" "\u2189-\u24B5" "\u2460-\u24FF"
    "\u25A0-\u27BF"      # 几何/装饰符号
    "\u3000-\u303F"      # CJK 标点
    "\uFE10-\uFE4F" "\uFE50-\uFE6F" "\uFEFD-\uFF0F"
    "\uFF1A\uFF20" "\uFF3B-\uFF40" "\uFF5B-\uFF65" "\uFFE0-\uFFEF"
    "]",
    flags=re.UNICODE,
)

_SLUGGER_KEY = "_github_slugger"


def _slug(text):
    """单条标题 -> GitHub 锚点（不含重复标题的去重后缀）"""
    s = text.strip().lower()
    s = GH_EXTRA.sub("", s)
    s = re.sub(r"[^\w\u4e00-\u9fff\- ]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s", "-", s)  # 每个空格一个连字符，不折叠
    return s.strip("-")


class GithubSlugger:
    """与 github-slugger 行为一致：首次出现不带后缀，重复依次加 -1 / -2 ..."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._seen = {}

    def __call__(self, value, separator="-", **_kwargs):
        base = _slug(value)
        n = self._seen.get(base, 0)
        self._seen[base] = n + 1
        if n == 0:
            return base
        return "{}-{}".format(base, n)


# ==================== 数学公式保护 ====================
# 块级 $$...$$：可跨行，且不要求 $$ 独占一行，
# 因为列表/引用内缩进的公式同样需要保护（10 章就有这种情况）。
_MATH_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
# 行内 $...$：不跨行。要求内容含 LaTeX 特征字符（^ _ { } \），
# 避免把正文里的「$5 到 $10」误判成公式而保护起来。
_MATH_INLINE = re.compile(
    r"(?<!\$)\$(?!\$)([^$\n]*?[\^_{}\\][^$\n]*?)(?<!\$)\$(?!\$)")
# 围栏代码块：里面的 $ 是代码，不参与保护
_FENCE = re.compile(r"^(?P<fence>```+|~~~+).*?^(?P=fence)$", re.S | re.M)

_MATH_PREFIX = "ZZMATHSHIELD"
_MATH_SUFFIX = "ZZENDSHIELD"
_FENCE_PREFIX = "ZZFENCESHIELD"

# 按页存放被保护的公式原文：src_path -> [公式, ...]
_MATH_STORE = {}


def _protect_math(text):
    """把公式换成哨兵 token，返回 (处理后的 markdown, 公式原文列表)。"""
    fences = []

    def keep_fence(m):
        fences.append(m.group(0))
        return "%s%d%s" % (_FENCE_PREFIX, len(fences) - 1, _MATH_SUFFIX)

    # 先把围栏代码块摘出去，避免动代码里的 $
    text = _FENCE.sub(keep_fence, text)

    store = []

    def take(m):
        store.append(m.group(0))
        return "%s%d%s" % (_MATH_PREFIX, len(store) - 1, _MATH_SUFFIX)

    text = _MATH_BLOCK.sub(take, text)
    text = _MATH_INLINE.sub(take, text)

    def back_fence(m):
        return fences[int(m.group(1))]

    text = re.sub(_FENCE_PREFIX + r"(\d+)" + _MATH_SUFFIX, back_fence, text)
    return text, store


def _restore_math(html_text, store):
    """把哨兵 token 还原成公式原文。

    公式原文是 LaTeX，可能含 < > &，直接塞回 HTML 会破坏结构，
    因此统一做 HTML 转义；浏览器解析后 MathJax 读到的仍是原始字符。
    """
    def back(m):
        i = int(m.group(1))
        if i >= len(store):
            return m.group(0)
        return _html.escape(store[i], quote=False)

    return re.sub(_MATH_PREFIX + r"(\d+)" + _MATH_SUFFIX, back, html_text)


def on_config(config):
    """把 toc 扩展的 slugify 换成 GitHub 兼容版本。"""
    slugger = GithubSlugger()

    # 扩展参数放在 mdx_configs，markdown_extensions 只保留扩展名
    mdx_configs = config.get("mdx_configs")
    if not isinstance(mdx_configs, dict):
        mdx_configs = {}

    toc_cfg = mdx_configs.get("toc")
    if not isinstance(toc_cfg, dict):
        toc_cfg = {}
    toc_cfg.setdefault("permalink", True)
    toc_cfg.setdefault("permalink_title", "锚定本节")
    toc_cfg["slugify"] = slugger
    mdx_configs["toc"] = toc_cfg
    config["mdx_configs"] = mdx_configs

    # toc 是 MkDocs 内置扩展，理论上必在，这里兜底保证存在
    extensions = list(config.get("markdown_extensions") or [])
    if "toc" not in extensions:
        extensions.append("toc")
    config["markdown_extensions"] = extensions

    # 供 on_page_markdown 重置去重表
    config[_SLUGGER_KEY] = slugger
    return config


def on_page_markdown(markdown, page, config, files):
    """每渲染一页前重置去重表，并把公式替换成哨兵 token 保护起来。"""
    slugger = config.get(_SLUGGER_KEY)
    if slugger is not None:
        slugger.reset()

    protected, store = _protect_math(markdown)
    _MATH_STORE[page.file.src_path] = store
    return protected


def on_page_content(html, page, config, files, **_kwargs):
    """HTML 生成后还原被保护的公式。"""
    store = _MATH_STORE.pop(page.file.src_path, None)
    if store:
        html = _restore_math(html, store)
    return html
