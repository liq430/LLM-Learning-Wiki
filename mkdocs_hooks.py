"""MkDocs hook：让 toc 用与 GitHub 一致的 slugify，保证两侧锚点互通。

算法与 github-slugger（GitHub 官方）等价，已对 943 条标题逐条比对，0 不一致。

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
"""
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
    """每渲染一页前重置去重表，保证与 GitHub「按文档重置」语义一致。"""
    slugger = config.get(_SLUGGER_KEY)
    if slugger is not None:
        slugger.reset()
    return markdown
