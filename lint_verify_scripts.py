#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
假 PASS 检测器（AST 静态分析）

## 检测的病害

「遍历 N 个对象、失败就 continue、最后比较总数」这类校验脚本的通用缺陷：

    for x in items:
        try:
            r = probe(x)
        except Exception:
            problems.append(x)
            continue                  # ← 累加器保持 0
        gs += r
    ok = (... and gs == gt)           # 0 == 0 成立 → 打印 PASS

全站挂掉和全站健康，汇总行打印出来一模一样。
真实事故见 verify_final.py（已废弃，保留作反面教材）。

## 判定条件（三条同时满足才报风险）

1. 循环体内存在 try/except，且 handler 里以 `continue` 结束
2. 收尾处存在 `ok = (... A == B ...)` 形式的判定（A/B 为累加器）
3. 循环内**没有**一个被 `+=` 递增、且出现在 ok 判定式里的计数变量
   —— 也就是缺少 `verified` 那样的显式防线

## 用法

    python3 lint_verify_scripts.py [目录或文件 ...]

退出码：0 = 未发现风险；1 = 发现风险（CI 可直接用）
"""
import ast
import pathlib
import sys


def _contains_continue(node: ast.AST) -> bool:
    """handler 体里是否有 continue（含嵌套在 if 里的）"""
    for n in ast.walk(node):
        if isinstance(n, ast.Continue):
            return True
    return False


def _risky_loops(tree: ast.AST):
    """找出「except 里 continue」的循环"""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                if any(_contains_continue(h) for h in child.handlers):
                    found.append(node)
                    break
    return found


def _incremented_names(loop: ast.AST):
    """循环内被 += 递增的变量名（排除 except handler 内部的）"""
    names = set()
    skip = set()
    for node in ast.walk(loop):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                for sub in ast.walk(h):
                    skip.add(id(sub))
    for node in ast.walk(loop):
        if id(node) in skip:
            continue
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def _ok_expressions(tree: ast.AST):
    """收尾判定式：ok = (... == ...) 以及 return 0 if ok else 1 里的 ok"""
    outs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ok":
                    outs.append(node.value)
    return outs


def _has_eq(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Compare):
            if any(isinstance(o, ast.Eq) for o in n.ops):
                return True
    return False


def _names_in(node: ast.AST):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _appended_names_in_handlers(loop: ast.AST):
    """except handler 里被 .append(...) 的容器变量名（如 problems）"""
    names = set()
    for node in ast.walk(loop):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                for sub in ast.walk(h):
                    if (isinstance(sub, ast.Call)
                            and isinstance(sub.func, ast.Attribute)
                            and sub.func.attr == "append"
                            and isinstance(sub.func.value, ast.Name)):
                        names.add(sub.func.value.id)
    return names


def _has_real_guard(ok_expr: ast.AST, inc: set, appended: set) -> bool:
    """
    判定 ok 表达式里是否存在「真防线」。

    两种合法防线：
      (a) 显式计数：Eq 比较中**恰好一侧**是循环内递增的变量，另一侧不是
          —— `verified == len(items)`
          注意不能只看「有没有递增变量参与比较」：
          `gs == gt` 两侧都是累加器，全挂时 0 == 0 照样成立——那正是病害本身。
      (b) 问题清单取反：ok 里含 `not problems`，且 problems 在 handler 里被 append
          —— 弱一档，但确实能挡住全挂的假 PASS
    """
    for n in ast.walk(ok_expr):
        if isinstance(n, ast.Compare) and any(isinstance(o, ast.Eq) for o in n.ops):
            left = n.left
            for right in n.comparators:
                for a, b in ((left, right), (right, left)):
                    a_inc = isinstance(a, ast.Name) and a.id in inc
                    b_inc = isinstance(b, ast.Name) and b.id in inc
                    if a_inc and not b_inc:
                        return True                      # (a)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            if isinstance(n.operand, ast.Name) and n.operand.id in appended:
                return True                              # (b)
    return False


def analyze(path: pathlib.Path):
    """返回 (风险列表, 备注)"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as e:
        return [], f"解析失败：{e}"

    loops = _risky_loops(tree)
    if not loops:
        return [], "无 except→continue 循环"

    oks = [o for o in _ok_expressions(tree) if _has_eq(o)]
    if not oks:
        return [], "无 `ok = (... == ...)` 判定式"

    risks = []
    for loop in loops:
        inc = _incremented_names(loop)
        appended = _appended_names_in_handlers(loop)
        guarded = any(_has_real_guard(o, inc, appended) for o in oks)
        if not guarded:
            risks.append({
                "line": getattr(loop, "lineno", "?"),
                "incremented": sorted(inc) or "（无）",
                "appended": sorted(appended) or "（无）",
            })
    return risks, ""


def main(argv):
    targets = argv[1:] or ["."]
    files = []
    for t in targets:
        p = pathlib.Path(t)
        if p.is_file():
            files.append(p)
        else:
            files.extend(sorted(p.rglob("*.py")))

    # 跳过本工具自身与已知已废弃的脚本（它就是反面教材，不必再报）
    files = [f for f in files if f.name not in {"lint_verify_scripts.py"}]

    risky_files = []
    # 漏斗统计：防止「因为压根没匹配到任何东西而报 PASS」——
    # 这个检测器自己也不能患上它要检测的那种病。
    stat = {"parse_fail": 0, "risky_loop": 0, "ok_expr": 0,
            "guarded": 0, "unguarded": 0}
    for f in files:
        risks, note = analyze(f)
        deprecated = "DEPRECATED" in f.read_text(encoding="utf-8", errors="ignore")
        if note.startswith("解析失败"):
            stat["parse_fail"] += 1
        if risks:
            stat["unguarded"] += 1
            risky_files.append((f, risks, deprecated))
        elif note == "无 except→continue 循环":
            pass
        elif note == "无 `ok = (... == ...)` 判定式":
            stat["risky_loop"] += 1
        else:
            stat["guarded"] += 1

    print(f"扫描 {len(files)} 个 .py 文件")
    print(f"  漏斗：含 except→continue 循环 {stat['risky_loop'] + stat['guarded'] + stat['unguarded']} 个"
          f"（其中有 ok 判定式 {stat['guarded'] + stat['unguarded']} 个，"
          f"有防线 {stat['guarded']} / 无防线 {stat['unguarded']}）"
          f" | 无 ok 判定式 {stat['risky_loop']} 个 | 解析失败 {stat['parse_fail']} 个")
    if stat["guarded"] + stat["unguarded"] == 0:
        print("  !! 警告：一个含风险循环且有 ok 判定式的文件都没匹配到，"
              "PASS 可能只是「没查到」，请核对漏斗计数")
    print()
    if not risky_files:
        print("  ===> PASS：未发现「假 PASS」风险模式")
        return 0

    print("  ===> 发现风险（except→continue + 最终比较累加器 + 无显式计数）：\n")
    for f, risks, deprecated in risky_files:
        tag = "  [已标记 DEPRECATED]" if deprecated else ""
        print(f"  {f}{tag}")
        for r in risks:
            print(f"      循环 @L{r['line']}：递增变量 {r['incremented']}，"
                  f"handler 内 append 的容器 {r['appended']}；"
                  f"但 ok 判定式里没有 `verified == len(items)` 也没有 "
                  f"`not <该容器>`")
    print("\n  修法：加 `verified += 1`（放在所有 continue 之后），"
          "并让 `verified == len(items)` 进入 ok 判定式。")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
