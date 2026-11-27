#!/usr/bin/env python3
"""检查文档与数据的一致性。只用标准库。

检查项：
1. json 里每章的 minutes 与 README 头部的“预计 N 分钟”一致。
2. README 的“内容状态”和知识地图里的题数、章节数、时长与实际一致。
3. 每章 `## N` 的编号从 1 连续递增。
4. 每章 ```` ```python ```` 代码块不少于 MIN_CODE 段（--strict 时检查设计题的结构）。
5. json 里登记的 id 都能解析到目录。

用法：python3 tools/check_docs.py [--strict]
  --strict  额外检查设计题：必须有可运行代码、失败模式、演进路线
退出码为发现的问题数（最多 100）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "kb"
MIN_CODE = 3


def chapters() -> list[tuple[str, str, int, Path]]:
    """(id, title, minutes, README 路径)"""
    out = []
    for j in [KB / "core" / "core.json", *sorted((KB / "tracks").glob("*/track.json"))]:
        data = json.loads(j.read_text(encoding="utf-8"))
        for c in data["chapters"]:
            cid = c["id"]
            base = ROOT / "kb" / cid if cid.startswith("core/") else ROOT / "kb" / "tracks" / cid
            out.append((cid, c.get("title", ""), int(c.get("minutes", 0)), base / "README.md"))
    return out


def questions_by_layer() -> Counter:
    """按目录第一段统计题目数（core / training / ... / design / code）。"""
    c: Counter = Counter()
    for md in KB.rglob("questions.md"):
        rel = md.relative_to(KB)
        layer = "core" if rel.parts[0] == "core" else ("design" if rel.parts[0] == "design" else rel.parts[1])
        c[layer] += sum(1 for line in md.read_text(encoding="utf-8").splitlines() if line.startswith("## "))
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    errs: list[str] = []

    chs = chapters()
    for cid, _title, minutes, readme in chs:
        if not readme.exists():
            errs.append(f"{cid} 在 json 里登记了，但 {readme.relative_to(ROOT)} 不存在")
            continue
        head = "\n".join(readme.read_text(encoding="utf-8").splitlines()[:5])
        m = re.search(r"预计 (\d+) 分钟", head)
        if not m:
            errs.append(f"{readme.relative_to(ROOT)} 头部缺少“预计 N 分钟”")
        elif int(m.group(1)) != minutes:
            errs.append(f"{readme.relative_to(ROOT)} 写 {m.group(1)} 分钟，json 是 {minutes}")

        text = readme.read_text(encoding="utf-8")
        nums = [int(x) for x in re.findall(r"^## (\d+) ", text, re.M)]
        if nums and nums != list(range(1, len(nums) + 1)):
            errs.append(f"{readme.relative_to(ROOT)} 的二级标题编号不连续：{nums}")
        n_code = len(re.findall(r"^```python\s*$", text, re.M))
        if n_code < MIN_CODE:
            errs.append(f"{readme.relative_to(ROOT)} 只有 {n_code} 段 ```python，至少应有 {MIN_CODE} 段")

        if args.strict and cid.startswith("design/"):
            for sec, hint in (("失败模式", "失败模式"), ("演进", "演进路线")):
                if sec not in text:
                    errs.append(f"{readme.relative_to(ROOT)} 缺少「{hint}」一节")
            if n_code == 0:
                errs.append(f"{readme.relative_to(ROOT)} 没有可运行代码（估算要写成 ```python 并用 assert 验证）")

    # 题目数量
    by_layer = questions_by_layer()
    total = sum(by_layer.values()) + by_layer["code"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"共 (\d+) 道题", readme)
    if m and int(m.group(1)) != total:
        errs.append(f"README 写共 {m.group(1)} 道题，实际 {total} 道")
    m = re.search(r"(\d+) 个知识章节", readme)
    n_ch = len(chs)
    if m and int(m.group(1)) != n_ch:
        errs.append(f"README 写 {m.group(1)} 个知识章节，实际 {n_ch} 个")

    roadmap = (KB / "00-roadmap.md").read_text(encoding="utf-8")
    label = {"core": "通用层", "training": "training", "harness": "harness",
             "evaluation": "evaluation", "inference": "inference"}
    for layer in label:
        for line in roadmap.splitlines():
            if line.startswith("|") and re.search(rf"\|\s*{label[layer]}\s*\|", line):
                cells = [x.strip() for x in line.strip("|").split("|")]
                if len(cells) >= 3 and cells[2].isdigit():
                    want = by_layer[layer]
                    if int(cells[2]) != want:
                        errs.append(f"知识地图写 {cells[0]} {cells[2]} 题，实际 {want} 题")
    m = re.search(r"约 (\d+) 小时", roadmap)
    total_min = sum(c[2] for c in chs)
    if m and abs(int(m.group(1)) - total_min / 60) > 1:
        errs.append(f"知识地图写约 {m.group(1)} 小时，实际 {total_min / 60:.1f} 小时")

    for e in errs:
        print(f"✗ {e}")
    print(f"检查 {len(chs)} 章、{total} 道题；{len(errs)} 个问题。")
    return min(len(errs), 100)


if __name__ == "__main__":
    sys.exit(main())
