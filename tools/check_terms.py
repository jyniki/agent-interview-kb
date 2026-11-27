#!/usr/bin/env python3
"""按术语表检查禁用叫法。只用标准库。

只收“在任何语境下都不对”的写法，避免把正常的构词误报成术语错误：
- 判定器 / 判分器 → 评分器
- 编排者 → 主 Agent
- 追踪与回放 / 轨迹与回放 / 记录与回放 → Trace 与回放
- 评审委员会 / 裁判模型 这类固定搭配允许，不作检查

用法：python3 tools/check_terms.py
退出码为发现的问题数（最多 100）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BANNED = [
    (r"判定器", "统一叫「评分器」"),
    (r"判分器", "统一叫「评分器」"),
    (r"编排者", "统一叫「主 Agent」"),
    (r"追踪与回放", "统一叫「Trace 与回放」"),
    (r"轨迹与回放", "统一叫「Trace 与回放」"),
    (r"记录与回放", "统一叫「Trace 与回放」"),
]


def main() -> int:
    errs: list[str] = []
    for md in sorted(ROOT.rglob("*.md")):
        rel = md.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.name == "CONTRIBUTING.md":
            continue  # 术语表本身要列出禁用叫法
        for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for pat, hint in BANNED:
                if re.search(pat, line):
                    errs.append(f"{rel}:{no} 出现「{pat}」；{hint}")
    for e in errs:
        print(f"✗ {e}")
    print(f"检查 {len(BANNED)} 条术语规则；{len(errs)} 个问题。")
    return min(len(errs), 100)


if __name__ == "__main__":
    sys.exit(main())
