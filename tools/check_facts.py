#!/usr/bin/env python3
"""检查跨章共用的关键数字有没有写岔。只用标准库。

两类检查：
1. 数值口径：同一事实在多章出现时，出现的数值必须落在允许集合里。
   （例如任何提到 H100 带宽的行，数值只能是 3.35 TB/s）
2. 漂移写法：已知的错误变体直接报错（例如 989 TFLOPS）。

新增章节时如果引入新的共用数字，在 NUMERIC 里加一条。

用法：python3 tools/check_facts.py
退出码为发现的问题数（最多 100）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "kb"


def norm(v: str) -> str:
    """把 147,456 / 147456 / 3.35 统一成可比较的字符串。"""
    v = v.replace(",", "").strip()
    if v.isdigit():
        return str(int(v))
    try:
        return f"{float(v):g}"
    except ValueError:
        return v


# name: 给人看的事实名
# context: 行里必须命中的上下文正则；value: 要抽出来比较的数值正则（第 1 个分组）
# allowed: 允许的数值（已归一化）；note: 报错时的提示
NUMERIC = [
    {
        "name": "H100 SXM 显存带宽",
        "pattern": r"H100[^。\n]{0,60}?([\d.]+)\s*TB/s",
        "allowed": {"3.35"},
        "note": "H100 SXM 标称带宽是 3.35 TB/s",
    },
    {
        "name": "H100 BF16 算力",
        "pattern": r"H100[^。\n]{0,60}?([\d,]+)\s*TFLOPS",
        "allowed": {"990", "1979"},  # 稠密 / 带稀疏
        "note": "官方标称 1,979 TFLOPS（带稀疏），稠密约 990",
    },
    {
        "name": "Qwen3-8B 每 token KV（字节）",
        "pattern": r"Qwen3-8B[^。\n]{0,40}?([\d,]{5,})\s*B\b",
        "allowed": {"147456"},
        "note": "2 × 36 层 × 8 个 KV 头 × 128 × 2 B = 147,456 B",
    },
    {
        "name": "Qwen3-8B 每 token KV（KiB）",
        "pattern": r"Qwen3-8B[^。\n]{0,40}?([\d.]+)\s*KiB",
        "allowed": {"144"},
        "note": "147,456 B 按 1024 换算是 144 KiB",
    },
    {
        "name": "Llama 3 8B 参数总数",
        "pattern": r"Llama 3 8B[^。\n]{0,20}?([\d,]{9,})",
        "allowed": {"8030261248"},
        "note": "和 safetensors 的总数一致：8,030,261,248",
    },
    {
        "name": "前缀缓存写价",
        "pattern": r"(?:5 分钟)?缓存写[^。\n]{0,12}?([\d.]+)\s*倍",
        "allowed": {"1.25"},
        "not_before": r"小时",
        "note": "5 分钟缓存写是 1.25 倍（1 小时写是 2 倍，另行表述）",
    },
    {
        "name": "前缀缓存 1 小时写价",
        "pattern": r"1 小时缓存[^。\n]{0,12}?([\d.]+)\s*倍",
        "allowed": {"2"},
        "note": "1 小时缓存写是 2 倍",
    },
]

# 已知的错误变体：直接报错
FORBIDDEN = [
    (r"989\s*TFLOPS", "H100 稠密算力按 990 TFLOPS 写"),
    (r"3\.3\s*TB/s", "H100 带宽是 3.35 TB/s"),
    (r"143\s*KiB", "Qwen3-8B 每 token KV 是 144 KiB"),
    (r"147,?456\s*KiB", "147,456 的单位是字节，不是 KiB"),
]

# 至少要出现一次的锚点事实：防止改稿时被整段删掉
PRESENCE = [
    (r"3\.35\s*TB/s", 5, "H100 带宽"),
    (r"990\s*TFLOPS", 5, "H100 稠密算力"),
    (r"144\s*KiB", 5, "Qwen3-8B 每 token KV"),
    (r"8,030,261,248", 1, "Llama 3 8B 参数总数"),
]


def md_files() -> list[Path]:
    return [p for p in sorted(KB.rglob("*.md"))
            if not any(part.startswith(".") for part in p.relative_to(ROOT).parts)]


def main() -> int:
    errs: list[str] = []
    counts = {name: 0 for name in ("H100 带宽", "H100 算力", "Qwen3-8B KV")}

    files = md_files()
    for md in files:
        for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                continue  # 代码块里的数字由 run_snippets 的 assert 管
            rel = md.relative_to(ROOT)
            for rule in NUMERIC:
                for m in re.finditer(rule["pattern"], line):
                    raw = m.group(1)
                    nb = rule.get("not_before")
                    if nb and re.search(nb, line[max(0, m.start() - 8):m.start()]):
                        continue
                    val = norm(raw)
                    if val not in rule["allowed"]:
                        errs.append(f"{rel}:{no} {rule['name']}写成 {raw}；{rule['note']}")
            for pat, hint in FORBIDDEN:
                if re.search(pat, line):
                    errs.append(f"{rel}:{no} 出现已知错误写法 {pat!r}；{hint}")

    text = "\n".join(md.read_text(encoding="utf-8") for md in files)
    for pat, least, name in PRESENCE:
        n = len(re.findall(pat, text))
        if n < least:
            errs.append(f"锚点事实「{name}」只出现 {n} 次，至少应有 {least} 次；可能被误删")

    counts["H100 带宽"] = len(re.findall(r"3\.35\s*TB/s", text))
    counts["H100 算力"] = len(re.findall(r"990\s*TFLOPS", text))
    counts["Qwen3-8B KV"] = len(re.findall(r"144\s*KiB", text))

    for e in errs:
        print(f"✗ {e}")
    print(f"检查 {len(files)} 个文件；共用数字出现次数：" +
          "、".join(f"{k} {v}" for k, v in counts.items()) +
          f"；{len(errs)} 个问题。")
    return min(len(errs), 100)


if __name__ == "__main__":
    sys.exit(main())
