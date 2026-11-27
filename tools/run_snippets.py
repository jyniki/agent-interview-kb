#!/usr/bin/env python3
"""运行知识库里所有标成 ```python 的代码块，确认它们能跑、结尾的 assert 成立。

每个代码块在独立的子进程里运行，互不影响。只允许标准库和 numpy。
不需要运行的代码块写成 ```python no-run 或 ```text。

用法：
  python3 tools/run_snippets.py            运行 kb/ 下全部代码块
  python3 tools/run_snippets.py kb/core/01-llm-foundations/README.md   只跑指定文件
  python3 tools/run_snippets.py --list     只列出会运行的代码块
退出码为失败的代码块数（最多 100）。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 60


def snippets(md: Path):
    """返回 (起始行号, 代码) 列表。只收 ```python，跳过 ```python no-run。"""
    out, buf, start, active = [], [], 0, False
    for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not active and s.startswith("```"):
            info = s[3:].strip().split()
            if info[:1] == ["python"] and "no-run" not in info:
                active, buf, start = True, [], no
            elif s != "```":
                # 其他语言或 no-run 的代码块：跳到它的结束标记
                active = None
            elif active is None:
                active = False
            continue
        if active is None:
            if s == "```":
                active = False
            continue
        if active:
            if s == "```":
                out.append((start, "\n".join(buf)))
                active = False
            else:
                buf.append(line)
    return out


def run(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "snippet.py"
        f.write_text(code, encoding="utf-8")
        try:
            # -E -P 而不是 -I：不读 PYTHON* 环境变量、不把脚本目录加进 sys.path，
            # 但保留用户 site-packages，否则用 pip install --user 装的 numpy 会找不到。
            p = subprocess.run([sys.executable, "-E", "-P", str(f)], cwd=d, capture_output=True,
                               text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, f"超时（{TIMEOUT} 秒）"
    if p.returncode == 0:
        return True, ""
    tail = (p.stderr or p.stdout).strip().splitlines()[-6:]
    return False, "\n".join(tail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    files = [Path(f).resolve() for f in a.files] or sorted((ROOT / "kb").rglob("*.md"))
    # 手撕题的参考实现是独立 .py 文件，整份跑
    py_files = [] if a.files else sorted((ROOT / "kb" / "coding" / "solutions").glob("*.py"))
    total, failed = 0, []
    for md in files:
        for line, code in snippets(md):
            total += 1
            where = f"{md.relative_to(ROOT)}:{line}"
            if a.list:
                print(where)
                continue
            ok, msg = run(code)
            if not ok:
                failed.append((where, msg))
    if a.list:
        print(f"共 {total} 段（另有 {len(py_files)} 个参考实现文件）")
        return 0
    for py in py_files:
        total += 1
        ok, msg = run(py.read_text(encoding="utf-8"))
        if not ok:
            failed.append((str(py.relative_to(ROOT)), msg))
    for where, msg in failed:
        print(f"✗ {where}\n  " + msg.replace("\n", "\n  "))
    print(f"运行 {total} 段代码，失败 {len(failed)} 段。")
    return min(len(failed), 100)


if __name__ == "__main__":
    sys.exit(main())
