#!/usr/bin/env python3
"""检查知识库里的外部链接能否打开。只用标准库。

用法：python3 tools/check_links.py [--timeout 15] [--workers 8]
网络错误、4xx、5xx 都会列出；退出码为失败的链接数（最多 100）。
部分站点拒绝脚本访问（返回 403 或 429），这类结果单独列为“需人工确认”，不计入失败。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL_RE = re.compile(r"\]\((https?://[^)\s]+)\)")
CODE_RE = re.compile(r"`[^`]*`")
SOFT = {401, 403, 405, 429}  # 常见的反爬或需要登录，不算死链
UA = "Mozilla/5.0 (compatible; agent-interview-kb-linkcheck)"


def collect() -> dict[str, list[str]]:
    urls: dict[str, list[str]] = {}
    for md in sorted(ROOT.rglob("*.md")):
        if any(p.startswith(".") for p in md.relative_to(ROOT).parts):
            continue
        fenced = False
        for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
            if fenced:
                continue
            for u in URL_RE.findall(CODE_RE.sub("", line)):  # 代码里的 URL 是示例，不检查
                urls.setdefault(u.split("#", 1)[0], []).append(f"{md.relative_to(ROOT)}:{no}")
    return urls


def probe(url: str, timeout: float) -> tuple[str, int | None, str]:
    res = _probe(url, timeout)
    if res[1] is None or res[1] >= 500:  # 5xx 和网络错误常是暂时的，隔一会儿再试一次
        time.sleep(3)
        res = _probe(url, timeout)
    return res


def _probe(url: str, timeout: float) -> tuple[str, int | None, str]:
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return url, r.status, ""
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (400, 403, 404, 405, 501):
                continue  # 有的站点不支持 HEAD，用 GET 再试一次
            return url, e.code, ""
        except Exception as e:  # noqa: BLE001
            if method == "HEAD":
                continue
            return url, None, type(e).__name__
    return url, None, "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    urls = collect()
    bad, soft = [], []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for url, code, err in ex.map(lambda u: probe(u, a.timeout), urls):
            if code is not None and code < 400:
                continue
            (soft if code in SOFT else bad).append((url, code, err))
    for title, rows in (("失败", bad), ("需人工确认", soft)):
        if rows:
            print(f"## {title}（{len(rows)}）")
            for url, code, err in rows:
                print(f"- {code or err} {url}  ← {', '.join(urls[url][:3])}")
    print(f"共检查 {len(urls)} 个链接，失败 {len(bad)}，需人工确认 {len(soft)}。")
    return min(len(bad), 100)


if __name__ == "__main__":
    sys.exit(main())
