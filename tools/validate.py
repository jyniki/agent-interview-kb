#!/usr/bin/env python3
"""校验知识库：题目格式、题号唯一、章节文件存在、站内链接有效、skill 元数据合规。

用法：python3 tools/validate.py        有错误时退出码为 1
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "interview-coach" / "scripts"))
from kb import KB, TAGS  # noqa: E402

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
REQUIRED = ("**一句话答案**", "**要点**", "**追问链**")
PREFIX = {"core": "core-", "training": "train-", "harness": "harness-", "evaluation": "eval-",
          "inference": "infer-", "design": "design-", "code": "code-"}
ANSWER_MAX = 100
ALLOWED_KEYS = ["一句话答案", "要点", "追问链", "易错点"]
BOLD_KEY_RE = re.compile(r"^\*\*([^*\n]+)\*\*[：:]", re.M)
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def check_questions(kb: KB) -> list[str]:
    errs = []
    for q in kb.all_questions():
        where = f"{kb.rel(q.file)}:{q.line} {q.id}"
        if q.level not in (1, 2, 3):
            errs.append(f"{where} 缺少 level 或取值不在 1-3")
        if not q.tags:
            errs.append(f"{where} 缺少 tags")
        for t in q.tags:
            if t not in TAGS:
                errs.append(f"{where} 未知 tag：{t}（可选 {', '.join(TAGS)}）")
        for key in REQUIRED:
            if key not in q.body:
                errs.append(f"{where} 缺少 {key}")
        if not q.answer:
            errs.append(f"{where} 一句话答案为空")
        elif len(q.answer) > ANSWER_MAX:
            errs.append(f"{where} 一句话答案 {len(q.answer)} 字，超过 {ANSWER_MAX} 字上限（10 秒内要说完）")
        keys = BOLD_KEY_RE.findall(q.body)
        extra = [k for k in keys if k not in ALLOWED_KEYS]
        if extra:
            errs.append(f"{where} 出现了格式外的加粗字段 {extra}，会截断解析；正文小标题不要用 **xx**： 的写法")
        order = [k for k in keys if k in ALLOWED_KEYS]
        if order != sorted(order, key=ALLOWED_KEYS.index) or len(order) != len(set(order)):
            errs.append(f"{where} 字段顺序应为 一句话答案 → 要点 → 追问链 → 易错点，且不重复")
        for line in q.followups.splitlines():
            if re.match(r"^\s*\d+\.", line) and "→" not in line:
                errs.append(f"{where} 追问缺少 → 后的回答要点：{line.strip()[:40]}")
        group = q.chapter.split("/")[0] if q.chapter else "code"
        prefix = PREFIX[group]
        if not q.id.startswith(prefix):
            errs.append(f"{where} 题号应以 {prefix} 开头")
    return errs


_ANCHORS: dict = {}


def anchors(md: Path) -> set:
    """按 GitHub 的规则从标题生成锚点：小写、去掉标点、空格换成连字符、重名加 -1、-2。"""
    if md in _ANCHORS:
        return _ANCHORS[md]
    seen: dict = {}
    out = set()
    in_code = False
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        slug = re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        out.add(slug if n == 0 else f"{slug}-{n}")
        seen[slug] = n + 1
    _ANCHORS[md] = out
    return out


def check_links() -> list[str]:
    errs = []
    for md in ROOT.rglob("*.md"):
        if any(p.startswith(".") for p in md.relative_to(ROOT).parts):
            continue
        in_code = False
        for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for target in LINK_RE.findall(line):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                path, _, anchor = target.partition("#")
                dest = (md.parent / path) if path else md
                if not dest.exists():
                    errs.append(f"{md.relative_to(ROOT)}:{no} 链接指向不存在的文件：{target}")
                elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                    errs.append(f"{md.relative_to(ROOT)}:{no} 链接的锚点在目标文件里不存在：{target}")
    return errs


MODEL_RE = re.compile(r"(Opus|Sonnet|Haiku|Claude|GPT-?\d|o\d-|Gemini|Llama|Qwen|DeepSeek|Kimi|GLM|Mistral)", re.I)
SCORE_RE = re.compile(r"\d+(\.\d+)?\s*%")
SCORE_WORD_RE = re.compile(r"准确率|成功率|解决率|通过率|得分|分数|胜率|pass@|resolved|评测上|榜")


def check_scores() -> list[str]:
    """规则：不写模型在 Benchmark 上的分数和排名。同一行出现模型名、百分数和评分用语时报错。"""
    errs = []
    for md in (ROOT / "kb").rglob("*.md"):
        for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if MODEL_RE.search(line) and SCORE_RE.search(line) and SCORE_WORD_RE.search(line):
                errs.append(f"{md.relative_to(ROOT)}:{no} 疑似写了模型的评测分数（规则不允许）：{line.strip()[:80]}")
    return errs


def check_skills() -> list[str]:
    errs = []
    for sk in (ROOT / "skills").glob("*/SKILL.md"):
        text = sk.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        if not m:
            errs.append(f"{sk} 缺少 frontmatter")
            continue
        meta = dict(re.findall(r"^(\w[\w-]*):\s*(.+)$", m.group(1), re.M))
        name, desc = meta.get("name", ""), meta.get("description", "")
        if not SKILL_NAME_RE.match(name) or len(name) > 64 or name != sk.parent.name:
            errs.append(f"{sk} name 不合规或与目录名不一致：{name}")
        if not 1 <= len(desc) <= 1024:
            errs.append(f"{sk} description 长度应为 1-1024，现为 {len(desc)}")
        if len(text.splitlines()) > 500:
            errs.append(f"{sk} 超过 500 行，细节应移到 references/")
    return errs


def main() -> int:
    kb = KB(ROOT)
    errs = kb.errors + check_questions(kb) + check_links() + check_scores() + check_skills()
    ready = sum(1 for c in kb.chapters.values() if c.ready)
    n = sum(1 for _ in kb.all_questions())
    for e in errs:
        print(f"✗ {e}")
    print(f"{len(kb.chapters)} 章（{ready} 章有题），{n} 道题，{len(errs)} 个问题。")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
