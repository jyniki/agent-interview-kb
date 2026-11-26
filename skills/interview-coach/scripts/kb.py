"""读取知识库：章节目录、题目、方向定义。只用标准库。

知识库根目录的确定顺序：环境变量 AIKB_KB → 本文件所在仓库（解析软链接后）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

TRACKS = ("training", "harness", "evaluation", "inference")
TAGS = ("core",) + TRACKS
QID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d{3}$")
HEADING_RE = re.compile(r"^## (\S+)\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^- (level|tags):\s*(.+?)\s*$")
ANSWER_KEY = "**一句话答案**"
FOLLOWUP_KEY = "**追问链**"


def kb_root() -> Path:
    env = os.environ.get("AIKB_KB")
    if env:
        return Path(env).expanduser().resolve()
    # scripts/ → interview-coach/ → skills/ → 仓库根
    return Path(__file__).resolve().parents[3]


@dataclass
class Question:
    id: str
    title: str
    file: Path
    line: int
    level: int | None = None
    tags: list[str] = field(default_factory=list)
    body: str = ""
    chapter: str | None = None

    @property
    def answer(self) -> str:
        return _section(self.body, ANSWER_KEY)

    @property
    def followups(self) -> str:
        return _section(self.body, FOLLOWUP_KEY)


def _section(body: str, key: str) -> str:
    """取 **键**：之后到下一个 **xx**： 之前的文本。"""
    idx = body.find(key)
    if idx < 0:
        return ""
    rest = body[idx + len(key):].lstrip("：: ")
    m = re.search(r"\n\*\*[^*\n]+\*\*[：:]", rest)
    return (rest[: m.start()] if m else rest).strip()


def parse_questions(path: Path) -> tuple[list[Question], list[str]]:
    """解析一个 questions.md。返回 (题目列表, 格式错误列表)。"""
    questions: list[Question] = []
    errors: list[str] = []
    cur: Question | None = None
    body: list[str] = []
    in_code = False

    def flush():
        if cur is not None:
            cur.body = "\n".join(body).strip()
            questions.append(cur)

    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if raw.startswith("```"):
            in_code = not in_code
        m = HEADING_RE.match(raw) if not in_code else None
        if m and QID_RE.match(m.group(1)):
            flush()
            cur, body = Question(m.group(1), m.group(2), path, no), []
            continue
        if m and raw.startswith("## ") and re.search(r"-\d{3}$", m.group(1)):
            errors.append(f"{path}:{no} 题号格式不对：{m.group(1)}")
        if cur is None:
            continue
        f = FIELD_RE.match(raw)
        if f and not body:
            key, val = f.groups()
            if key == "level":
                cur.level = int(val) if val.isdigit() else -1
            else:
                cur.tags = [t.strip() for t in val.strip("[]").split(",") if t.strip()]
            continue
        body.append(raw)
    flush()
    return questions, errors


@dataclass
class Chapter:
    id: str          # 如 core/01-agent-architecture、training/03-grpo-family
    title: str
    minutes: int
    weight: int = 1
    tags: list[str] = field(default_factory=list)   # 设计题用：相关方向
    path: Path | None = None
    questions: list[Question] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.questions)


def chapter_dir(root: Path, cid: str) -> Path:
    group, name = cid.split("/", 1)
    if group in ("core", "design"):
        return root / "kb" / group / name
    return root / "kb" / "tracks" / group / name


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class KB:
    def __init__(self, root: Path | None = None):
        self.root = root or kb_root()
        self.chapters: dict[str, Chapter] = {}
        self.core_order: list[str] = []
        self.design_order: list[str] = []
        self.tracks: dict[str, dict] = {}
        self.questions: dict[str, Question] = {}
        self.coding: list[Question] = []
        self.errors: list[str] = []
        self._load()

    def _add_chapter(self, spec: dict) -> str:
        cid = spec["id"]
        ch = Chapter(cid, spec["title"], int(spec.get("minutes", 90)), int(spec.get("weight", 1)),
                     list(spec.get("tags", [])))
        ch.path = chapter_dir(self.root, cid)
        if not (ch.path / "README.md").exists():
            self.errors.append(f"章节 {cid} 缺少 {ch.path / 'README.md'}")
        qfile = ch.path / "questions.md"
        if qfile.exists():
            qs, errs = parse_questions(qfile)
            self.errors.extend(errs)
            for q in qs:
                q.chapter = cid
            ch.questions = qs
        self.chapters[cid] = ch
        return cid

    def _load(self):
        core = load_json(self.root / "kb" / "core" / "core.json")
        self.core_order = [self._add_chapter(s) for s in core["chapters"]]
        for t in TRACKS:
            tf = self.root / "kb" / "tracks" / t / "track.json"
            if not tf.exists():
                self.errors.append(f"缺少方向定义 {tf}")
                continue
            spec = load_json(tf)
            spec["chapter_ids"] = [self._add_chapter(s) for s in spec["chapters"]]
            for pre in spec.get("prerequisites", []):
                if pre not in self.core_order:
                    self.errors.append(f"{tf} 引用了不存在的前置章节 {pre}")
            self.tracks[t] = spec
        df = self.root / "kb" / "design" / "design.json"
        if df.exists():
            self.design_order = [self._add_chapter(s) for s in load_json(df)["chapters"]]
            for cid in self.design_order:
                bad = [t for t in self.chapters[cid].tags if t not in TAGS]
                if bad or not self.chapters[cid].tags:
                    self.errors.append(f"{df} 里 {cid} 的 tags 不合法：{self.chapters[cid].tags}")
        cfile = self.root / "kb" / "coding" / "questions.md"
        if cfile.exists():
            self.coding, errs = parse_questions(cfile)
            self.errors.extend(errs)
        for q in self.all_questions():
            if q.id in self.questions:
                other = self.questions[q.id]
                self.errors.append(f"题号重复 {q.id}：{other.file}:{other.line} 与 {q.file}:{q.line}")
            self.questions[q.id] = q

    def all_questions(self):
        for ch in self.chapters.values():
            yield from ch.questions
        yield from self.coding

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root))
