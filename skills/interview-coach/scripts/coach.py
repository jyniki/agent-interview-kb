#!/usr/bin/env python3
"""interview-coach 的确定性部分：画像、排计划、每日任务、记录作答、按掌握程度排复习。

只用 Python 标准库。用户数据放在 $AIKB_HOME（默认 ~/.agent-interview-kb/），不写进知识库仓库。

子命令一览（详细参数用 -h 查看）：
  init      写入学习画像
  profile   查看画像
  diagnose  抽诊断题
  plan      排新学顺序和日期
  today     生成当天任务
  learned   标记章节学完，把本章题目放进复习队列
  next      取下一道要练的题（不含答案）
  show      显示题目；加 --answer 显示评分参考
  record    记录一次作答得分并重排
  debrief   记录真实面试被问到的题
  mock      按方向组一套模拟面试题
  stats     掌握情况统计
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb import KB, TRACKS, Question  # noqa: E402

PROFILE = "profile.json"
STATE = "state.json"
LOG = "log.jsonl"
REVIEW_MINUTES = 5          # 每道复习题按 5 分钟估
NEW_SHARE = 0.6             # 学习日里新学内容占的时间比例
REVIEW_SHARE = 0.3          # 学习日里复习占的比例，其余给手撕
REST_REVIEW_SHARE = 0.6     # 复习日里复习占的比例，其余给模拟或设计题
WEEKLY = 7
RETIRE_AFTER = 4            # 连续达标 4 次（含每周一次那轮）后停止复习


# ---------- 存储 ----------

def home() -> Path:
    p = Path(os.environ.get("AIKB_HOME", "~/.agent-interview-kb")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(name: str, default=None):
    f = home() / name
    if not f.exists():
        return default
    return json.loads(f.read_text(encoding="utf-8"))


def write_json(name: str, data) -> None:
    f = home() / name
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(f)


def append_log(entry: dict) -> None:
    with open(home() / LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_profile() -> dict:
    p = read_json(PROFILE)
    if p is None:
        die("还没有学习画像。先运行 init（见 references/intake.md）。")
    return p


SCHEMA = 1  # 用户数据格式版本；格式变化时递增并在 load_state 里迁移


def load_state() -> dict:
    s = read_json(STATE, {})
    v = s.get("schema", SCHEMA)
    if v > SCHEMA:
        die(f"{home() / STATE} 的格式版本是 {v}，比脚本支持的 {SCHEMA} 新。请先 git pull 更新知识库。")
    s["schema"] = SCHEMA
    s.setdefault("plan", [])
    s.setdefault("learned", {})
    s.setdefault("items", {})
    s.setdefault("custom", {})
    return s


def die(msg: str, code: int = 2):
    print(f"错误：{msg}", file=sys.stderr)
    sys.exit(code)


def parse_date(s: str | None) -> dt.date:
    if not s:
        return dt.date.today()
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        die(f"日期格式应为 YYYY-MM-DD：{s}")


def csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def out(data, as_json: bool, render) -> None:
    if as_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(render(data))


# ---------- 画像 ----------

def cmd_init(a, kb: KB):
    if (home() / PROFILE).exists() and not a.force:
        die(f"{home() / PROFILE} 已存在。要覆盖请加 --force；覆盖画像不会删除作答记录。")
    tracks = csv(a.tracks)
    bad = [t for t in tracks if t not in TRACKS]
    if bad or not tracks:
        die(f"--tracks 需从 {', '.join(TRACKS)} 中选，收到：{a.tracks}")
    primary = a.primary or tracks[0]
    if primary not in tracks:
        die("--primary 必须是 --tracks 之一")
    known_ids = set(kb.chapters)
    for opt in ("weak", "strong", "skip"):
        for x in csv(getattr(a, opt)):
            if x not in known_ids and x not in TRACKS and x != "core":
                die(f"--{opt} 里的 {x} 不是章节 ID 或方向名。章节 ID 用 `plan --list` 查看。")
    if not 30 <= a.daily_minutes <= 720:
        die("--daily-minutes 应在 30 到 720 之间")
    profile = {
        "tracks": tracks,
        "primary": primary,
        "daily_minutes": a.daily_minutes,
        "rest_weekdays": [int(x) for x in csv(a.rest_weekdays)],
        "start": parse_date(a.start).isoformat(),
        "interview": parse_date(a.interview).isoformat() if a.interview else None,
        "weak": csv(a.weak),
        "strong": csv(a.strong),
        "skip": csv(a.skip),
        "background": a.background or "",
        "created": dt.date.today().isoformat(),
    }
    write_json(PROFILE, profile)
    print(f"已写入 {home() / PROFILE}")
    print(json.dumps(profile, ensure_ascii=False, indent=2))


def cmd_profile(a, kb: KB):
    print(json.dumps(load_profile(), ensure_ascii=False, indent=2))


def level_of(profile: dict, kb: KB, cid: str) -> str:
    """章节对这个用户属于 weak / strong / normal。可以按章节 ID 或方向名标注。"""
    group = cid.split("/", 1)[0]
    for key in ("weak", "strong"):
        marks = profile.get(key, [])
        if cid in marks or group in marks:
            return key
    return "normal"


# ---------- 计划 ----------

def chapter_sequence(profile: dict, kb: KB) -> list[str]:
    """新学顺序：先通用层（方向前置的章节排前面），再按主方向优先交替排各方向章节。"""
    skip = set(profile.get("skip", []))
    prereq: list[str] = []
    for t in profile["tracks"]:
        for c in kb.tracks[t].get("prerequisites", []):
            if c not in prereq:
                prereq.append(c)
    core = prereq + [c for c in kb.core_order if c not in prereq]
    core = [c for c in core if c not in skip]

    lists = {t: [c for c in kb.tracks[t]["chapter_ids"] if c not in skip] for t in profile["tracks"]}
    order = [profile["primary"]] + [t for t in profile["tracks"] if t != profile["primary"]]
    tracks_seq: list[str] = []
    # 主方向每轮排 2 章，其他方向每轮 1 章
    while any(lists.values()):
        for t in order:
            take = 2 if t == profile["primary"] else 1
            for _ in range(take):
                if lists[t]:
                    tracks_seq.append(lists[t].pop(0))
    picked = set(profile["tracks"])
    design = [c for c in kb.design_order if c not in skip and picked & set(kb.chapters[c].tags)]
    return core + tracks_seq + design


def chapter_minutes(profile: dict, kb: KB, cid: str) -> int:
    m = kb.chapters[cid].minutes
    lv = level_of(profile, kb, cid)
    if lv == "strong":
        return max(30, m // 2)
    if lv == "weak":
        return int(m * 1.5)
    return m


def is_rest(profile: dict, day: dt.date) -> bool:
    return day.weekday() in profile.get("rest_weekdays", [])


def build_plan(profile: dict, kb: KB, start: dt.date, learned: dict) -> tuple[list[dict], list[str]]:
    budget = max(15, int(profile["daily_minutes"] * NEW_SHARE))
    seq = [c for c in chapter_sequence(profile, kb) if c not in learned]
    plan, warnings = [], []
    day, left = start, budget
    while is_rest(profile, day):
        day += dt.timedelta(days=1)
    for cid in seq:
        need = chapter_minutes(profile, kb, cid)
        first = day
        while True:
            use = min(need, left)
            need -= use
            left -= use
            if need <= 0:
                break
            day += dt.timedelta(days=1)
            while is_rest(profile, day):
                day += dt.timedelta(days=1)
            left = budget
        ch = kb.chapters[cid]
        plan.append({
            "chapter": cid,
            "title": ch.title,
            "start": first.isoformat(),
            "end": day.isoformat(),
            "minutes": chapter_minutes(profile, kb, cid),
            "level": level_of(profile, kb, cid),
            "ready": ch.ready,
        })
        if left <= 10:
            day += dt.timedelta(days=1)
            while is_rest(profile, day):
                day += dt.timedelta(days=1)
            left = budget
    if profile.get("interview") and plan:
        iv = dt.date.fromisoformat(profile["interview"])
        last = dt.date.fromisoformat(plan[-1]["end"])
        if last >= iv:
            late = [p["chapter"] for p in plan if dt.date.fromisoformat(p["end"]) >= iv]
            warnings.append(
                f"按每天 {profile['daily_minutes']} 分钟，新学要排到 {last}，晚于面试日 {iv}。"
                f"有 {len(late)} 章排不进：{', '.join(late)}。"
                "可以增加每天时长、把已掌握的章节加入 --skip，或把非主方向章节放到面试后。"
            )
        elif (iv - last).days < 7:
            warnings.append(f"新学在 {last} 结束，离面试只剩 {(iv - last).days} 天做全量复习和模拟，偏紧。")
    return plan, warnings


def cmd_plan(a, kb: KB):
    if a.list:
        rows = [f"{cid}\t{kb.chapters[cid].title}\t{'有题' if kb.chapters[cid].ready else '待补'}"
                for cid in kb.chapters]
        print("\n".join(rows))
        return
    profile = load_profile()
    state = load_state()
    start = parse_date(a.start) if a.start else max(parse_date(profile["start"]), parse_date(a.date))
    plan, warnings = build_plan(profile, kb, start, state["learned"])
    if not a.dry_run:
        state["plan"] = plan
        state["plan_made"] = parse_date(a.date).isoformat()
        write_json(STATE, state)

    def render(d):
        lines = [f"# 学习计划（{len(d['plan'])} 章，{d['plan'][0]['start'] if d['plan'] else '-'} 起）", ""]
        lines.append("| 开始 | 结束 | 章节 | 分钟 | 程度 | 内容 |")
        lines.append("|---|---|---|---|---|---|")
        for p in d["plan"]:
            lines.append(f"| {p['start']} | {p['end']} | {p['chapter']} {p['title']} | {p['minutes']} | "
                         f"{ {'weak': '弱项', 'strong': '强项', 'normal': '一般'}[p['level']] } | "
                         f"{'有题' if p['ready'] else '待补'} |")
        for w in d["warnings"]:
            lines.append(f"\n> 注意：{w}")
        return "\n".join(lines)

    out({"plan": plan, "warnings": warnings}, a.json, render)


# ---------- 复习排期 ----------

def schedule(item: dict, score: int, day: dt.date, interview: str | None) -> dict:
    """按得分更新一道题的复习状态。规则见 references/scoring.md。"""
    item = dict(item)
    item["reps"] = item.get("reps", 0) + 1
    item["last_score"] = score
    item["last"] = day.isoformat()
    item["retired"] = False
    if score <= 0:
        item["streak"], item["interval"] = 0, 1
    elif score == 1:
        item["streak"], item["interval"] = 0, 3
    else:
        item["streak"] = item.get("streak", 0) + 1
        prev = item.get("interval", 0)
        if item["streak"] >= RETIRE_AFTER:
            item["retired"] = True
        elif item["streak"] >= 3:
            item["interval"] = WEEKLY
        else:
            item["interval"] = max(2, prev * 2)
    if score == 2:
        item.pop("priority", None)  # 面试真题答到达标后，回到普通排期
    if item["retired"]:
        item["due"] = None
        return item
    due = day + dt.timedelta(days=item["interval"])
    if interview:
        iv = dt.date.fromisoformat(interview)
        # 面试前最后一天之后就没有意义了：最晚排在面试前一天
        if day < iv - dt.timedelta(days=1) < due:
            due = iv - dt.timedelta(days=1)
    item["due"] = due.isoformat()
    return item


def ensure_item(state: dict, qid: str, due: dt.date) -> dict:
    it = state["items"].get(qid)
    if it is None:
        it = {"due": due.isoformat(), "interval": 0, "streak": 0, "reps": 0}
        state["items"][qid] = it
    return it


def due_items(state: dict, day: dt.date) -> list[tuple[str, dict]]:
    rows = [(q, it) for q, it in state["items"].items()
            if not it.get("retired") and it.get("due") and dt.date.fromisoformat(it["due"]) <= day]
    # 面试真题优先，然后是上次得分低的，再按到期早晚
    rows.sort(key=lambda r: (0 if r[1].get("priority") else 1,
                             r[1].get("last_score", -1) if r[1].get("reps") else 3,
                             r[1]["due"]))
    return rows


# ---------- 每日任务 ----------

def describe(kb: KB, qid: str, state: dict) -> dict:
    q = kb.questions.get(qid)
    if q is None:
        c = state["custom"].get(qid)
        if c:
            return {"id": qid, "title": c["text"], "where": "真实面试题（自己整理答案）", "chapter": None}
        return {"id": qid, "title": "（题目已从知识库删除）", "where": "", "chapter": None}
    return {"id": qid, "title": q.title, "where": f"{kb.rel(q.file)}:{q.line}",
            "chapter": q.chapter, "level": q.level}


def pick_coding(kb: KB, state: dict, day: dt.date) -> dict | None:
    if not kb.coding:
        return None
    items = state["items"]
    due = [q for q in kb.coding if q.id in items and not items[q.id].get("retired")
           and items[q.id].get("due") and dt.date.fromisoformat(items[q.id]["due"]) <= day]
    fresh = [q for q in kb.coding if q.id not in items]
    q = (due or fresh or [None])[0]
    if q is None:
        return None
    return {"id": q.id, "title": q.title, "where": f"{kb.rel(q.file)}:{q.line}", "new": q.id not in items}


def pick_design(kb: KB, profile: dict, state: dict) -> dict | None:
    """复习日推荐一道设计题：只在与所选方向相关、已学完的设计题里选，优先没练过和上次得分低的。"""
    picked = set(profile["tracks"])
    cands = [c for c in kb.design_order if c in state["learned"] and picked & set(kb.chapters[c].tags)]
    if not cands:
        return None

    def key(cid):
        scores = [state["items"].get(q.id, {}).get("last_score", -1) for q in kb.chapters[cid].questions]
        return (min(scores) if scores else -1, state["learned"][cid])

    cid = sorted(cands, key=key)[0]
    ch = kb.chapters[cid]
    return {"chapter": cid, "title": ch.title, "readme": kb.rel(ch.path / "README.md"),
            "minutes": ch.minutes}


def cmd_today(a, kb: KB):
    profile = load_profile()
    state = load_state()
    day = parse_date(a.date)
    rest = is_rest(profile, day)
    daily = profile["daily_minutes"]
    result = {"date": day.isoformat(), "weekday": "一二三四五六日"[day.weekday()], "rest_day": rest,
              "new": [], "behind_days": 0, "reviews": [], "review_overflow": 0, "coding": None,
              "mock": None, "notes": []}

    if not state["plan"]:
        result["notes"].append("还没有计划：先运行 plan。")
    elif not rest:
        pending = [p for p in state["plan"] if p["chapter"] not in state["learned"]]
        if pending:
            budget = int(daily * NEW_SHARE)
            for p in pending:
                if dt.date.fromisoformat(p["start"]) > day and result["new"]:
                    break
                mode = {"weak": "先看本章示例和推导，再合书作答",
                        "strong": "快速过一遍，直接闭卷做题",
                        "normal": "精读后合书口述要点，再做题"}[p["level"]]
                result["new"].append({**p, "mode": mode,
                                      "readme": kb.rel(kb.chapters[p["chapter"]].path / "README.md")})
                budget -= p["minutes"]
                if budget <= 0 or dt.date.fromisoformat(p["end"]) > day:
                    break
            planned = dt.date.fromisoformat(pending[0]["start"])
            result["behind_days"] = max(0, (day - planned).days)
            if result["behind_days"] > 3:
                result["notes"].append(f"比计划落后 {result['behind_days']} 天，建议运行 `plan` 从今天重排。")
        else:
            result["notes"].append("计划里的章节都学完了，进入全量复习和模拟阶段。")

    share = REST_REVIEW_SHARE if rest else REVIEW_SHARE
    cap = max(3, int(daily * share / REVIEW_MINUTES))
    if not state["plan"] or not [p for p in state["plan"] if p["chapter"] not in state["learned"]]:
        cap = max(cap, int(daily * 0.7 / REVIEW_MINUTES))
    due = [r for r in due_items(state, day) if not r[0].startswith("code-")]
    result["reviews"] = [{**describe(kb, q, state), "last_score": it.get("last_score"),
                          "due": it["due"], "priority": bool(it.get("priority"))} for q, it in due[:cap]]
    result["review_overflow"] = max(0, len(due) - cap)
    if not rest:
        result["coding"] = pick_coding(kb, state, day)
    else:
        result["mock"] = "做一次 30-45 分钟模拟面试：运行 `mock`，流程见 references/mock.md。"
        d = pick_design(kb, profile, state)
        if d:
            result["design"] = d
    if profile.get("interview"):
        left = (dt.date.fromisoformat(profile["interview"]) - day).days
        if left >= 0:
            result["notes"].append(f"距面试 {left} 天。")

    def render(d):
        L = [f"# {d['date']}（周{d['weekday']}）{'复习日' if d['rest_day'] else '学习日'}", ""]
        if d["new"]:
            L.append("## 新学")
            for n in d["new"]:
                tag = "" if n["ready"] else "（本章题目待补：读 README 里的资料，学完口述要点）"
                L.append(f"- {n['chapter']} {n['title']}，约 {n['minutes']} 分钟。{n['mode']}。{tag}")
                L.append(f"  - 原文：{n['readme']}")
            L.append("- 学完运行 `learned <章节ID>`，本章题目会进入复习队列，当天先练一轮。")
            L.append("")
        L.append(f"## 复习（{len(d['reviews'])} 题）")
        if not d["reviews"]:
            L.append("- 今天没有到期的题。")
        for r in d["reviews"]:
            last = {None: "新题", 0: "上次 0 分", 1: "上次 1 分", 2: "上次 2 分"}[r.get("last_score")]
            star = "【面试真题】" if r["priority"] else ""
            L.append(f"- {star}{r['id']} {r['title']}（{last}）")
        if d["review_overflow"]:
            L.append(f"- 另有 {d['review_overflow']} 题已到期，时间够就继续 `next`。")
        L.append("")
        if d["coding"]:
            c = d["coding"]
            L.append("## 手撕")
            L.append(f"- {c['id']} {c['title']}（{'新题' if c['new'] else '到期复习'}）：{c['where']}")
            L.append("")
        if d["mock"]:
            L.append("## 模拟面试")
            L.append(f"- {d['mock']}")
            if d.get("design"):
                g = d["design"]
                L.append(f"- 或者限时 {g['minutes']} 分钟白板一道设计题：{g['chapter']} {g['title']}（{g['readme']}），流程见 references/design.md。")
            L.append("")
        for n in d["notes"]:
            L.append(f"> {n}")
        return "\n".join(L).rstrip()

    if result.get("design") is None:
        result.pop("design", None)
    out(result, a.json, render)


def cmd_learned(a, kb: KB):
    state = load_state()
    day = parse_date(a.date)
    for cid in a.chapters:
        if cid not in kb.chapters:
            die(f"没有章节 {cid}。用 `plan --list` 查看章节 ID。")
        state["learned"][cid] = day.isoformat()
        n = 0
        for q in kb.chapters[cid].questions:
            if q.id not in state["items"]:
                ensure_item(state, q.id, day)
                n += 1
        print(f"{cid} 已标记学完；{n} 道新题进入复习队列，今天到期。")
    write_json(STATE, state)


def cmd_next(a, kb: KB):
    state = load_state()
    day = parse_date(a.date)
    skip = set(csv(a.exclude))
    rows = [r for r in due_items(state, day) if r[0] not in skip]
    if a.chapter:
        rows = [r for r in rows if (kb.questions.get(r[0]) and kb.questions[r[0]].chapter == a.chapter)]
    if not a.coding:
        rows = [r for r in rows if not r[0].startswith("code-")]
    if not rows:
        out({"next": None}, a.json, lambda d: "今天到期的题都练完了。")
        return
    qid, it = rows[0]
    d = describe(kb, qid, state)
    d.update({"last_score": it.get("last_score"), "remaining": len(rows) - 1})
    out({"next": d}, a.json,
        lambda x: f"{x['next']['id']} {x['next']['title']}\n位置：{x['next']['where']}\n"
                  f"今天还剩 {x['next']['remaining']} 题")


def cmd_show(a, kb: KB):
    q = kb.questions.get(a.qid)
    if q is None:
        die(f"没有题目 {a.qid}")
    data = {"id": q.id, "title": q.title, "level": q.level, "tags": q.tags,
            "where": f"{kb.rel(q.file)}:{q.line}", "chapter": q.chapter}
    if a.answer:
        data["answer"] = q.answer
        data["followups"] = q.followups
        data["body"] = q.body

    def render(d):
        s = f"{d['id']} {d['title']}\n难度：{d['level']}  位置：{d['where']}"
        if "body" in d:
            s += "\n\n" + d["body"]
        return s

    out(data, a.json, render)


def cmd_record(a, kb: KB):
    if a.qid not in kb.questions and not a.qid.startswith("custom-"):
        die(f"没有题目 {a.qid}")
    if a.score not in (0, 1, 2):
        die("得分只能是 0、1、2")
    profile = read_json(PROFILE, {}) or {}
    state = load_state()
    day = parse_date(a.date)
    it = ensure_item(state, a.qid, day)
    new = schedule(it, a.score, day, profile.get("interview"))
    if a.source == "interview":
        new["priority"] = True
        new["due"] = (day + dt.timedelta(days=1)).isoformat()
        new["retired"] = False
    state["items"][a.qid] = new
    write_json(STATE, state)
    append_log({"date": day.isoformat(), "qid": a.qid, "score": a.score,
                "source": a.source, "note": a.note or ""})
    nxt = "不再安排复习" if new.get("retired") else f"下次 {new['due']}"
    print(f"{a.qid} 记 {a.score} 分，{nxt}（连续达标 {new.get('streak', 0)} 次）。")


def cmd_debrief(a, kb: KB):
    state = load_state()
    day = parse_date(a.date)
    if a.qid:
        if a.qid not in kb.questions:
            die(f"没有题目 {a.qid}")
        qid = a.qid
    else:
        qid = f"custom-{len(state['custom']) + 1:03d}"
        state["custom"][qid] = {"text": a.text, "company": a.company or "", "date": day.isoformat()}
    write_json(STATE, state)
    a2 = argparse.Namespace(qid=qid, score=a.score, date=a.date, source="interview",
                            note=(a.note or "") + (f"｜{a.company}" if a.company else ""))
    cmd_record(a2, kb)


def cmd_mock(a, kb: KB):
    profile = load_profile()
    state = load_state()
    track = a.track or profile["primary"]
    if track not in TRACKS:
        die(f"--track 需从 {', '.join(TRACKS)} 中选")
    if getattr(a, "design", False):
        cands = [c for c in kb.design_order if track in kb.chapters[c].tags]
        if not cands:
            die(f"{track} 方向还没有设计题。")
        def dkey(cid):
            sc = [state["items"].get(q.id, {}).get("last_score", -1) for q in kb.chapters[cid].questions]
            return min(sc) if sc else -1
        cid = sorted(cands, key=dkey)[0]
        ch = kb.chapters[cid]
        data = {"track": track, "design": {"chapter": cid, "title": ch.title, "minutes": ch.minutes,
                "readme": kb.rel(ch.path / "README.md"),
                "questions": [describe(kb, q.id, state) for q in ch.questions]}}
        out(data, a.json, lambda d: f"# 设计题模拟：{d['design']['title']}（限时 {d['design']['minutes']} 分钟）\n"
            f"题目与评分参考：{d['design']['readme']}\n流程见 references/design.md。")
        return
    pool_core = [q for c in kb.core_order for q in kb.chapters[c].questions]
    pool_track = [q for c in kb.tracks[track]["chapter_ids"] for q in kb.chapters[c].questions]
    tagged = [q for q in pool_core if track in q.tags]

    def score_key(q: Question):
        it = state["items"].get(q.id, {})
        # 没练过、上次分低的排前面；难度高的排前面
        return (it.get("last_score", -1) if it.get("reps") else -1, -(q.level or 1))

    picked: list[Question] = []
    for pool, k in ((pool_track, math.ceil(a.n * 0.5)), (tagged, math.ceil(a.n * 0.3)), (pool_core, a.n)):
        for q in sorted(pool, key=score_key):
            if len(picked) >= a.n or k <= 0:
                break
            if q not in picked:
                picked.append(q)
                k -= 1
    data = {"track": track, "questions": [describe(kb, q.id, state) for q in picked]}
    notes = []
    if not pool_track:
        notes.append(f"{track} 方向还没有题目，这套题全部来自通用层。")
    if len(picked) < a.n:
        notes.append(f"题库里可用的题只有 {len(picked)} 道，少于要求的 {a.n} 道。")
    if notes:
        data["note"] = "".join(notes)
    out(data, a.json, lambda d: "\n".join(
        [f"# 模拟面试：{d['track']}（{len(d['questions'])} 题）"] +
        [f"{i}. {q['id']} {q['title']}" for i, q in enumerate(d["questions"], 1)] +
        ([f"\n> {d['note']}"] if d.get("note") else [])))


def cmd_diagnose(a, kb: KB):
    profile = load_profile()
    chapters = [c for c in chapter_sequence(profile, kb) if kb.chapters[c].ready]
    picked = []
    for c in chapters:
        qs = sorted(kb.chapters[c].questions, key=lambda q: abs((q.level or 2) - 2))
        if qs:
            picked.append(qs[0])
        if len(picked) >= a.n:
            break
    state = load_state()
    out({"questions": [describe(kb, q.id, state) for q in picked]}, a.json,
        lambda d: "\n".join(f"- {q['chapter']}：{q['id']} {q['title']}" for q in d["questions"])
        or "知识库里还没有可用的诊断题。")


def cmd_stats(a, kb: KB):
    state = load_state()
    day = parse_date(a.date)
    profile = read_json(PROFILE, {}) or {}
    rows = []
    for cid in (chapter_sequence(profile, kb) if profile else list(kb.chapters)):
        ch = kb.chapters[cid]
        if not ch.questions:
            continue
        its = [state["items"].get(q.id) for q in ch.questions]
        tried = [i for i in its if i and i.get("reps")]
        rows.append({
            "chapter": cid, "title": ch.title, "total": len(ch.questions), "tried": len(tried),
            "solid": sum(1 for i in tried if i.get("last_score") == 2),
            "weak": sum(1 for i in tried if i.get("last_score") == 0),
            "retired": sum(1 for i in tried if i.get("retired")),
            "learned": state["learned"].get(cid),
        })
    days = set()
    log = home() / LOG
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                days.add(json.loads(line)["date"])
    streak, d = 0, day
    while d.isoformat() in days:
        streak += 1
        d -= dt.timedelta(days=1)
    worst = sorted(((q, it) for q, it in state["items"].items() if it.get("reps")),
                   key=lambda r: (r[1].get("last_score", 3), r[1].get("streak", 0)))[:5]
    data = {"chapters": rows, "practice_days": len(days), "streak": streak,
            "due_today": len(due_items(state, day)),
            "weakest": [{**describe(kb, q, state), "last_score": it.get("last_score")} for q, it in worst
                        if it.get("last_score", 2) < 2]}

    def render(d):
        L = [f"练习 {d['practice_days']} 天，连续 {d['streak']} 天；今天到期 {d['due_today']} 题。", "",
             "| 章节 | 题数 | 已练 | 达标 | 0 分 | 已停复习 | 学完日期 |", "|---|---|---|---|---|---|---|"]
        for r in d["chapters"]:
            L.append(f"| {r['chapter']} | {r['total']} | {r['tried']} | {r['solid']} | {r['weak']} | "
                     f"{r['retired']} | {r['learned'] or '-'} |")
        if d["weakest"]:
            L += ["", "最需要补的题："] + [f"- {w['id']} {w['title']}（上次 {w['last_score']} 分）"
                                          for w in d["weakest"]]
        return "\n".join(L)

    out(data, a.json, render)


# ---------- 入口 ----------

def main(argv=None):
    p = argparse.ArgumentParser(prog="coach.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--date", help="按这一天计算，默认今天（YYYY-MM-DD）")
        sp.add_argument("--json", action="store_true", help="输出 JSON")
        sp.set_defaults(fn=fn)
        return sp

    sp = add("init", cmd_init, "写入学习画像")
    sp.add_argument("--tracks", required=True, help=f"逗号分隔，可选 {','.join(TRACKS)}")
    sp.add_argument("--primary", help="主方向，默认 --tracks 的第一个")
    sp.add_argument("--daily-minutes", type=int, required=True, help="每天可用分钟数")
    sp.add_argument("--rest-weekdays", default="6", help="复习日（不学新内容），0=周一…6=周日，逗号分隔；传空串表示没有")
    sp.add_argument("--start", help="开始日期，默认今天")
    sp.add_argument("--interview", help="预计面试日期")
    sp.add_argument("--weak", help="弱项：章节 ID 或方向名，逗号分隔")
    sp.add_argument("--strong", help="强项：章节 ID 或方向名，逗号分隔")
    sp.add_argument("--skip", help="已掌握、不用学的章节 ID，逗号分隔")
    sp.add_argument("--background", help="一句话背景（只存本地）")
    sp.add_argument("--force", action="store_true")

    add("profile", cmd_profile, "查看画像")

    sp = add("diagnose", cmd_diagnose, "抽诊断题")
    sp.add_argument("-n", type=int, default=8)

    sp = add("plan", cmd_plan, "排新学计划")
    sp.add_argument("--start", help="从这天开始排，默认今天和画像开始日期中较晚的")
    sp.add_argument("--dry-run", action="store_true", help="只显示，不保存")
    sp.add_argument("--list", action="store_true", help="列出全部章节 ID")

    add("today", cmd_today, "生成当天任务")

    sp = add("learned", cmd_learned, "标记章节学完")
    sp.add_argument("chapters", nargs="+")

    sp = add("next", cmd_next, "取下一道要练的题")
    sp.add_argument("--chapter")
    sp.add_argument("--exclude", help="跳过这些题号，逗号分隔")
    sp.add_argument("--coding", action="store_true", help="包括手撕题")

    sp = add("show", cmd_show, "显示题目")
    sp.add_argument("qid")
    sp.add_argument("--answer", action="store_true", help="显示评分参考（作答之后才用）")

    sp = add("record", cmd_record, "记录得分")
    sp.add_argument("qid")
    sp.add_argument("score", type=int)
    sp.add_argument("--note", help="卡住的点")
    sp.add_argument("--source", choices=("drill", "mock", "interview"), default="drill")

    sp = add("debrief", cmd_debrief, "记录真实面试题")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--qid", help="知识库里对应的题号")
    g.add_argument("--text", help="知识库里没有时，写下题目原文")
    sp.add_argument("--score", type=int, default=1, help="当时答得怎样，0/1/2，默认 1")
    sp.add_argument("--company", help="公司或轮次（只存本地）")
    sp.add_argument("--note")

    sp = add("mock", cmd_mock, "组模拟面试题")
    sp.add_argument("--track")
    sp.add_argument("--design", action="store_true", help="改为抽一道系统设计整题")
    sp.add_argument("-n", type=int, default=6)

    add("stats", cmd_stats, "掌握情况统计")

    a = p.parse_args(argv)
    kb = KB()
    if kb.errors and a.cmd not in ("profile",):
        print("知识库有格式问题（运行 tools/validate.py 查看）：" + kb.errors[0], file=sys.stderr)
    a.fn(a, kb)


if __name__ == "__main__":
    main()
