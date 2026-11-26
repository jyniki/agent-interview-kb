import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills" / "interview-coach" / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

import coach  # noqa: E402
import validate  # noqa: E402
from kb import KB, parse_questions  # noqa: E402

D = dt.date(2026, 12, 1)  # 周二


def run(*argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coach.main(list(argv))
    return buf.getvalue()


def run_json(*argv):
    return json.loads(run(*argv, "--json"))


class ScheduleTest(unittest.TestCase):
    def test_zero_next_day(self):
        it = coach.schedule({}, 0, D, None)
        self.assertEqual(it["due"], "2026-12-02")
        self.assertEqual(it["streak"], 0)

    def test_one_three_days(self):
        self.assertEqual(coach.schedule({}, 1, D, None)["due"], "2026-12-04")

    def test_two_doubles_then_weekly_then_retire(self):
        it, day, intervals = {}, D, []
        for _ in range(4):
            it = coach.schedule(it, 2, day, None)
            if it["retired"]:
                break
            intervals.append(it["interval"])
            day = dt.date.fromisoformat(it["due"])
        self.assertEqual(intervals, [2, 4, 7])
        self.assertTrue(it["retired"])
        self.assertIsNone(it["due"])

    def test_failure_resets_streak(self):
        it = coach.schedule({}, 2, D, None)
        it = coach.schedule(it, 2, D, None)
        it = coach.schedule(it, 1, D, None)
        self.assertEqual(it["streak"], 0)
        self.assertEqual(it["interval"], 3)

    def test_capped_before_interview(self):
        it = {"interval": 30, "streak": 1}
        it = coach.schedule(it, 2, D, "2026-12-10")
        self.assertEqual(it["due"], "2026-12-09")


class ParseTest(unittest.TestCase):
    def test_parse_fields_and_sections(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "q.md"
            p.write_text(
                "# t\n\n## core-x-001 标题一\n- level: 2\n- tags: [core, harness]\n\n"
                "**一句话答案**：答案一。\n\n**要点**：\n- a\n\n**追问链**：\n1. 追问\n\n"
                "```\n## core-x-999 代码块里的不算\n```\n"
                "## core-x-002 标题二\n- level: 1\n- tags: [core]\n\n**一句话答案**：答案二\n",
                encoding="utf-8")
            qs, errs = parse_questions(p)
            self.assertEqual(errs, [])
            self.assertEqual([q.id for q in qs], ["core-x-001", "core-x-002"])
            self.assertEqual(qs[0].level, 2)
            self.assertEqual(qs[0].tags, ["core", "harness"])
            self.assertEqual(qs[0].answer, "答案一。")
            self.assertIn("追问", qs[0].followups)


class RepoTest(unittest.TestCase):
    def test_repo_validates(self):
        kb = KB(ROOT)
        errs = kb.errors + validate.check_questions(kb) + validate.check_links() + validate.check_scores() + validate.check_skills()
        self.assertEqual(errs, [])


class FlowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AIKB_HOME"] = self.tmp.name
        os.environ.pop("AIKB_KB", None)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("AIKB_HOME", None)

    def init(self, *extra):
        run("init", "--tracks", "training,harness", "--daily-minutes", "180",
            "--start", D.isoformat(), "--interview", "2027-01-20", *extra)

    def test_plan_order_and_rest_days(self):
        self.init("--weak", "core/12-rl-basics", "--skip", "core/01-llm-foundations")
        plan = run_json("plan", "--date", D.isoformat())["plan"]
        ids = [p["chapter"] for p in plan]
        self.assertNotIn("core/01-llm-foundations", ids)
        # 前置章节排在其他通用章节之前
        self.assertLess(ids.index("core/11-post-training-basics"), ids.index("core/06-memory"))
        # 通用层在垂类之前；主方向先出现
        first_track = next(i for i, c in enumerate(ids) if not c.startswith("core/"))
        self.assertTrue(all(c.startswith("core/") for c in ids[:first_track]))
        self.assertTrue(ids[first_track].startswith("training/"))
        rl = next(p for p in plan if p["chapter"] == "core/12-rl-basics")
        # 弱项按章节时长的 1.5 倍排；时长从 core.json 读
        base = KB(ROOT).chapters["core/12-rl-basics"].minutes
        self.assertEqual(rl["minutes"], int(base * 1.5))
        for p in plan:
            for key in ("start", "end"):
                self.assertNotEqual(dt.date.fromisoformat(p[key]).weekday(), 6)

    def test_plan_warns_when_too_late(self):
        run("init", "--tracks", "training,harness,evaluation,inference", "--daily-minutes", "60",
            "--start", D.isoformat(), "--interview", "2026-12-20")
        res = run_json("plan", "--date", D.isoformat())
        self.assertTrue(res["warnings"])

    def test_full_flow(self):
        self.init()
        run("plan", "--date", D.isoformat())
        today = run_json("today", "--date", D.isoformat())
        self.assertTrue(today["new"])
        self.assertEqual(today["reviews"], [])
        self.assertIsNotNone(today["coding"])

        out = run("learned", "core/02-agent-architecture", "--date", D.isoformat())
        n = len(KB(ROOT).chapters["core/02-agent-architecture"].questions)
        self.assertIn(f"{n} 道新题", out)
        nxt = run_json("next", "--date", D.isoformat())["next"]
        self.assertEqual(nxt["id"], "core-agent-001")
        self.assertNotIn("answer", nxt)

        run("record", "core-agent-001", "0", "--date", D.isoformat())
        run("record", "core-agent-002", "1", "--date", D.isoformat())
        run("record", "core-agent-003", "2", "--date", D.isoformat())
        state = json.loads((Path(self.tmp.name) / "state.json").read_text())
        self.assertEqual(state["items"]["core-agent-001"]["due"], "2026-12-02")
        self.assertEqual(state["items"]["core-agent-002"]["due"], "2026-12-04")
        self.assertEqual(state["items"]["core-agent-003"]["due"], "2026-12-03")
        log = (Path(self.tmp.name) / "log.jsonl").read_text().splitlines()
        self.assertEqual(len(log), 3)

        later = run_json("today", "--date", "2026-12-02")
        ids = [r["id"] for r in later["reviews"]]
        self.assertIn("core-agent-001", ids)
        self.assertNotIn("core-agent-002", ids)
        # 没练过的新题和 0 分题在前面
        self.assertEqual(later["reviews"][0]["id"], "core-agent-001")

        stats = run_json("stats", "--date", "2026-12-02")
        ch = next(c for c in stats["chapters"] if c["chapter"] == "core/02-agent-architecture")
        self.assertEqual((ch["tried"], ch["solid"], ch["weak"]), (3, 1, 1))

    def test_debrief_priority(self):
        self.init()
        run("debrief", "--text", "讲讲 MoE 路由", "--score", "0", "--company", "某公司",
            "--date", D.isoformat())
        run("learned", "core/02-agent-architecture", "--date", D.isoformat())
        today = run_json("today", "--date", "2026-12-02")
        self.assertEqual(today["reviews"][0]["id"], "custom-001")
        self.assertTrue(today["reviews"][0]["priority"])

    def test_rest_day_has_no_new(self):
        self.init()
        run("plan", "--date", D.isoformat())
        sunday = run_json("today", "--date", "2026-12-06")
        self.assertTrue(sunday["rest_day"])
        self.assertEqual(sunday["new"], [])
        self.assertIsNotNone(sunday["mock"])

    def test_mock_prefers_track(self):
        self.init()
        qs = run_json("mock", "--track", "training", "-n", "4")["questions"]
        self.assertEqual(len(qs), 4)
        self.assertTrue(qs[0]["id"].startswith("train-"))

    def test_design_in_plan_and_mock(self):
        self.init()
        plan = run_json("plan", "--date", D.isoformat())["plan"]
        ids = [p["chapter"] for p in plan]
        designs = [c for c in ids if c.startswith("design/")]
        # 只排和所选方向（training、harness）相关的设计题，并且排在最后
        self.assertIn("design/01-enterprise-kb-qa", designs)
        kb = KB(ROOT)
        for c in designs:
            self.assertTrue({"training", "harness"} & set(kb.chapters[c].tags), c)
        expected = {c for c in kb.design_order if {"training", "harness"} & set(kb.chapters[c].tags)}
        self.assertEqual(set(designs), expected)
        self.assertTrue(all(c.startswith("design/") for c in ids[ids.index(designs[0]):]))
        res = run_json("mock", "--design", "--track", "harness")
        self.assertTrue(res["design"]["chapter"].startswith("design/"))
        self.assertTrue(res["design"]["questions"])

    def test_design_suggested_on_rest_day(self):
        self.init()
        run("plan", "--date", D.isoformat())
        run("learned", "design/01-enterprise-kb-qa", "--date", D.isoformat())
        sunday = run_json("today", "--date", "2026-12-06")
        self.assertEqual(sunday["design"]["chapter"], "design/01-enterprise-kb-qa")

    def test_schema_version(self):
        self.init()
        run("plan", "--date", D.isoformat())
        f = Path(self.tmp.name) / "state.json"
        st = json.loads(f.read_text())
        self.assertEqual(st["schema"], coach.SCHEMA)
        st["schema"] = coach.SCHEMA + 1
        f.write_text(json.dumps(st))
        with self.assertRaises(SystemExit):
            run("today", "--date", D.isoformat())

    def test_bad_inputs(self):
        with self.assertRaises(SystemExit):
            run("init", "--tracks", "nope", "--daily-minutes", "100")
        self.init()
        with self.assertRaises(SystemExit):
            run("record", "core-agent-001", "3")
        with self.assertRaises(SystemExit):
            run("record", "no-such-001", "1")


if __name__ == "__main__":
    unittest.main()
