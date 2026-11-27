"""code-eng-001 最小 Agent Loop。

循环调用模型：模型给出工具调用就校验参数、执行工具、把结果按调用 ID 追加回消息；
模型给出最终答案就返回。另设最大步数、预算、死循环检测，工具异常转成工具结果而不炸掉循环。

复杂度：
- run_agent：时间 O(S·(模型调用 + 工具执行))，S 为步数；空间 O(S + 消息长度)。
- 单次工具参数校验 O(参数个数)；每步死循环检测 O(调用数)。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class ToolArgError(ValueError):
    """工具参数不满足 schema。"""


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


@dataclass
class Tool:
    """工具注册表的一项：名字、可调用对象、参数 schema（JSON Schema 的子集）。"""

    name: str
    fn: Callable[..., Any]
    parameters: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class AgentResult:
    status: str  # "ok" | "max_steps" | "budget_exhausted" | "loop_detected"
    messages: list
    steps: int
    tool_calls: int
    answer: Optional[str]
    remaining_budget: Optional[float]


def validate_args(schema: dict, args: dict) -> None:
    """校验必填字段、未知字段和基础类型；不通过抛 ToolArgError。"""
    if not isinstance(args, dict):
        raise ToolArgError("参数必须是 JSON 对象")
    if not schema:
        return
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            raise ToolArgError(f"缺少必填参数：{key}")
    allow_extra = bool(schema.get("additionalProperties", False))
    for key, value in args.items():
        if key not in props:
            if not allow_extra:
                raise ToolArgError(f"未知参数：{key}")
            continue
        spec = props[key] or {}
        expected = spec.get("type")
        if expected is not None:
            checker = _TYPE_CHECKS.get(expected)
            if checker is None:
                raise ToolArgError(f"schema 里不认识的类型：{expected}")
            if not checker(value):
                raise ToolArgError(f"参数 {key} 应为 {expected}，实际是 {type(value).__name__}")
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            raise ToolArgError(f"参数 {key} 不在可选值 {enum} 内")


def _canonical(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _parse_arguments(raw: Any) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ToolArgError(f"arguments 应为字符串或对象，实际是 {type(raw).__name__}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolArgError(f"arguments 不是合法 JSON：{exc.msg}") from None
    if not isinstance(parsed, dict):
        raise ToolArgError("arguments 解析后必须是 JSON 对象")
    return parsed


def _error_result(kind: str, message: str) -> dict:
    return {"ok": False, "error": message, "error_type": kind}


def run_agent(
    model: Callable[[list], dict],
    tools: list,
    messages: list,
    *,
    max_steps: int = 8,
    budget: Optional[float] = None,
    parallel: bool = True,
    max_consecutive_repeats: int = 3,
) -> AgentResult:
    """跑一个最小 Agent Loop。

    model(messages) -> {"content": str|None, "tool_calls": [...], "usage": {"cost": 数值}}
    工具调用格式与 OpenAI 兼容：{"id", "type": "function", "function": {"name", "arguments"}}。
    """
    if max_steps < 1:
        raise ValueError("max_steps 至少为 1")
    if max_consecutive_repeats < 1:
        raise ValueError("max_consecutive_repeats 至少为 1")

    registry: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in registry:
            raise ValueError(f"工具名重复：{tool.name}")
        registry[tool.name] = tool

    transcript = [dict(m) for m in messages]  # 不改调用方的列表
    remaining = budget
    steps = 0
    executed = 0
    answer: Optional[str] = None
    status: Optional[str] = None
    last_sig: Optional[str] = None
    repeats = 0
    seen_ids: set[str] = set()

    while steps < max_steps:
        if remaining is not None and remaining <= 0:
            status = "budget_exhausted"
            break

        response = model(transcript)
        steps += 1
        usage = response.get("usage") or {}
        cost = float(usage.get("cost", 1.0)) if isinstance(usage, dict) else 1.0
        if remaining is not None:
            remaining -= cost

        calls = response.get("tool_calls") or []
        assistant: dict[str, Any] = {"role": "assistant", "content": response.get("content")}
        if calls:
            assistant["tool_calls"] = calls
        transcript.append(assistant)

        if not calls:
            status = "ok"
            answer = response.get("content")
            break

        # 死循环检测：同一批（工具名, 规范化参数）连续出现太多次就终止，重复的这次不执行。
        sig = json.dumps(
            sorted(_canonical_call(c) for c in calls), ensure_ascii=False
        )
        repeats = repeats + 1 if sig == last_sig else 1
        last_sig = sig
        if repeats > max_consecutive_repeats:
            status = "loop_detected"
            break

        # 先做本地判定（未知工具、重复 ID、参数错误），再执行。
        outcomes: list[dict] = []
        for call in calls:
            call_id = call.get("id")
            if not call_id:
                outcomes.append(_error_result("bad_call", "工具调用缺少 id"))
                continue
            if call_id in seen_ids:
                outcomes.append(_error_result("duplicate_id", f"调用 ID 重复：{call_id}"))
                continue
            seen_ids.add(call_id)
            outcomes.append(_prepare_call(call, registry))

        todo = [i for i, o in enumerate(outcomes) if o.get("_run")]
        if todo and parallel and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=len(todo)) as pool:
                futures = {i: pool.submit(_invoke, outcomes[i]["_tool"], outcomes[i]["_args"]) for i in todo}
                for i, fut in futures.items():
                    outcomes[i] = fut.result()
        else:
            for i in todo:
                outcomes[i] = _invoke(outcomes[i]["_tool"], outcomes[i]["_args"])

        for call, outcome in zip(calls, outcomes):
            executed += 1
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "name": (call.get("function") or {}).get("name", ""),
                    "content": json.dumps(outcome, ensure_ascii=False, default=str),
                }
            )

    if status is None:
        status = "max_steps"

    return AgentResult(
        status=status,
        messages=transcript,
        steps=steps,
        tool_calls=executed,
        answer=answer,
        remaining_budget=remaining,
    )


def _canonical_call(call: dict) -> str:
    function = call.get("function") or {}
    return f"{function.get('name', '')}|{_canonical(function.get('arguments', ''))}"


def _prepare_call(call: dict, registry: dict) -> dict:
    function = call.get("function") or {}
    name = function.get("name")
    tool = registry.get(name)
    if tool is None:
        return _error_result("unknown_tool", f"未注册的工具：{name!r}")
    try:
        args = _parse_arguments(function.get("arguments"))
        validate_args(tool.parameters, args)
    except ToolArgError as exc:
        return _error_result("bad_arguments", str(exc))
    return {"_run": True, "_tool": tool, "_args": args}


def _invoke(tool: Tool, args: dict) -> dict:
    try:
        return {"ok": True, "result": tool.fn(**args)}
    except Exception as exc:  # noqa: BLE001 —— 工具异常必须变成工具结果交给模型
        return _error_result("tool_exception", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- 自测
# 只在直接运行时执行；被 import 时不跑断言、不产生任何副作用。
# 直接 `python3 <本文件>` 即可运行；CI 的 run_snippets 也是整份执行。
if __name__ == "__main__":
    # ---------------------------------------------------------------- 断言

    def _scripted(responses):
        """按顺序返回预设响应的假模型，同时记录它看到的消息快照。"""

        class _Model:
            def __init__(self):
                self.calls = 0
                self.snapshots = []

            def __call__(self, messages):
                self.snapshots.append([dict(m) for m in messages])
                index = min(self.calls, len(responses) - 1)
                self.calls += 1
                return dict(responses[index])

        return _Model()


    def _call(call_id, name, arguments):
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


    # 工具：会记录收到的参数，可选择性抛异常。
    calls_seen: list[dict] = []


    def get_weather(city, unit="celsius"):
        calls_seen.append({"city": city, "unit": unit})
        return {"city": city, "temp": 21, "unit": unit}


    tools = [
        Tool(
            "get_weather",
            get_weather,
            {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                "required": ["city"],
            },
        ),
        Tool("add", lambda a, b: a + b, {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}),
    ]

    # 1) 直接给最终答案：一次模型调用，状态 ok，消息只追加 assistant。
    model = _scripted([{"content": "你好", "tool_calls": []}])
    user_msgs = [{"role": "user", "content": "hi"}]
    result = run_agent(model, tools, user_msgs, max_steps=4)
    assert result.status == "ok" and result.answer == "你好"
    assert result.steps == 1 and result.tool_calls == 0
    assert result.messages[-1]["role"] == "assistant" and result.messages[-1]["content"] == "你好"
    assert user_msgs == [{"role": "user", "content": "hi"}], "不能改动调用方传入的消息列表"

    # 2) 一次工具调用后给最终答案：工具结果要带 tool_call_id 和 name，模型下一步能看到它。
    calls_seen.clear()
    model = _scripted(
        [
            {"content": None, "tool_calls": [_call("c1", "get_weather", '{"city": "北京"}')]},
            {"content": "北京 21 度", "tool_calls": []},
        ]
    )
    result = run_agent(model, tools, [{"role": "user", "content": "北京天气"}], max_steps=4)
    assert result.status == "ok" and result.steps == 2 and result.tool_calls == 1
    assert calls_seen == [{"city": "北京", "unit": "celsius"}]
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "c1" and tool_msgs[0]["name"] == "get_weather"
    assert json.loads(tool_msgs[0]["content"]) == {"ok": True, "result": {"city": "北京", "temp": 21, "unit": "celsius"}}
    # 模型第二次调用时能看到 assistant 的工具调用和工具结果
    assert model.snapshots[1][-1]["role"] == "tool"

    # 3) 达到最大步数：状态 max_steps，模型被调用的次数正好等于 max_steps。
    model = _scripted([{"content": None, "tool_calls": [_call("cX", "add", '{"a": 1, "b": 2}')]}])
    result = run_agent(model, tools, [], max_steps=3, max_consecutive_repeats=99)
    assert result.status == "max_steps" and result.steps == 3 and model.calls == 3
    assert result.tool_calls == 3

    # 4) 预算耗尽：每次模型调用扣 1，预算 2 只能撑两次调用。
    model = _scripted([{"content": None, "tool_calls": [_call("cY", "add", '{"a": 1, "b": 2}')]}])
    result = run_agent(model, tools, [], max_steps=99, budget=2, max_consecutive_repeats=99)
    assert result.status == "budget_exhausted" and result.steps == 2 and result.remaining_budget == 0
    assert model.calls == 2

    # 5) 工具抛异常：循环不崩，错误作为工具结果回到模型，模型可以继续并给出答案。
    def explode(**kwargs):
        raise RuntimeError("boom")


    tools_with_fail = tools + [Tool("explode", explode, {})]
    model = _scripted(
        [
            {"content": None, "tool_calls": [_call("e1", "explode", "{}")]},
            {"content": "工具挂了，我换个办法", "tool_calls": []},
        ]
    )
    result = run_agent(model, tools_with_fail, [], max_steps=4)
    assert result.status == "ok" and result.answer.startswith("工具挂了")
    err = json.loads([m for m in result.messages if m["role"] == "tool"][0]["content"])
    assert err["ok"] is False and "boom" in err["error"] and err["error_type"] == "tool_exception"
    assert "boom" in model.snapshots[1][-1]["content"], "模型必须看到异常信息"

    # 6) 参数错误：缺必填、未知参数、类型错、JSON 非法、arguments 为空串。
    calls_seen.clear()
    model = _scripted([{"content": "ok", "tool_calls": []}])
    bad = [
        _call("b1", "get_weather", "{}"),                          # 缺 city
        _call("b2", "get_weather", '{"city": "上海", "x": 1}'),     # 未知参数
        _call("b3", "get_weather", '{"city": 3}'),                  # 类型错
        _call("b4", "get_weather", "{not json"),                    # 非法 JSON
        _call("b5", "get_weather", '{"city": "上海", "unit": "kelvin"}'),  # enum 越界
        _call("b6", "get_weather", ""),                             # 空串 ≠ {}
    ]
    model = _scripted([{"content": None, "tool_calls": bad}, {"content": "ok", "tool_calls": []}])
    result = run_agent(model, tools, [], max_steps=4)
    assert result.status == "ok" and calls_seen == [], "参数不合法时工具不能被执行"
    errs = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
    assert len(errs) == 6
    assert all(e["ok"] is False and e["error_type"] == "bad_arguments" for e in errs)

    # 7) 未注册工具：错误结果，不抛异常。
    model = _scripted([{"content": None, "tool_calls": [_call("u1", "rm_rf", "{}")]}, {"content": "no", "tool_calls": []}])
    result = run_agent(model, tools, [], max_steps=3)
    err = json.loads([m for m in result.messages if m["role"] == "tool"][0]["content"])
    assert err["error_type"] == "unknown_tool"

    # 8) 同一批里调用 ID 重复：第二条判错，第一条照常执行。
    calls_seen.clear()
    model = _scripted(
        [
            {"content": None, "tool_calls": [_call("dup", "add", '{"a": 1, "b": 1}'), _call("dup", "add", '{"a": 2, "b": 2}')]},
            {"content": "done", "tool_calls": []},
        ]
    )
    result = run_agent(model, tools, [], max_steps=3)
    outcomes = [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]
    assert outcomes[0] == {"ok": True, "result": 2}
    assert outcomes[1]["error_type"] == "duplicate_id"

    # 9) 并行工具调用：三个调用真的同时执行（用 barrier 证明），结果按调用顺序、按 ID 对回去。
    import threading

    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    finish_order: list[str] = []


    def rendezvous(tag):
        barrier.wait()  # 串行执行会在这里超时并抛 BrokenBarrierError
        with lock:
            finish_order.append(tag)
        return tag


    parallel_tools = [Tool("rendezvous", rendezvous, {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]})]
    model = _scripted(
        [
            {
                "content": None,
                "tool_calls": [
                    _call("p1", "rendezvous", '{"tag": "one"}'),
                    _call("p2", "rendezvous", '{"tag": "two"}'),
                    _call("p3", "rendezvous", '{"tag": "three"}'),
                ],
            },
            {"content": "all done", "tool_calls": []},
        ]
    )
    result = run_agent(model, parallel_tools, [], max_steps=3, parallel=True)
    assert result.status == "ok" and len(finish_order) == 3
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["p1", "p2", "p3"], "结果顺序要跟调用顺序一致"
    assert [json.loads(m["content"])["result"] for m in tool_msgs] == ["one", "two", "three"]

    # 10) 关掉并行时，同样的三个调用会串行执行 —— 用 barrier 验证会超时（这里改用计数验证顺序）。
    serial_order: list[str] = []
    serial_tools = [Tool("step", lambda tag: (serial_order.append(tag), tag)[1], {"type": "object", "properties": {"tag": {"type": "string"}}, "required": ["tag"]})]
    model = _scripted(
        [
            {"content": None, "tool_calls": [_call("s1", "step", '{"tag": "a"}'), _call("s2", "step", '{"tag": "b"}')]},
            {"content": "done", "tool_calls": []},
        ]
    )
    run_agent(model, serial_tools, [], max_steps=3, parallel=False)
    assert serial_order == ["a", "b"]

    # 11) 死循环检测：模型一直重放同一个调用，达到上限后终止，重复的那次不再执行。
    calls_seen.clear()
    model = _scripted([{"content": None, "tool_calls": [_call("L", "add", '{"a": 1, "b": 2}')]}])
    result = run_agent(model, tools, [], max_steps=99, max_consecutive_repeats=3)
    assert result.status == "loop_detected"
    assert result.steps == 4 and result.tool_calls == 3, "执行 3 次，第 4 次识别为重复后终止"

    # 12) 参数等价的调用也算重复（键顺序不同、空白不同）。
    model = _scripted([{"content": None, "tool_calls": [_call("L1", "add", '{"a": 1, "b": 2}')]}])
    r1 = run_agent(model, tools, [], max_steps=99, max_consecutive_repeats=2)
    model = _scripted([{"content": None, "tool_calls": [_call("L2", "add", '{ "b": 2, "a": 1 }')]}])
    r2 = run_agent(model, tools, [], max_steps=99, max_consecutive_repeats=2)
    assert r1.status == r2.status == "loop_detected" and r1.steps == r2.steps == 3

    # 13) 重复检测只看“连续”：中间换了一次调用就重新计数。
    model = _scripted(
        [
            {"content": None, "tool_calls": [_call("a1", "add", '{"a": 1, "b": 1}')]},
            {"content": None, "tool_calls": [_call("a2", "add", '{"a": 9, "b": 9}')]},
            {"content": None, "tool_calls": [_call("a3", "add", '{"a": 1, "b": 1}')]},
            {"content": "fin", "tool_calls": []},
        ]
    )
    result = run_agent(model, tools, [], max_steps=9, max_consecutive_repeats=2)
    assert result.status == "ok" and result.steps == 4

    # 14) content 和 tool_calls 同时出现：content 要保留在 assistant 消息里。
    model = _scripted(
        [
            {"content": "我先查一下天气", "tool_calls": [_call("mix", "get_weather", '{"city": "广州"}')]},
            {"content": "查完了", "tool_calls": []},
        ]
    )
    result = run_agent(model, tools, [], max_steps=3)
    assistant_msgs = [m for m in result.messages if m["role"] == "assistant"]
    assert assistant_msgs[0]["content"] == "我先查一下天气" and assistant_msgs[0]["tool_calls"][0]["id"] == "mix"

    # 15) 非法参数与 max_steps 校验。
    for bad_kwargs in ({"max_steps": 0}, {"max_consecutive_repeats": 0}):
        try:
            run_agent(model, tools, [], **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_kwargs} 应该抛 ValueError")

    # 16) 工具名重复在启动时报错。
    try:
        run_agent(model, [Tool("dup", lambda: 1, {}), Tool("dup", lambda: 2, {})], [])
    except ValueError:
        pass
    else:
        raise AssertionError("重复工具名应抛 ValueError")

    # 17) 工具返回不可 JSON 序列化的对象时不崩，序列化成字符串。
    import datetime

    weird = [Tool("now", lambda: datetime.datetime(2024, 1, 1), {})]
    model = _scripted([{"content": None, "tool_calls": [_call("w1", "now", "{}")]}, {"content": "ok", "tool_calls": []}])
    result = run_agent(model, weird, [], max_steps=3)
    assert "2024-01-01" in [m for m in result.messages if m["role"] == "tool"][0]["content"]
