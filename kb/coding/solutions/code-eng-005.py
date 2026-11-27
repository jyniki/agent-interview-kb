"""code-eng-005 流式工具调用增量的拼接与解析。

输入 OpenAI 兼容的 chunk 流（choices[0].delta.tool_calls），按 index 累积每个调用：
id 出现时记下，name、arguments 按片段拼接；流在 finish_reason == "tool_calls" 结束后
再整体解析 JSON 并校验，解析失败不崩溃，把错误挂在该调用上交给上层。

复杂度：
- add_chunk O(片段长度)；
- finish O(总参数长度 + 调用数·log 排序)；
- 空间 O(所有片段之和)；每个 index 一份缓冲，缓冲只在 finish 时一次性解析。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


class IncompleteStreamError(RuntimeError):
    """流在 finish_reason == "tool_calls" 之前中断，缓冲里的调用不能执行。"""


class AssemblerStateError(RuntimeError):
    """finish 之后又喂了 chunk。"""


@dataclass
class AssembledCall:
    index: int
    id: str = ""
    name: str = ""
    arguments: Any = None          # 解析成功的 dict；失败时为 None
    raw_arguments: str = ""
    error: str | None = None       # 解析 / 校验错误，非空表示这个调用不能执行
    complete: bool = True          # 流是否带着 finish_reason 正常结束

    @property
    def is_valid(self) -> bool:
        return self.complete and self.error is None and bool(self.name)

    def to_message_call(self) -> dict:
        """转回 OpenAI 消息格式，方便作为工具结果前的 assistant.tool_calls 回填。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments if self.arguments is not None else self.raw_arguments,
                                        ensure_ascii=False, sort_keys=True),
            },
        }


@dataclass
class _Buffer:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""
    raw_parts: list = field(default_factory=list)


def _append_fragment(current: str, fragment: Any) -> str:
    """拼接字符串片段：空片段跳过；与当前完全相同视为“重复整段重发”，跳过。

    这样既能处理 id/name 被切碎（逐片串联），也能处理某些服务端每个分片都重复
    发送完整 id/name 的情况，不会拼出 "call_acall_a"。
    """
    if fragment is None:
        return current
    if not isinstance(fragment, str):
        fragment = str(fragment)
    if fragment == "" or fragment == current:
        return current
    return current + fragment


class ToolCallAssembler:
    """按 index 聚合流式工具调用分片。"""

    def __init__(self) -> None:
        self._buffers: dict[int, _Buffer] = {}
        self._content_parts: list[str] = []
        self.finish_reason: str | None = None
        self._finished = False
        self.chunks_seen = 0

    # ------------------------------------------------------------ 输入

    def add_chunk(self, chunk: dict) -> None:
        if self._finished:
            raise AssemblerStateError("finish 之后不能再喂 chunk，请新建 assembler")
        self.chunks_seen += 1
        if not isinstance(chunk, dict):
            raise TypeError("chunk 必须是 dict")
        choices = chunk.get("choices") or []
        if not choices:
            return  # 心跳、usage-only 之类的空 chunk
        choice = choices[0]  # 只处理第 0 个 choice
        if choice.get("finish_reason") is not None:
            self.finish_reason = choice["finish_reason"]
        delta = choice.get("delta") or {}

        content = delta.get("content")
        if content:
            self._content_parts.append(content)

        for fragment in delta.get("tool_calls") or []:
            self._add_fragment(fragment)

    def _add_fragment(self, fragment: dict) -> None:
        index = fragment.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"tool_calls[].index 必须是整数，实际 {index!r}")
        buf = self._buffers.get(index)
        if buf is None:
            buf = self._buffers[index] = _Buffer(index=index)
        buf.id = _append_fragment(buf.id, fragment.get("id"))
        function = fragment.get("function") or {}
        buf.name = _append_fragment(buf.name, function.get("name"))
        args = function.get("arguments")
        if args is None:
            return
        if isinstance(args, str):
            buf.raw_parts.append(args)
            buf.arguments += args
        elif isinstance(args, dict):
            # 少数服务端直接给对象；转成 JSON 文本统一处理
            text = json.dumps(args, ensure_ascii=False, sort_keys=True)
            buf.raw_parts.append(text)
            buf.arguments += text
        else:
            raise TypeError(f"arguments 应为字符串或对象，实际 {type(args).__name__}")

    # ------------------------------------------------------------ 输出

    @property
    def content(self) -> str:
        return "".join(self._content_parts)

    @property
    def raw_chunk_count(self) -> int:
        return sum(len(b.raw_parts) for b in self._buffers.values())

    def pending_indices(self) -> list[int]:
        return sorted(self._buffers)

    def finish(self, allow_incomplete: bool = False, finish_reason: str | None = None) -> list[AssembledCall]:
        """结束流并返回按 index 排序的调用列表。

        finish_reason 不是 "tool_calls" 却有缓冲，说明流提前断了：默认抛
        IncompleteStreamError；allow_incomplete=True 时照常返回，但调用标记为
        complete=False，调用方不得执行。
        """
        if self._finished:
            raise AssemblerStateError("finish 只能调用一次")
        self._finished = True
        reason = finish_reason if finish_reason is not None else self.finish_reason
        if self._buffers and reason != "tool_calls" and not allow_incomplete:
            raise IncompleteStreamError(
                f"流在 finish_reason={reason!r} 时结束，但还有 {len(self._buffers)} 个未完成的工具调用"
            )
        complete = reason == "tool_calls"
        calls = [self._materialize(idx, complete) for idx in sorted(self._buffers)]
        if not self._buffers and reason == "tool_calls":
            return []  # 说要有调用但一个分片都没来：返回空列表，由上层按异常处理
        return calls

    def _materialize(self, index: int, complete: bool) -> AssembledCall:
        buf = self._buffers[index]
        call = AssembledCall(index=index, id=buf.id, name=buf.name, raw_arguments=buf.arguments, complete=complete)
        text = buf.arguments.strip()
        if not buf.name:
            call.error = "缺少工具名"
            return call
        if text == "":
            # 参数为空串：当作 {} 处理，必填字段留给 schema 校验
            call.arguments = {}
            return call
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            call.error = f"arguments 不是合法 JSON：{exc.msg}（位置 {exc.pos}）"
            return call
        if not isinstance(parsed, dict):
            call.error = f"arguments 解析结果应为对象，实际是 {type(parsed).__name__}"
            return call
        call.arguments = parsed
        return call


def assemble_tool_calls(chunks, allow_incomplete: bool = False) -> list[AssembledCall]:
    """把一整条 chunk 流拼成调用列表。"""
    assembler = ToolCallAssembler()
    for chunk in chunks:
        assembler.add_chunk(chunk)
    return assembler.finish(allow_incomplete=allow_incomplete)


def chunk_of(deltas, finish_reason=None) -> dict:
    """测试辅助：构造一个只有一个 choice 的 chunk。"""
    return {"choices": [{"index": 0, "delta": deltas, "finish_reason": finish_reason}]}


def tool_fragment(index, *, id=None, name=None, arguments=None) -> dict:
    fragment = {"index": index, "type": "function", "function": {}}
    if id is not None:
        fragment["id"] = id
    if name is not None:
        fragment["function"]["name"] = name
    if arguments is not None:
        fragment["function"]["arguments"] = arguments
    return fragment


# ---------------------------------------------------------------- 自测
# 只在直接运行时执行；被 import 时不跑断言、不产生任何副作用。
# 直接 `python3 <本文件>` 即可运行；CI 的 run_snippets 也是整份执行。
if __name__ == "__main__":
    # ---------------------------------------------------------------- 断言

    import random

    # 1) 基本形态：id/name 只在首片，arguments 分片到达，按 index 拼起来。
    chunks = [
        chunk_of({"role": "assistant", "content": ""}),
        chunk_of({"tool_calls": [tool_fragment(0, id="call_1", name="get_weather", arguments="")]}),
        chunk_of({"tool_calls": [tool_fragment(0, arguments='{"city":')]}),
        chunk_of({"tool_calls": [tool_fragment(0, arguments='"北京"}')]}),
        chunk_of({}, finish_reason="tool_calls"),
    ]
    calls = assemble_tool_calls(chunks)
    assert len(calls) == 1
    assert calls[0].id == "call_1" and calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "北京"} and calls[0].is_valid
    assert calls[0].to_message_call()["function"]["name"] == "get_weather"

    # 2) 一次回复多个并行调用：交错下发也能按 index 归位，输出按 index 排序。
    chunks = [
        chunk_of({"tool_calls": [tool_fragment(1, id="c_b", name="search")]}),
        chunk_of({"tool_calls": [tool_fragment(0, id="c_a", name="weather")]}),
        chunk_of({"tool_calls": [tool_fragment(1, arguments='{"q": "py'), tool_fragment(0, arguments='{"city": "上海"}')]}),
        chunk_of({"tool_calls": [tool_fragment(1, arguments='thon"}')]}),
        chunk_of({}, finish_reason="tool_calls"),
    ]
    calls = assemble_tool_calls(chunks)
    assert [c.index for c in calls] == [0, 1]
    assert calls[0].arguments == {"city": "上海"} and calls[1].arguments == {"q": "python"}
    assert [c.id for c in calls] == ["c_a", "c_b"]

    # 3) 随机切片 + 多调用交错：拼接结果与原始调用逐字一致（顺序无关）。
    originals = [
        {"id": "call_x", "name": "search_flights", "args": {"from": "北京", "to": "上海", "date": "2025-06-01", "stops": 0, "tags": ["直飞", "早班"]}},
        {"id": "call_y", "name": "book_hotel", "args": {"city": "上海", "nights": 2, "prefs": {"breakfast": True}}},
        {"id": "call_z", "name": "noop", "args": {}},
    ]


    def split_randomly(text: str, rng: random.Random) -> list[str]:
        if text == "":
            return []
        cuts = sorted(rng.sample(range(1, len(text)), k=min(len(text) - 1, rng.randint(0, 4))))
        pieces, prev = [], 0
        for cut in cuts:
            pieces.append(text[prev:cut])
            prev = cut
        pieces.append(text[prev:])
        return pieces


    def build_stream(originals, rng, split_id_name=False):
        """把原始调用切碎并交错。返回 (chunks, 每个调用的期望值)。"""
        per_call: list[list[dict]] = []
        for order, original in enumerate(originals):
            args_text = json.dumps(original["args"], ensure_ascii=False) if original["args"] else ""
            fragments = [tool_fragment(order, id=original["id"], name=original["name"])]
            if split_id_name:  # id 和 name 也被切碎
                fragments[0] = tool_fragment(order)
                for piece in split_randomly(original["id"], rng):
                    fragments.append(tool_fragment(order, id=piece))
                for piece in split_randomly(original["name"], rng):
                    fragments.append(tool_fragment(order, name=piece))
            for piece in split_randomly(args_text, rng):
                fragments.append(tool_fragment(order, arguments=piece))
            per_call.append(fragments)

        chunks = []
        cursor = [0] * len(per_call)
        while any(cursor[i] < len(per_call[i]) for i in range(len(per_call))):
            batch = []
            for i in range(len(per_call)):
                if cursor[i] < len(per_call[i]):
                    batch.append(per_call[i][cursor[i]])
                    cursor[i] += 1
            # 同一 chunk 里多个调用的分片顺序也随机化，验证按 index 而不是按位置聚合
            rng.shuffle(batch)
            chunks.append(chunk_of({"tool_calls": batch}))
        chunks.append(chunk_of({}, finish_reason="tool_calls"))
        return chunks


    for seed in range(20):
        rng = random.Random(seed)
        stream = build_stream(originals, rng, split_id_name=bool(seed % 2))
        calls = assemble_tool_calls(stream)
        assert len(calls) == len(originals)
        for call, original in zip(calls, originals):
            assert call.id == original["id"], (seed, call.id)
            assert call.name == original["name"], (seed, call.name)
            assert call.arguments == original["args"], (seed, call.arguments)
            assert call.is_valid

    # 4) 空的 arguments 片段（整个流只给 name）：当作 {}，交给 schema 校验。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [tool_fragment(0, id="c", name="get_time")]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert calls[0].arguments == {} and calls[0].is_valid

    # 5) arguments 是非对象 JSON：不抛异常，标记错误。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [tool_fragment(0, id="c", name="f", arguments="[1, 2]")]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert calls[0].arguments is None and "对象" in calls[0].error and not calls[0].is_valid

    # 6) arguments 是截断的 JSON（流正常结束但参数不完整）：错误挂在调用上，不崩。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [tool_fragment(0, id="c", name="f", arguments='{"a": 1, "b"')]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert calls[0].error and calls[0].error.startswith("arguments 不是合法 JSON")
    assert calls[0].raw_arguments == '{"a": 1, "b"'

    # 7) 流在 finish_reason 之前中断：默认抛 IncompleteStreamError，不返回可执行调用。
    truncated = [
        chunk_of({"tool_calls": [tool_fragment(0, id="c", name="f", arguments='{"a":')]}),
    ]
    try:
        assemble_tool_calls(truncated)
    except IncompleteStreamError:
        pass
    else:
        raise AssertionError("截断的流必须报不完整")

    # 8) finish_reason = "stop" 但缓冲里有调用：同样视为不完整。
    try:
        assemble_tool_calls(
            [
                chunk_of({"tool_calls": [tool_fragment(0, id="c", name="f", arguments="{}")]}),
                chunk_of({}, finish_reason="stop"),
            ]
        )
    except IncompleteStreamError:
        pass
    else:
        raise AssertionError("finish_reason=stop 时的调用不能执行")

    # 9) allow_incomplete=True：返回但 complete=False，is_valid 为假。
    calls = assemble_tool_calls(truncated, allow_incomplete=True)
    assert len(calls) == 1 and calls[0].complete is False and calls[0].is_valid is False

    # 10) finish_reason = "tool_calls" 但一个分片都没有：返回空列表。
    assert assemble_tool_calls([chunk_of({}, finish_reason="tool_calls")]) == []

    # 11) 同一个 chunk 里 content 和 tool_calls 同时出现：分开累积，互不污染。
    assembler = ToolCallAssembler()
    assembler.add_chunk(chunk_of({"content": "我先查"}))
    assembler.add_chunk(chunk_of({"content": "一下。", "tool_calls": [tool_fragment(0, id="c", name="weather", arguments="{}")]}))
    assembler.add_chunk(chunk_of({}, finish_reason="tool_calls"))
    calls = assembler.finish()
    assert assembler.content == "我先查一下。"
    assert len(calls) == 1 and calls[0].arguments == {}

    # 12) 某些服务端每个分片都重复完整 id / name：不应拼成 "call_1call_1"。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [tool_fragment(0, id="call_7", name="weather", arguments="")]}),
            chunk_of({"tool_calls": [tool_fragment(0, id="call_7", name="weather", arguments='{"city"')]}),
            chunk_of({"tool_calls": [tool_fragment(0, id="call_7", name="weather", arguments=': "北京"}')]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert calls[0].id == "call_7" and calls[0].name == "weather" and calls[0].arguments == {"city": "北京"}

    # 13) 缺少 index 的单调用服务端：默认为 0。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert calls[0].index == 0 and calls[0].is_valid

    # 14) arguments 以对象形式下发（非标准）：也能处理。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": {"x": 1}}}]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert calls[0].arguments == {"x": 1}

    # 15) 缺少工具名：标记错误，不抛异常。
    calls = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [tool_fragment(0, id="c", arguments="{}")]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )
    assert not calls[0].is_valid and calls[0].error == "缺少工具名"

    # 16) finish 之后不能再喂 chunk，也不能重复 finish。
    assembler = ToolCallAssembler()
    assembler.add_chunk(chunk_of({}, finish_reason="tool_calls"))
    assembler.finish()
    for action in (lambda: assembler.add_chunk(chunk_of({})), assembler.finish):
        try:
            action()
        except AssemblerStateError:
            pass
        else:
            raise AssertionError("状态机应拒绝 finish 之后的输入")

    # 17) 空 chunk / 心跳 chunk 被忽略，不影响计数与结果。
    assembler = ToolCallAssembler()
    assembler.add_chunk({"choices": []})
    assembler.add_chunk({})
    assembler.add_chunk(chunk_of({"tool_calls": [tool_fragment(0, id="c", name="f", arguments="{}")]}))
    assembler.add_chunk(chunk_of({}, finish_reason="tool_calls"))
    assert len(assembler.finish()) == 1 and assembler.chunks_seen == 4

    # 18) index 不是整数直接报错（错位比悄悄拼错更安全）。
    try:
        ToolCallAssembler().add_chunk(chunk_of({"tool_calls": [tool_fragment("0", name="f")]}))
    except TypeError:
        pass
    else:
        raise AssertionError("非整数 index 应报错")

    # 19) to_message_call 能直接回填进 assistant 消息，再交给 tool 结果。
    call = assemble_tool_calls(
        [
            chunk_of({"tool_calls": [tool_fragment(0, id="c", name="f", arguments='{"a": 1}')]}),
            chunk_of({}, finish_reason="tool_calls"),
        ]
    )[0]
    msg = call.to_message_call()
    assert msg["id"] == "c" and json.loads(msg["function"]["arguments"]) == {"a": 1}

    # 20) 多条流互不影响（每次响应新建 assembler）。
    a = assemble_tool_calls([chunk_of({"tool_calls": [tool_fragment(0, id="a", name="f", arguments="{}")]}), chunk_of({}, finish_reason="tool_calls")])
    b = assemble_tool_calls([chunk_of({"tool_calls": [tool_fragment(0, id="b", name="g", arguments="{}")]}), chunk_of({}, finish_reason="tool_calls")])
    assert a[0].id == "a" and b[0].id == "b" and a[0].name == "f" and b[0].name == "g"
