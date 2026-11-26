# Trace、回放与成本统计

> 层级：垂类 · harness · 预计 150 分钟 · 题目：[questions.md](questions.md)
> 前置：[harness/01 Agent Loop 与 Harness 的边界](../01-loop-and-harness/README.md)（Task、Turn、Step 三个粒度）、[harness/02 上下文装配、压缩与断点续跑](../02-context-compaction/README.md)（Event Log 与 Checkpoint）、[harness/05 模型网关、流式、缓存与熔断降级](../05-gateway-reliability/README.md)（四项计量）

## 1 这一章解决什么问题

[01](../01-loop-and-harness/README.md) 说日志要按 Step 记，[02](../02-context-compaction/README.md) 说 Event Log 是事实来源。面试官接着会问四件事：

1. 一条记录到底有哪些字段？用 OpenTelemetry 怎么打点？
2. 线上出了一个 bad case，能不能原样复现？
3. 这个月的模型账单，能不能拆到每个租户、每个任务、每一步？
4. Trace 存多少、存多久、里面的用户数据怎么办？

之前常见的做法是打文本日志，例如“第 3 步工具报错”。它有三个缺陷。第一，看不出父子关系：并行的子 Agent 和重试混在一起，排不出因果顺序。第二，没有记下外部输入：模型输出、工具结果、时间都没存，只能重跑，而重跑的结果和线上不一样。第三，token 数散落在各层：Agent 框架、网关、SDK 各报一次，加起来就重复了。

这一章从工程侧回答这四个问题。怎么用轨迹给 Agent 打分，见 [evaluation/03 轨迹与长程任务评测](../../evaluation/03-trajectory-eval/README.md)。线上评测和发布门禁见 [evaluation/05 线上评测、A/B 与回归门禁](../../evaluation/05-online-regression/README.md)。学完你应该能：

- 说出 Span 的数据模型（ID、父子关系、起止时间、属性、状态），写一个最小 tracer，并从 Span 树算出每类操作的耗时；
- 写出一条 Step 记录的字段，说出哪些字段缺了就没法复盘；
- 用 OpenTelemetry 的 GenAI 语义约定给一次 Agent 运行画出 Span 树，说出关键属性名和哪些是 Opt-In；
- 算出头部采样和尾部采样的存储量，以及漏掉错误的概率；
- 写一个录制回放层，说出确定性回放要记什么，回放时怎样检出分叉；
- 按四项口径统计 token，按带生效日期的价格表算钱，避开重复计数；
- 设计 Trace 的脱敏、保留期和删除。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| r | 头部采样率，例如 0.1 表示留 10% 的 Trace |
| p_e | 错误 Trace 占全部 Trace 的比例 |
| p_s | 慢 Trace（超过时延阈值）占的比例 |
| k | 一次事故里出错的 Trace 条数 |
| U | 未缓存输入 token 数（uncached input） |
| C_r | 缓存读 token 数（cache read） |
| C_w | 缓存写 token 数（cache write） |
| O | 输出 token 数，含推理 token |
| I | 总输入 token 数，I = U + C_r + C_w |
| P_in、P_out | 每百万 token 的输入、输出单价 |
| m_w、m_r | 缓存写、缓存读相对输入单价的倍率 |
| d(s) | Span s 的时长，结束时间减开始时间 |

## 2 Trace 的数据模型

### 2.1 Span 有哪些字段

OpenTelemetry（OTel）的 Traces 概念文档把 Trace 定义为一个请求走过的路径，它由一组 Span 组成。Span 是一个工作单元。一个 Span 包含这些信息：

| 字段 | 含义 | 在 Agent 里的例子 |
|---|---|---|
| Name | Span 的名字 | `chat model-x`、`execute_tool search` |
| Span Context | trace ID（16 字节）、span ID（8 字节）、trace flags、trace state；这部分可序列化、可跨进程传播 | 调用另一个服务里的工具时放进请求头 |
| Parent span ID | 父 Span 的 ID；为空的是根 Span | 工具 Span 的父 Span 是 `invoke_agent` |
| Start / End | 起止时间戳 | 算模型调用耗时、工具耗时 |
| Attributes | 键值对 | `gen_ai.request.model`、`gen_ai.usage.input_tokens` |
| Events | Span 内某个时刻发生的事，带时间戳 | 流式输出的首个 token、一次重试 |
| Links | 指向其他 Span（可以在别的 Trace 里），表示因果关系 | 审批后恢复的运行指回暂停前的 Span |
| Status | `Unset`（默认）、`Error`、`Ok` | 工具超时设 `Error` |
| Span Kind | `CLIENT`、`SERVER`、`INTERNAL`、`PRODUCER`、`CONSUMER` | 调远程模型是 `CLIENT`，进程内的 Agent 逻辑是 `INTERNAL` |

一个 Trace 里所有 Span 共享同一个 trace ID。每个 Span 只记自己父亲的 ID，不记孩子。这样 Span 可以各自独立上报，乱序到达后端，后端再按 parent ID 拼成树。代价是：父 Span 先丢了，孩子就成了“孤儿”，树拼不完整（见第 10 节）。

### 2.2 从 Span 树算耗时

**问题**：一个任务跑了 8 秒，慢在模型、工具还是 Harness 自己？

把 Span 树看成区间的嵌套。记 d(s) 为 Span s 的时长，children(s) 为它的直接子 Span。定义 s 的**自身耗时**为：s 存在、但没有任何直接子 Span 在运行的时间。

```text
self(s) = d(s) − | ∪_{c ∈ children(s)} [start_c, end_c] |
```

`| ∪ |` 是子区间并集的长度。子 Span 串行、互不重叠时，并集长度就是各段之和：

```text
self(s) = d(s) − Σ_{c ∈ children(s)} d(c)          （子 Span 不重叠时）
```

把所有 Span 的自身耗时加起来，每一段时间恰好被算一次：它只属于覆盖它的最深那个 Span。所以：

```text
Σ_{s ∈ Trace} self(s) = d(root)
```

这是推导，第 2.4 节的代码会验证。它给出两个用法：

- **按类型汇总**：Σ d(s) over 所有 `chat` Span，是“花在模型上的总时间”。Σ d(s) over 所有 `execute_tool` Span，是工具时间。根 Span 的自身耗时就是 Harness 自己的开销（解析、路由、压缩）。
- **并行时别直接相减**：主 Agent 同时派出 3 个子 Agent，每个跑 10 秒，父 Span 只跑了 11 秒。Σ d(c) = 30 秒，比父 Span 还长，直接相减得负数。要用区间并集：并集是 10 秒，自身耗时是 1 秒。

想找“缩短哪一段能让任务变快”，要看关键路径：从根 Span 出发，每层选结束最晚的那个子 Span 往下走。并行的子 Agent 里，只有最慢的那个在关键路径上。

### 2.3 跨进程传播

工具在另一个服务里执行时，Trace 靠请求头接上。OTel 默认用 W3C TraceContext，请求头是 `traceparent`：

```text
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^ ^^
          version        trace-id（32 位十六进制）  parent-id（16 位）  trace-flags（01 = sampled）
```

W3C 规范写明，trace-id 全为 0 是无效值，接收方必须忽略这个 `traceparent`。parent-id 是调用方当前 Span 的 ID，下游以它为父开新 Span。trace-flags 的最低位是 sampled 标志，用来告诉下游“上游决定记录这条 Trace”，下游据此保持一致的采样决定（第 5 节）。

Baggage 可以顺路带任意键值，比如租户 ID。OTel 文档提醒不要放凭据、API Key 或个人信息，因为下游可能把它记进日志或转发给不可信的服务。

### 2.4 最小 tracer

下面用标准库写一个 tracer：用 `contextvars` 记当前 Span，进入 `with` 时以当前 Span 为父，退出时记下结束时间；异常时把状态设成 `ERROR` 并填 `error.type`。时钟是注入的假时钟，所以每次运行耗时完全一样。

```python
import contextvars, os, re

class FakeClock:                      # 注入的时钟：测试里手动推进，结果可复现
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, s): self.t += s

class Tracer:
    def __init__(self, clock):
        self.clock, self.spans = clock, []
        self.trace_id = os.urandom(16).hex()          # 16 字节，32 个十六进制字符
        self._cur = contextvars.ContextVar("cur", default=None)

    def span(self, name, kind="INTERNAL", **attrs):
        tracer = self
        class _S:
            def __enter__(s):
                parent = tracer._cur.get()
                s.rec = {"name": name, "kind": kind, "attrs": dict(attrs),
                         "span_id": os.urandom(8).hex(),       # 8 字节，16 个十六进制字符
                         "parent": parent["span_id"] if parent else None,
                         "start": tracer.clock.now(), "status": "UNSET"}
                s.tok = tracer._cur.set(s.rec)
                return s.rec
            def __exit__(s, et, ev, tb):
                s.rec["end"] = tracer.clock.now()
                if et is not None:                    # 异常：状态设 ERROR，填 error.type
                    s.rec["status"] = "ERROR"
                    s.rec["attrs"]["error.type"] = et.__name__
                tracer._cur.reset(s.tok)
                tracer.spans.append(s.rec)            # 孩子先结束，先于父亲上报
                return False                          # 不吞异常
        return _S()

    def traceparent(self):                            # W3C: version-traceid-parentid-flags
        return f"00-{self.trace_id}-{self._cur.get()['span_id']}-01"

def union_len(intervals):                             # 区间并集长度
    total, end = 0.0, float("-inf")
    for a, b in sorted(intervals):
        if b > end:
            total += b - max(a, end)
            end = b
    return total

def self_time(spans):
    return {s["span_id"]: (s["end"] - s["start"]) -
            union_len([(c["start"], c["end"]) for c in spans if c["parent"] == s["span_id"]])
            for s in spans}

clk = FakeClock(); tr = Tracer(clk)
with tr.span("invoke_agent triage", **{"gen_ai.operation.name": "invoke_agent",
                                        "gen_ai.agent.name": "triage"}):
    with tr.span("chat model-x", kind="CLIENT", **{"gen_ai.operation.name": "chat"}):
        clk.sleep(1.2)
    try:
        with tr.span("execute_tool search", **{"gen_ai.tool.name": "search"}):
            hdr = tr.traceparent()                   # 跨服务调用工具时放进请求头
            clk.sleep(5.0)
            raise TimeoutError("upstream")
    except TimeoutError:
        pass
    with tr.span("execute_tool search", **{"gen_ai.tool.name": "search"}):
        clk.sleep(0.8)                                # 重试成功
    with tr.span("chat model-x", kind="CLIENT", **{"gen_ai.operation.name": "chat"}):
        clk.sleep(0.9)
    clk.sleep(0.1)                                    # Harness 自己的解析和路由

spans = tr.spans
dur = lambda s: s["end"] - s["start"]
root = [s for s in spans if s["parent"] is None]
assert len(root) == 1 and root[0]["name"] == "invoke_agent triage"
assert spans[-1] is root[0]                           # 根 Span 最后结束、最后上报
kids = sorted([s for s in spans if s["parent"] == root[0]["span_id"]], key=lambda s: s["start"])
assert [c["name"].split()[0] for c in kids] == ["chat", "execute_tool", "execute_tool", "chat"]

st = self_time(spans)
assert abs(dur(root[0]) - 8.0) < 1e-9
assert abs(sum(st.values()) - dur(root[0])) < 1e-9    # 自身耗时加总 = 根 Span 时长
assert abs(st[root[0]["span_id"]] - 0.1) < 1e-9       # Harness 开销 0.1 秒

by_kind = {}
for s in spans:
    k = s["name"].split()[0]
    by_kind[k] = by_kind.get(k, 0) + dur(s)
assert abs(by_kind["chat"] - 2.1) < 1e-9 and abs(by_kind["execute_tool"] - 5.8) < 1e-9

failed = [s for s in spans if s["status"] == "ERROR"]
assert len(failed) == 1 and failed[0]["attrs"]["error.type"] == "TimeoutError"
assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", hdr)
assert hdr.split("-")[2] == failed[0]["span_id"]      # 下游 Span 的父亲是这个工具 Span

# 并行子 Agent：3 个各 10 秒、同时跑，父 Span 11 秒。直接相减得负数，并集才对
par = [{"span_id": "p", "parent": None, "start": 0.0, "end": 11.0}] + \
      [{"span_id": f"c{i}", "parent": "p", "start": 0.5, "end": 10.5} for i in range(3)]
assert 11.0 - sum(dur(c) for c in par[1:]) < 0        # 错误算法
assert abs(self_time(par)["p"] - 1.0) < 1e-9          # 并集：自身耗时 1 秒
print("ok", by_kind)
```

这段代码验证了三件事：Span 按父 ID 能拼回一棵树；自身耗时加总等于根 Span 时长；并行子 Span 要用区间并集。它也展示了真实 SDK 的一个行为：孩子先结束、先上报，根 Span 最后才到。尾部采样要等根 Span 到了才能做决定，原因就在这里（第 5.2 节）。

## 3 一条 Step 记录有哪些字段

**问题**：只记一行“第 3 步工具报错”，复盘时不知道模型当时看到了什么、参数从哪来。**做法**：每个 Step 记一条结构化事件，按 Task、Turn、Step 三级编号串起来。正文大的内容（完整上下文、工具结果）单独存，事件里只放引用。

```json
{
  "task_id": "t_8f2", "turn_id": 3, "step_id": 7, "parent_step_id": 6,
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736", "ts_start": "…05.120Z", "ts_end": "…06.480Z",
  "versions": {"prompt": "triage@v12", "model": "model-x-2026-09", "toolset": "ts@41", "harness": "1.8.3"},
  "model_call": {"request_model": "model-x", "response_model": "model-x-2026-09",
                 "finish_reason": "tool_use",
                 "usage": {"uncached": 2434, "cache_read": 15800, "cache_write": 0, "output": 312},
                 "context_ref": "blob://ctx/t_8f2/7.json"},
  "tool": {"name": "search_orders", "call_id": "call_31", "args_hash": "a91c…",
           "args_ref": "blob://args/t_8f2/7.json", "status": "error", "error_type": "timeout",
           "retry": 1, "idempotency_key": "t_8f2-7", "side_effect": false},
  "budget": {"steps_used": 7, "tokens_used": 96120, "usd_used": 0.41}
}
```

`usage` 用的是四项口径，总输入是前三项之和：2434 + 15800 + 0 = 18234。口径为什么要统一，见第 7 节。

| 字段组 | 回答什么问题 | 缺了会怎样 |
|---|---|---|
| 三级编号 + `parent_step_id` | 这一步属于哪个任务、哪一轮，由哪一步引出 | 并行的子 Agent 和重试混在一起，排不出顺序 |
| 时间戳 | 慢在哪一段 | 只能看到总耗时，不知道是模型慢还是工具慢 |
| 版本指纹 | 这次运行用的是哪版 prompt、模型、工具集、Harness | 分不清“新版本引入的问题”和“这类问题一直存在” |
| 上下文引用 | 模型当时看到了什么 | 无法判断错因是模型、上下文还是工具，归因方法见 [evaluation/03](../../evaluation/03-trajectory-eval/README.md) |
| 工具调用 | 参数、结果、错误类型、重试次数、幂等键、有无副作用 | 恢复和回放时不知道哪些动作已经生效 |
| 预算快照 | 到这一步累计花了多少 | 超预算时说不清是哪一步吃掉的 |

**两个细节**：`request_model` 和 `response_model` 分开记，请求里常是别名，响应里才是实际服务的版本。记“实际发给模型的上下文”，不是“消息历史”，压缩、裁剪、注入检索结果之后两者不一样。

**存储代价**：上下文全文每步都存，存储随步数近似平方增长。推导：第 i 步的上下文约含前 i 步的内容，设每步新增 c 个 token，第 i 步的上下文约 i·c，n 步总共存 Σ i·c ≈ c·n²/2。一个 50 步、每步新增 2K token 的任务，全文每步都存约 2K × 50² / 2 = 2.5M token；按内容哈希去重后只存增量，约 2K × 50 = 100K token，差 25 倍。常见做法：稳定前缀只存一份，每步只存增量和引用。

**Step 记录和 Span 的关系**：两者是同一件事的两种视图。Span 是给 Trace 后端看的，强调时序和父子关系，按 OTel 约定命名；Step 记录是给复盘、回放和计费看的，强调完整性。常见实现是 Harness 在一处埋点同时产出两者：Span 带 `task_id`、`step_id` 属性，Step 记录带 `trace_id`、`span_id`，两边能互相跳转。

## 4 OpenTelemetry GenAI 语义约定

**为什么用标准**：自己定义字段，换观测后端、接第三方框架时要重写。OTel 为生成式 AI 定义了一组语义约定，属性名以 `gen_ai.` 开头。

**规范状态**（2026-11-26 查阅）：这组约定已从 OTel 主仓库移到单独的 `open-telemetry/semantic-conventions-genai` 仓库，文档页和其中的属性状态都是 Development（开发中）。属性名可能还会改，落地前以当时版本为准。

### 4.1 三类 Span

| Span | 名字格式 | Kind | 关键属性 |
|---|---|---|---|
| 推理调用 `gen_ai.client.inference` | `{gen_ai.operation.name} {gen_ai.request.model}`，如 `chat model-x` | `CLIENT`；模型在同一进程里时可以是 `INTERNAL` | `gen_ai.operation.name`、`gen_ai.provider.name`、`gen_ai.request.model`、`gen_ai.response.model`、`gen_ai.response.finish_reasons`、`gen_ai.usage.input_tokens`、`gen_ai.usage.output_tokens`、`gen_ai.usage.cache_read.input_tokens`、`gen_ai.usage.cache_write.input_tokens`、`gen_ai.usage.reasoning.output_tokens` |
| 进程内 Agent 调用 | `{gen_ai.operation.name} {gen_ai.agent.name}`，如 `invoke_agent triage` | `INTERNAL` | `gen_ai.operation.name` 为 `invoke_agent`（模型驱动控制流）或 `invoke_workflow`（应用代码定义流程）、`gen_ai.agent.name`、`gen_ai.conversation.id` |
| 工具执行 `gen_ai.execute_tool` | `execute_tool {gen_ai.tool.name}` | `INTERNAL` | `gen_ai.tool.name`、`gen_ai.tool.call.id`、`gen_ai.tool.type`、`error.type` |

`gen_ai.operation.name` 有一组预定义值，推理类的有 `chat`、`generate_content`、`text_completion`、`realtime_inference`；此外还有 `embeddings`、`execute_tool`、`invoke_agent`、`invoke_workflow`、`plan`、`retrieval`、`create_agent` 和几种记忆操作。约定要求：预定义值适用时必须用它。

推理 Span 的 Kind 一般是 `CLIENT`。约定同时写明，一次客户端请求可能在服务端对应多次推理，服务端被埋点时会出现嵌套的推理 Span；统计操作数或 token 时，要按 `service.name` 分组。

一次带工具调用的 Agent 运行，Span 树大致是：

```text
invoke_agent triage                      (INTERNAL, conversation.id=c_19)
├── chat model-x                         (CLIENT, finish_reasons=["tool_use"], usage...)
├── execute_tool search_orders           (INTERNAL, tool.call.id=call_31, status=Error, error.type=timeout)
├── execute_tool search_orders           (INTERNAL, 重试)
├── invoke_agent refund-checker          (子 Agent)
│   └── chat model-y
└── chat model-x                         (finish_reasons=["stop"])
```

### 4.2 面试常被追问的规则

- **内容默认不采集**。`gen_ai.input.messages`、`gen_ai.output.messages`、`gen_ai.system_instructions`、`gen_ai.tool.definitions`、`gen_ai.tool.call.arguments`、`gen_ai.tool.call.result` 都是 Opt-In。约定也给了 `*_ref` 属性（如 `gen_ai.input.messages_ref`），用一个 URI 指向外部存储里的正文。这和第 3 节“事件里只放引用”是同一个思路。
- **压缩要能看出来**。`gen_ai.conversation.compacted` 标记这次调用用的是不是压缩后的上下文。复盘时它能帮你判断“模型忘了”是不是压缩造成的。
- **`gen_ai.agent.name` 必须是低基数**。不要用框架默认名或类型名，否则按 Agent 聚合指标时会炸出大量无意义的分组。
- **采样参数有属性**。`gen_ai.request.temperature`、`gen_ai.request.top_p`、`gen_ai.request.seed` 都有定义，复现时要用到（第 6 节）。
- **token 的包含关系**。`gen_ai.usage.input_tokens` 应当包含缓存 token 和所有模态的输入；`cache_read`、`cache_write` 是它的子集；`gen_ai.usage.output_tokens` 应当包含推理 token。服务商同时报计费数和实际消耗数时，报计费数。约定的 Anthropic 页专门写了一句：Anthropic 的 `input_tokens` 不含缓存读，必须加上去才得到 `gen_ai.usage.input_tokens`。

**代价**：约定还在变；自动埋点只覆盖框架内的调用，自己写的工具要手动打 Span；Span 多了，存储和查询成本都上去。

### 4.3 异步与长任务

Agent 的长任务会暂停等审批、跨进程恢复。一个 Trace 拖几小时不好查，也不好采样：尾部采样器要把整个 Trace 缓存在内存里等它结束。常见做法：每次恢复开一个新 Trace，用 Span Link 指回暂停前的 Span；Task ID 作为属性写在每个根 Span 上，按 Task ID 能查到整串 Trace。OTel 文档举的 Span Link 例子正是这种情况：一个操作排队后异步执行，新 Trace 要和原 Trace 关联，又不能作为它的子 Span。

## 5 采样：存多少、漏掉多少

**问题**：每天 100 万条 Trace 全存，存储和查询都扛不住；按比例丢掉一部分，出事时那一条恰好没留下。

### 5.1 头部采样与尾部采样

OTel 的 Sampling 概念文档把采样分成两类：

| 方式 | 在哪决定 | 好处 | 代价 |
|---|---|---|---|
| 头部采样（head sampling） | Trace 开始时，不看整个 Trace | 简单、易配置、高效，管道的任何位置都能做 | 不能按整个 Trace 的数据做决定，单靠它保证不了“出错的都留下” |
| 尾部采样（tail sampling） | 看过 Trace 的全部或大部分 Span 之后 | 可以按错误、总时延、某个属性的值来留 | 难实现，规则随系统变化；难运维，采样组件要有状态、缓存大量数据；常常绑定厂商 |

最常见的头部采样是一致性概率采样：按 trace ID 的哈希和目标比例决定留不留。同一个 trace ID 在每个服务上算出的结果相同，所以留下的是完整的 Trace，不会缺 Span。决定还会通过 `traceparent` 的 sampled 标志传给下游。

OTel Collector 的 Tail Sampling Processor（contrib 仓库，2026-11 查阅时稳定性为 beta）实现了尾部采样。它的策略包括 `status_code`（按 `OK`、`ERROR`、`UNSET` 留）、`latency`（按 Trace 最早开始到最晚结束的时长留）、`probabilistic`（按比例）等。几个默认值直接影响 Agent 场景：

| 配置 | 默认值 | 含义 |
|---|---|---|
| `decision_wait` | 30s | 收到 Trace 后等多久再做决定 |
| `num_traces` | 50000 | 内存里最多缓存多少条 Trace |
| `block_on_overflow` | false | 缓存满了时，false 表示驱逐最老的 Trace |
| `decision_cache` 的 `sampled_cache_size` | 0 | 记住多少个“已决定保留”的 trace ID，用来处理决定之后才到的 Span |

### 5.2 数值例子

假设：每天 N = 100 万条 Trace，每条平均 50 KB（只含结构化属性，不含正文）；错误率 p_e = 0.5%；慢请求（超过 30 秒）比例 p_s = 1%，两者不重叠。

**存储量**：

```text
全量：           1,000,000 × 50 KB = 50 GB/天
头部采样 r=10%：  50 GB × 0.10 = 5 GB/天
尾部采样（错误全留 + 慢请求全留 + 其余按 1% 留）：
  比例 = p_e + p_s + 0.01 × (1 − p_e − p_s) = 0.005 + 0.01 + 0.01 × 0.985 = 0.02485
  存储 = 50 GB × 0.02485 ≈ 1.24 GB/天
```

**漏掉错误的概率**：头部采样不看结果，每条错误 Trace 被留下的概率都是 r。一次事故里有 k 条错误 Trace，一条都没留下的概率是：

```text
P(全部漏掉) = (1 − r)^k

r = 10%：k = 1 → 0.90；k = 5 → 0.59；k = 20 → 0.12；k = 50 → 0.005
```

批量故障（k 大）头部采样基本能抓到样本。用户投诉的单条 bad case（k = 1）有 90% 的概率查不到。尾部采样按 `status_code` 留，错误 Trace 的留存率是 100%，存储反而只有头部采样的四分之一。

**尾部采样器的内存**：缓存里同时有约“到达率 × `decision_wait`”条 Trace（Little 定律，见 [05 第 8 节](../05-gateway-reliability/README.md)）。100 万条/天约 11.6 条/秒，乘 30 秒约 350 条，远低于默认 `num_traces`。到达率升到 2000 条/秒，就是 60,000 条，超过默认的 50,000，最老的 Trace 会被驱逐。

**Agent 的长 Trace**：一个任务跑 5 分钟，`decision_wait` 只有 30 秒。决定做出时，后面的 Span 还没到。这些晚到的 Span 要靠 `decision_cache` 沿用之前的决定；它默认大小是 0。这是第 4.3 节建议“长任务拆成多个 Trace”的另一个原因。

**用采样数据估总量**：头部采样率是 r，按 Trace 估总量要乘 1/r。尾部采样里不同类别的保留率不同，直接用留下的 Trace 算错误率会严重偏高：上面的例子里，留下的 Trace 中错误占 0.005 / 0.02485 ≈ 20%，真实错误率是 0.5%。总量和比例要从指标算，Trace 只用来看个案。

下面的代码用 20 万条模拟 Trace 验证上面的数字：

```python
import hashlib, random

def keep_head(trace_id, rate):
    """一致性头部采样：只看 trace_id 的哈希，同一 Trace 在每个服务上的决定都相同。"""
    h = int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16)
    return h < rate * 2**32

def keep_tail(t, base_rate, slow_ms):
    """尾部采样：Trace 结束后看全貌。错误、慢请求全留，其余按比例。"""
    return t["error"] or t["ms"] > slow_ms or keep_head(t["id"], base_rate)

rng = random.Random(7)
N, P_E, P_S = 200_000, 0.005, 0.01              # 错误率 0.5%，1% 的请求超过 30 秒
traces = []
for i in range(N):
    u = rng.random()
    traces.append({"id": f"t{i}", "error": u < P_E,
                   "ms": 45_000 if P_E <= u < P_E + P_S else 8_000})

head = [t for t in traces if keep_head(t["id"], 0.10)]
tail = [t for t in traces if keep_tail(t, 0.01, 30_000)]
n_err = sum(t["error"] for t in traces)

# 存储量：头部约 10%；尾部约 0.5% + 1% + 1% × 98.5% = 2.485%
assert abs(len(head) / N - 0.10) < 0.005
assert abs(len(tail) / N - 0.02485) < 0.002
assert 50 * 0.02485 < 1.25                       # 每天 50 GB 全量 → 尾部约 1.24 GB
# 错误留存：头部约 10%，尾部 100%
assert abs(sum(t["error"] for t in head) / n_err - 0.10) < 0.03
assert sum(t["error"] for t in tail) == n_err

# 一次事故有 k 条错误 Trace，头部 10% 采样一条都没留下的概率 = 0.9^k
for k, p in [(1, 0.90), (5, 0.59), (20, 0.12), (50, 0.005)]:
    assert abs(0.9 ** k - p) < 0.01
# 用尾部采样留下的 Trace 算错误率：约 20%，真实是 0.5%
biased = sum(t["error"] for t in tail) / len(tail)
assert 0.17 < biased < 0.23
# 用头部采样估总量：乘 1/r，误差很小
assert abs(len(head) / 0.10 - N) / N < 0.05

# 尾部采样器内存：到达率 × decision_wait
assert round(1_000_000 / 86_400 * 30) == 347
assert 2000 * 30 > 50_000                        # 超过默认 num_traces，开始驱逐
print(len(head), len(tail), n_err, round(biased, 3))
```

### 5.3 Agent 场景的组合

| 数据 | 怎么采 | 为什么 |
|---|---|---|
| 指标（时长、token、状态、版本维度） | 全量，从推理 Span 或网关直接出计数器 | 成本、告警、比例都靠它，不能受采样影响 |
| Span 的结构化属性 | 尾部采样：错误、慢、超预算全留，其余低比例 | 存储小，个案能查 |
| 正文（上下文、工具参数和结果） | 先写短期存储；被尾部采样选中或收到差评，再转长期 | 差评往往在请求结束后很久才到，尾部采样器早已做完决定 |

流量很大时可以两段叠加：服务端先做一道头部采样保护管道，Collector 再做尾部采样。OTel 文档提到了这种组合。代价是头部丢掉的错误，尾部也救不回来；所以头部那一道只适合用在流量特别大、错误率又可以由指标监控的服务上。

## 6 回放与复现线上问题

**问题**：用户反馈“第三步答错了”。拿同样的输入重跑，结果对了，问题复现不了。

### 6.1 为什么同样的输入跑不出同样的结果

| 来源 | 出处写明的事实 | 应对 |
|---|---|---|
| 采样 | temperature 大于 0 时本来就随机 | 复现时记下采样参数；评估修复效果要跑多次 |
| 服务端数值不确定 | Anthropic 文档写明，temperature 设为 0 结果也不完全确定。OpenAI 的 `seed` 参数是 Beta，只承诺“尽力”确定，并让你看 `system_fingerprint` 判断后端是否变了 | 记下 `response_model`、`system_fingerprint` 等版本信息；不把“重跑一致”当作修复的证据 |
| 推理批次变化 | Thinking Machines 的文章分析：服务端负载变化导致 batch size 变化，而算子不是批次不变的，同一请求的数值就会变；他们用 Qwen3-235B 贪心解码跑 1000 次，得到 80 种不同的结果，换成批次不变的算子后 1000 次完全一致 | 自部署时可以开批次不变模式，代价是吞吐 |
| 自部署框架 | vLLM 文档写明，默认不保证可复现；离线可以关多进程或开批次不变，在线只能开批次不变；只在同一硬件、同一版本下才可复现 | 复现环境固定硬件和版本 |
| 外部世界 | 工具返回的数据变了：库存、网页、文件 | 记下工具结果原文，重放时注入 |
| 时间与随机数 | Harness 读了当前时间、生成了随机 ID、随机选了重试间隔 | 时钟和随机数都经过可替换接口 |
| 上下文装配 | 检索结果、记忆、压缩摘要不同 | 记下实际发给模型的上下文，不只记用户输入 |

数值不确定的机制见 [core/01 第 6.5 节](../../../core/01-llm-foundations/README.md#65-温度设为-0-为什么结果还会变)，这里只说结论：模型端的不确定性在 Harness 这一侧消除不了，只能把模型输出当作“外部输入”记下来。

### 6.2 三种回放

| 回放方式 | 做法 | 能回答什么 | 不能回答什么 |
|---|---|---|---|
| 查看回放 | 按 Trace 把每一步的上下文、模型输出、工具结果按顺序展示 | 当时发生了什么 | 改了 prompt 之后会怎样 |
| 确定性重放 | 模型输出和工具结果都从记录里取，不调真实服务，只重跑 Harness 代码 | Harness 的逻辑（解析、路由、权限、压缩）有没有问题；修了 bug 后同样的输入是否走对 | 模型在新上下文下会怎么选 |
| 从第 k 步重新执行 | 前 k 步用记录，第 k 步起调真实模型，工具仍用桩或沙箱 | 换 prompt、换模型后，这个 case 会不会变好 | 结果有随机性，要跑多次看分布 |

### 6.3 确定性重放要记录什么

确定性重放的前提是：Harness 是一个确定性函数，输入只有两类，一类是用户请求，一类是外部调用的返回值。只要外部调用的返回值按原顺序注入，Harness 的每一步就和线上一样。由此推出要录制的内容，就是 Harness 和外界的每一个接触点：

| 接触点 | 录什么 | 不录会怎样 |
|---|---|---|
| 模型调用 | 完整响应：文本、工具调用、`finish_reason`、usage | 重放时只能调真模型，结果会变 |
| 工具调用 | 结果原文、错误类型；有副作用的要记下执行回执 | 外部数据已经变了；有副作用的动作会被重放二次执行 |
| 时钟 | 每次读到的时间值 | “是否超时”“缓存是否过期”的分支走得不一样 |
| 随机数 | 种子，或者每次取到的值 | 随机 ID、重试抖动、采样顺序不同 |
| 配置与版本 | prompt 版本、工具集版本、开关、环境变量 | 重放时用了新配置，分不清是代码变了还是配置变了 |

每个事件除了返回值，还要记一个**请求指纹**：把这次调用的请求（发给模型的上下文、工具名和参数）规范化后取哈希。回放时，Harness 每发起一次外部调用，回放层就对比三件事：

1. 调用类型和录制的第 i 个事件一致；
2. 请求指纹和录制的一致；
3. 回放结束时，录制的事件刚好用完。

任何一条不满足，就是**分叉**（divergence）：Harness 在这一步做出了和线上不同的请求。分叉不是坏事。修了 bug 再回放，分叉的位置正是修复生效的位置。没改代码却分叉了，说明有未录制的外部输入（漏录了时钟、读了环境变量、字典遍历顺序不稳定）。

### 6.4 录制回放的实现

下面的 `Tape` 是录制回放层，所有外部调用都通过 `tape.call(kind, request, live_fn)`。录制模式下调真实函数并存下结果；回放模式下从记录取结果，并检查请求指纹。被测的 `agent` 是一个退款 Agent：模型随机选一个金额，工具真的“退款”（往 `side_effects` 里追加一条）。

```python
import hashlib, json, random

def fp(obj):                                          # 请求指纹：规范化 JSON 后取哈希
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]

class Divergence(Exception):
    pass

class Tape:
    """录制回放层。所有外部输入都经过 call()：模型、工具、时钟、随机数。"""
    def __init__(self, mode, events=None):
        self.mode, self.events, self.i = mode, (events if events is not None else []), 0

    def call(self, kind, request, live_fn):
        if self.mode == "record":
            result = live_fn()
            self.events.append({"kind": kind, "req": fp(request), "result": result})
            return result
        if self.i >= len(self.events):                # 回放时比录制多走了一步
            raise Divergence(f"step {self.i}: extra {kind} call")
        ev = self.events[self.i]
        if ev["kind"] != kind or ev["req"] != fp(request):
            raise Divergence(f"step {self.i}: expected {ev['kind']}/{ev['req']}, got {kind}/{fp(request)}")
        self.i += 1
        return ev["result"]                           # 不调真实服务

    def finish(self):
        if self.mode == "replay" and self.i != len(self.events):
            raise Divergence(f"stopped at {self.i}/{len(self.events)}")

# ---- “线上”的外部世界：模型随机、工具有副作用、时间在走 ----
side_effects = []
world_rng = random.Random(1)                          # 线上随机源（固定种子只为让本例可复现）
def live_model(ctx):
    if "refund_result" in ctx[-1]:
        return {"text": "已退款", "tool": None}
    amount = world_rng.choice([30, 80])               # 模型每次的选择不一样
    return {"text": "", "tool": {"name": "refund", "args": {"order": "o1", "amount": amount}}}
def live_tool(call):
    side_effects.append(call)                         # 有副作用：真的退了款
    return {"refund_result": "ok", "id": "r_" + str(len(side_effects))}
def live_clock():
    return 1_700_000_000 + len(side_effects)

def agent(tape, system="v1"):
    """被测的 Harness 逻辑：每个外部调用都走 tape。"""
    ctx = [{"system": system, "user": "退掉订单 o1"}]
    ts = tape.call("clock", {}, live_clock)
    seed = tape.call("rng", {}, lambda: world_rng.randrange(2**31))
    trace = []
    for _ in range(4):
        out = tape.call("model", {"ctx": ctx}, lambda: live_model(ctx))
        if out["tool"] is None:
            trace.append(("final", out["text"]))
            break
        res = tape.call("tool", out["tool"], lambda: live_tool(out["tool"]))
        trace.append(("tool", out["tool"]["args"]["amount"]))
        ctx = ctx + [res]
    tape.finish()
    return {"ts": ts, "seed": seed, "trace": trace}

rec = Tape("record")
online = agent(rec)
assert [e["kind"] for e in rec.events] == ["clock", "rng", "model", "tool", "model"]
assert len(side_effects) == 1

# 1) 确定性回放：结果和线上一致，真实工具一次没调
for _ in range(3):
    assert agent(Tape("replay", rec.events)) == online
assert len(side_effects) == 1

# 2) 改了 system prompt：在第一次模型调用（事件 2）检出分叉
try:
    agent(Tape("replay", rec.events), system="v2")
    raise AssertionError("divergence not detected")
except Divergence as e:
    assert str(e).startswith("step 2: expected model")

# 3) 录制被截断（少录了最后一次模型调用）：回放走到末尾时检出
try:
    agent(Tape("replay", rec.events[:-1]))
    raise AssertionError("divergence not detected")
except Divergence as e:
    assert str(e) == "step 4: extra model call"

# 4) 对照：直接重跑结果会变，而且每次都真的退了款
reruns = [agent(Tape("record"))["trace"][0] for _ in range(20)]
assert len(set(reruns)) == 2 and len(side_effects) == 21
print("online", online["trace"], "events", len(rec.events))
```

四个断言对应四个结论：回放一致且不产生副作用；改了 prompt 会在第一次模型调用处分叉；录制不完整能检出；直接重跑既不一致，又会重复执行副作用。

### 6.5 回放的工程要点

- **有副作用的工具绝不能在重放里真执行**。重放发了一封邮件、退了一笔款，就是一次线上事故。工具注册时要声明有无副作用，重放模式下有副作用的工具一律走桩。上面的代码里，重放根本不调 `live_fn`，这是最安全的默认。幂等键的设计见 [04 工具编排](../04-tool-orchestration/README.md)。
- **从第 k 步重新执行**：前 k 个事件走回放，第 k 步起模型改为调真实服务，工具仍然走桩或沙箱。实现上只要让 `Tape` 在第 k 个模型事件后切换模式。因为第 k 步起结果有随机性，要跑多次看错误率，不能看一次。
- **指纹的范围要选对**。指纹包含整段上下文时，任何改动都会分叉，适合验证“没改代码就应当完全一致”。想验证“改了 prompt 后工具调用是否还一样”，指纹可以只取工具名和参数。
- **录制格式要带版本**。Harness 升级后改了事件结构，旧录制就读不了。事件里写 `schema_version`，回放层负责迁移。

所以复现线上问题的第一步不是“重跑一次”，而是用记录做确定性重放，把问题定位到某一步的上下文；再从这一步起重新执行多次，看错误率。

**从 bad case 到回归用例**：复盘完的 case 要进评测集，否则下次改 prompt 还会再错。常见做法：把第 k 步的上下文快照和工具桩一起存成一条回归用例，期望行为写成可检查的断言（调对工具、参数满足条件、不越权）。回归集怎么接进发布流程，见 [evaluation/05 线上评测、A/B 与回归门禁](../../evaluation/05-online-regression/README.md)。

## 7 Token 与成本统计

**问题**：账单明显超出预估，但说不清钱花在哪个租户、哪个功能、哪一步。

### 7.1 先统一口径：四项计量

各家 API 的 usage 字段含义不同，直接相加会算错：

| 来源 | 缓存 token 怎么报 | `input_tokens` 含不含缓存 |
|---|---|---|
| Anthropic Messages API | `cache_read_input_tokens`、`cache_creation_input_tokens` | 不含。文档写明它只是最后一个缓存断点之后的部分；总输入 = 三者相加 |
| OpenAI Responses API | `usage.input_tokens_details.cached_tokens` | 含 |
| OTel GenAI 约定 | `gen_ai.usage.cache_read.input_tokens`、`gen_ai.usage.cache_write.input_tokens` | 应当含；服务商同时报计费数和实际消耗数时，报计费数 |

全库统一用四项口径，和 [05 第 8 节](../05-gateway-reliability/README.md) 一致：

```text
U   未缓存输入（uncached input）
C_r 缓存读（cache read）
C_w 缓存写（cache write）
O   输出（含推理 token）

总输入 I = U + C_r + C_w
```

归一的方向：Anthropic 的 `input_tokens` 直接就是 U；OpenAI 的 U = `input_tokens` − `cached_tokens`；写 OTel 属性时反过来，`gen_ai.usage.input_tokens` = U + C_r + C_w。常见做法是在网关或 SDK 封装层做一次归一，落库只存四项，不存原始字段的混合体。

### 7.2 再按价格算钱

单次调用的金额：

```text
cost = ( U·P_in + C_w·P_in·m_w + C_r·P_in·m_r + O·P_out ) / 10^6
```

P_in、P_out 是每百万 token 的单价，m_w、m_r 是缓存写和缓存读的倍率。Anthropic 2026-11 的定价文档写明：5 分钟缓存写是基础输入价的 1.25 倍，1 小时缓存写是 2 倍，缓存读在多数模型上是 0.1 倍；Batch API 输入输出都打五折，这些倍率可以叠加（查阅于 2026-11-26，以定价表为准）。服务端工具另外计费，比如网页搜索按次数收费，不在 token 里。

**价格表要带生效日期**。价格会调，账单要按调用发生那天的价格算。价格表的每一行写成（模型, 生效日, 单价, 倍率），计算时取“生效日 ≤ 调用日”的最新一行。用当前价格重算历史账单，降价之后历史成本会被低估。

**分词器不同，token 数就不同**。同一段文本在不同模型上 token 数不一样。Anthropic 定价文档写明，Claude 4.7 及之后的模型用了新分词器，同样的文本大约多出 30% 的 token。换模型后不能拿旧的 token 数估成本。

### 7.3 别重复计数

OTel 的 token 指标文档写了一条警告：不要不按操作名过滤，就把所有 Span 上的 `gen_ai.usage.*` 加起来。同一批 token 可能同时出现在 Agent Span 和推理 Span 上，也可能被网关和推理基础设施的不同层各报一次。文档给的查询是：按 trace ID 过滤，只取 `gen_ai.operation.name` 属于 `chat`、`generate_content`、`text_completion`、`realtime_inference` 的 Span，再按 `service.namespace`、`service.name` 分组。

另一个口径问题是采样：Trace 被采样后，用 Trace 加总会少算。文档建议能用指标就用指标，因为指标不受 Trace 采样影响。指标有两族：`gen_ai.client.inference.usage.*` 是单调计数器，用来算总量和成本；`gen_ai.client.inference.operation.input_tokens` 和 `.output_tokens` 是直方图，只用来看 p95、p99 和离群值，文档写明不要用它们算总量或成本。

Trace 用来看“这一个任务的钱花在哪”，计数器用来看总量和趋势。

### 7.4 成本汇总的实现

下面的代码做四件事：把两家的 usage 归一成四项；按生效日期取价格；在一条 Trace 里只取推理类 Span、按 `service.name` 分组；对比不过滤直接加的结果。价格是示例值，不是任何厂商的真实价格，倍率取 Anthropic 文档里 5 分钟缓存写和缓存读的写法。

```python
from datetime import date

# 价格表：示例值，单位美元/百万 token，不是任何厂商的真实价格。
# 倍率按 Anthropic 定价文档的写法：5 分钟缓存写 1.25 倍，缓存读 0.1 倍（查阅于 2026-11）。
PRICES = [  # (模型, 生效日, 输入, 输出, 缓存写倍率, 缓存读倍率)
    ("model-x", date(2026, 1, 1),  3.00, 15.00, 1.25, 0.10),
    ("model-x", date(2026, 10, 1), 2.50, 12.50, 1.25, 0.10),   # 10 月降价
]

def price_at(model, day):
    rows = [p for p in PRICES if p[0] == model and p[1] <= day]
    return max(rows, key=lambda p: p[1])                         # 调用当天已生效的最新一行

def normalize(provider, u):
    """把各家 usage 归一成四项：未缓存输入、缓存读、缓存写、输出。"""
    if provider == "anthropic":     # input_tokens 不含缓存部分
        return dict(uncached=u["input_tokens"], cache_read=u.get("cache_read_input_tokens", 0),
                    cache_write=u.get("cache_creation_input_tokens", 0), output=u["output_tokens"])
    if provider == "openai":        # input_tokens 已含 cached_tokens
        cr = u["input_tokens_details"]["cached_tokens"]
        return dict(uncached=u["input_tokens"] - cr, cache_read=cr, cache_write=0,
                    output=u["output_tokens"])
    raise ValueError(provider)

def cost(n, model, day):
    _, _, pin, pout, mw, mr = price_at(model, day)
    return (n["uncached"] * pin + n["cache_write"] * pin * mw
            + n["cache_read"] * pin * mr + n["output"] * pout) / 1e6

a = normalize("anthropic", {"input_tokens": 2_000, "cache_read_input_tokens": 16_000,
                            "cache_creation_input_tokens": 0, "output_tokens": 300})
o = normalize("openai", {"input_tokens": 18_000, "input_tokens_details": {"cached_tokens": 16_000},
                         "output_tokens": 300})
assert a == o                                         # 同样的用量，两家口径归一后一致
assert a["uncached"] + a["cache_read"] + a["cache_write"] == 18_000   # 总输入 = 前三项之和
# 把 Anthropic 的 input_tokens 当总输入：缓存命中率的分母错了
assert 16_000 / 2_000 > 1 and abs(16_000 / 18_000 - 0.889) < 1e-3

# 一条 Trace：Agent Span 挂了聚合 usage，Harness 的 SDK 和网关又各报了一次推理
u1 = dict(uncached=2_000, cache_read=0,      cache_write=20_000, output=500)   # 第 1 次：写缓存
u2 = dict(uncached=2_000, cache_read=16_000, cache_write=0,      output=300)   # 第 2 次：读缓存
agg = {k: u1[k] + u2[k] for k in u1}
spans = [
    {"op": "invoke_agent", "svc": "harness", "model": "model-x", "u": agg},
    {"op": "chat", "svc": "harness", "model": "model-x", "u": u1},
    {"op": "chat", "svc": "harness", "model": "model-x", "u": u2},
    {"op": "chat", "svc": "gateway", "model": "model-x", "u": u1},
    {"op": "chat", "svc": "gateway", "model": "model-x", "u": u2},
    {"op": "execute_tool", "svc": "harness", "model": None, "u": None},
]
INFER = {"chat", "generate_content", "text_completion", "realtime_inference"}

day = date(2026, 9, 15)
naive = sum(cost(s["u"], s["model"], day) for s in spans if s["u"])
by_svc = {}
for s in spans:
    if s["op"] in INFER:                              # 只取推理类 Span，再按 service.name 分组
        by_svc[s["svc"]] = by_svc.get(s["svc"], 0) + cost(s["u"], s["model"], day)

# 手算（单位：美元 × 10^6）
#   第 1 次：2000×3 + 20000×3×1.25 + 500×15 = 6000 + 75000 + 7500 = 88,500
#   第 2 次：2000×3 + 16000×3×0.1  + 300×15 = 6000 +  4800 + 4500 = 15,300
expected = (88_500 + 15_300) / 1e6
assert abs(by_svc["harness"] - expected) < 1e-12 and abs(by_svc["gateway"] - expected) < 1e-12
assert abs(naive - 3 * expected) < 1e-12              # 不过滤直接加：算成 3 倍
# 同样的用量按 10 月价格算更便宜；历史账单必须按调用当天的价格
assert abs(cost(u1, "model-x", date(2026, 10, 2)) / cost(u1, "model-x", day) - 2.5 / 3) < 1e-12
# 第 1 次的缓存写只用了一次就不划算：不打缓存断点时这 20000 token 按 1 倍算
no_cache = cost(dict(uncached=22_000, cache_read=0, cache_write=0, output=500), "model-x", day)
assert no_cache < cost(u1, "model-x", day)
print(f"task ${expected:.4f}, naive ${naive:.4f}")
```

一个任务两次调用共 $0.1038。不过滤直接加得到 $0.3114，是 3 倍：Agent Span 一份、Harness 推理 Span 一份、网关推理 Span 一份。按 `service.name` 分组后，Harness 和网关各自给出 $0.1038，两者能互相对账。最后一个断言说明缓存写只用一次时比不缓存更贵，这和 [05](../05-gateway-reliability/README.md) 里“只在会复用的地方打缓存断点”是同一件事。

### 7.5 归因到谁

每次模型调用都带上租户、功能、Task、Step 和 Agent 名字：

| 问题 | 需要的维度 | 常见发现 |
|---|---|---|
| 哪个租户花得多 | 租户 | 少数租户的长任务占了大头 |
| 哪个功能不划算 | 功能 + 任务成功与否 | 失败任务也在花钱，按“每个成功任务的成本”比才公平 |
| 一个任务里哪一步贵 | Step + 上下文长度 | 后期每一步都带着很长的上下文；缓存命中率掉了 |
| 子 Agent 值不值 | Agent 名字 + 父子关系 | 子 Agent 探索花的 token 远超它交回的总结 |

“每个成功任务的成本”= 总成本 / 成功任务数。功能 A 每任务 $0.10、成功率 90%，功能 B 每任务 $0.08、成功率 60%：A 是 0.10 / 0.9 ≈ $0.111，B 是 0.08 / 0.6 ≈ $0.133。只看每任务成本会选 B，按成功任务算 A 更便宜。

缓存命中率要单独看：C_r / I。它突然下降，多半是有人改了 system prompt 或工具定义的顺序，破坏了稳定前缀，见 [02](../02-context-compaction/README.md)。

**维度的基数要控制**。租户、功能、Agent 名可以做指标维度；Task ID、Step ID 基数太高，只放在 Trace 和 Step 记录里，不放进指标标签，否则时序数据库的序列数会爆。

## 8 隐私、保留与访问控制

**问题**：Trace 里有用户输入、工具参数、检索到的文档。Trace 后端的访问权限通常比业务数据库宽，保留时间又长，它本身就成了一个敏感数据副本。

### 8.1 先看清默认值

- **OTel GenAI 约定**：内容类属性（输入输出消息、系统指令、工具参数和结果）都是 Opt-In，默认不采。
- **OpenAI Agents SDK**：`trace_include_sensitive_data` 默认是 `True`，`generation_span()` 和 `function_span()` 会记下模型和函数调用的输入输出。可以在 `RunConfig` 里关掉，也可以用环境变量 `OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA` 设默认值。

两家默认值相反。接入框架时先查默认值，不要假设“框架自带的 Trace 不含正文”。

### 8.2 脱敏失败要拦住

OpenAI Agents SDK 的文档写明：trace processor 是互相独立的观察者，默认 provider 会捕获某个 processor 回调里的异常，然后继续调用其他 processor。所以注册在导出器之前的脱敏 processor，脱敏失败时并不能阻止导出器收到数据。文档的建议是：导出依赖脱敏成功时，把脱敏和发送放在同一个应用自己的导出器里，并用 `set_trace_processors()` 替换默认 processor。

这背后是一条通用原则：**脱敏要 fail closed**。脱敏出错时丢弃正文，只导出结构化字段；不要因为脱敏失败就原样放行。

### 8.3 正文和结构化字段分开存

| 数据 | 例子 | 保留期（示例） | 权限 |
|---|---|---|---|
| 指标 | token 计数器、时长直方图 | 长，按容量规划 | 宽，团队可看 |
| Span 结构化属性 | 模型名、工具名、状态、耗时、四项 token | 中，例如 400 天 | 工程团队 |
| 正文 | 上下文、工具参数和结果、模型输出 | 短，例如 30 天；被差评或事故钉住的转长期 | 严，按审计日志的标准管，访问要留痕 |

保留期是示例值，实际按合规要求和数据类型定。正文单独存还有一个好处：用户要求删除数据时，Trace 里只放了引用，按用户或会话 ID 删掉对象存储里的正文即可，结构化字段可以匿名化后保留。

### 8.4 最小实现：脱敏、fail closed 与保留期

```python
import re
from datetime import datetime, timedelta

PATTERNS = [                                          # 示例规则，真实系统要按数据类型补全
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"(?<!\d)1\d{10}(?!\d)")),
    ("APIKEY", re.compile(r"sk-[A-Za-z0-9]{16,}")),
]

def redact(text):
    for tag, p in PATTERNS:
        text = p.sub(f"[{tag}]", text)
    return text

def export(span, redactor):
    """脱敏和导出放在同一个函数里：脱敏失败就丢弃正文（fail closed）。"""
    out = {k: v for k, v in span.items() if k != "body"}
    try:
        out["body"] = redactor(span["body"])
    except Exception:
        out["body"], out["body_dropped"] = None, True
    return out

span = {"name": "chat model-x", "gen_ai.usage.output_tokens": 300,
        "body": "联系 alice@example.com 或 13812345678，key=sk-abcdefghijklmnop1234"}
ok = export(span, redact)
assert ok["body"] == "联系 [EMAIL] 或 [PHONE]，key=[APIKEY]"
assert ok["gen_ai.usage.output_tokens"] == 300        # 结构化字段不受影响

def broken(_):
    raise ValueError("bad pattern")
bad = export(span, broken)
assert bad["body"] is None and bad["body_dropped"]    # 不会把原文放出去
assert bad["gen_ai.usage.output_tokens"] == 300       # 成本统计仍然完整

# 保留期：结构化字段 400 天，正文 30 天；被差评钉住的正文不按 30 天删
RETAIN = {"fields": timedelta(days=400), "body": timedelta(days=30)}
def expired(kind, created, now, pinned=False):
    return (not pinned) and now - created > RETAIN[kind]

t0, now = datetime(2026, 9, 1), datetime(2026, 11, 26)
assert expired("body", t0, now) and not expired("fields", t0, now)
assert not expired("body", t0, now, pinned=True)

# 按用户删除：正文按 user_id 建索引，Trace 只留引用
blobs = {"blob://u1/a": "正文 1", "blob://u1/b": "正文 2", "blob://u2/c": "正文 3"}
index = {"u1": ["blob://u1/a", "blob://u1/b"], "u2": ["blob://u2/c"]}
for ref in index.pop("u1"):
    del blobs[ref]
assert list(blobs) == ["blob://u2/c"]
print(ok["body"])
```

正则脱敏只能挡住格式固定的数据（邮箱、手机号、密钥）。姓名、地址、病情这类自由文本，正则抓不全，要靠“默认不采正文、按需短期保留”来控制风险，不能指望脱敏规则兜底。

### 8.5 其他要点

- **Baggage 和 Span 属性里不放凭据和个人信息**。Baggage 会被传给下游，可能被记进日志（第 2.3 节）。
- **租户隔离**：多租户平台里，Trace 查询要按租户过滤，一个租户的工程师不能看到另一个租户的正文。
- **Trace 后端是第三方服务时**，正文出境就是数据出境，要和业务数据走一样的合规审批。

## 9 数值例子：一个月的 Trace 和账单

把前面几节的数字串起来，算一个虚构的客服 Agent 一个月的量。所有数字都是假设，只用来练手算。

| 量 | 假设 | 计算 |
|---|---|---|
| 任务数 | 每天 2 万个任务 | 每月 60 万 |
| 每个任务 | 8 次模型调用、6 次工具调用 | 每个 Trace 1 + 8 + 6 = 15 个 Span |
| 每次模型调用 | U = 1,500，C_r = 14,000，C_w = 500，O = 400 | I = 16,000 |
| 示例单价 | P_in = $3，P_out = $15，m_w = 1.25，m_r = 0.1 | 示例值 |

**每次调用的成本**：

```text
U·P_in        = 1,500 × 3        = 4,500
C_w·P_in·m_w  =   500 × 3 × 1.25 = 1,875
C_r·P_in·m_r  = 14,000 × 3 × 0.1 = 4,200
O·P_out       =   400 × 15       = 6,000
合计 16,575 / 10^6 ≈ $0.0166
```

每个任务 8 次调用约 $0.133，每月 60 万任务约 $79,560。

**缓存命中率** = C_r / I = 14,000 / 16,000 = 87.5%。假设有人改了 system prompt 的开头，命中率掉到 0：14,000 个 token 从 0.1 倍变成首次写入的 1.25 倍（5 分钟缓存），这一项从 4,200 变成 52,500，单次调用涨到 64,875 / 10^6 ≈ $0.065，约 3.9 倍。之后如果前缀稳定下来，命中率会恢复；但每次改前缀都要付一次这个代价。

**重复计数的后果**：如果 Agent 框架在 `invoke_agent` Span 上也挂了聚合 usage，网关再报一次推理 Span，不过滤直接加，月账单会被算成约 $238,680，是实际的 3 倍。

**Trace 存储**：每月 60 万个 Trace，每个 15 个 Span，共 900 万个 Span。设每个 Span 的结构化属性约 2 KB，全量约 18 GB/月。正文另算：每次模型调用的上下文 16K token，按 1 token 约 4 字节估是 64 KB；按内容哈希去重后每次只存增量，增量约 U + O = 1,900 token，约 7.6 KB。每月 480 万次调用，不去重约 307 GB，去重后约 36 GB。

**采样**：错误率 0.5%，每月约 3,000 个错误任务。头部采样 10% 平均只留 300 个；某一位用户投诉的那一条有 90% 的概率没留下。尾部采样错误全留，加上慢请求 1% 和其余 1%，留约 2.5%，也就是约 1.5 万个 Trace。

```text
单次调用成本              ≈ $0.0166
每任务（8 次）            ≈ $0.133
每月（60 万任务）          ≈ $79,560
命中率归零后单次          ≈ $0.065（3.9 倍）
不过滤三层叠加            ≈ $238,680（3 倍）
Span 结构化属性           ≈ 18 GB/月
正文：不去重 / 去重        ≈ 307 GB / 36 GB
```

下面的代码把这张表逐项算一遍：

```python
P_IN, P_OUT, M_W, M_R = 3.0, 15.0, 1.25, 0.10       # 示例单价，美元/百万 token
def cost(U, Cr, Cw, O):
    return (U * P_IN + Cw * P_IN * M_W + Cr * P_IN * M_R + O * P_OUT) / 1e6

U, Cr, Cw, O = 1_500, 14_000, 500, 400
I = U + Cr + Cw
assert I == 16_000 and Cr / I == 0.875               # 命中率 87.5%
call = cost(U, Cr, Cw, O)
assert abs(call - 0.016575) < 1e-12
tasks, calls_per_task = 20_000 * 30, 8
month = call * calls_per_task * tasks
assert round(month) == 79_560
miss = cost(U, 0, Cw + Cr, O)                        # 前缀被改：缓存读全变成缓存写
assert abs(miss - 0.064875) < 1e-12 and round(miss / call, 1) == 3.9
assert round(3 * month) == 238_680                   # 三层叠加重复计数

spans = tasks * (1 + 8 + 6)
assert spans == 9_000_000 and spans * 2 / 1e6 == 18  # 2 KB/Span → 18 GB
n_calls = tasks * calls_per_task
full = n_calls * I * 4 / 1e9                         # 1 token ≈ 4 字节
dedup = n_calls * (U + O) * 4 / 1e9
assert round(full) == 307 and round(dedup) == 36

errors = tasks * 0.005
assert errors == 3_000 and errors * 0.10 == 300
kept_tail = tasks * (0.005 + 0.01 + 0.01 * 0.985)
assert round(kept_tail) == 14_910
print(round(call, 4), round(month), round(miss / call, 2))
```

## 10 工程细节与常见坑

**坑 1：Trace 树断成几截。**
- 现象：Trace 后端里一个任务显示成好几个 Trace，或者工具 Span 成了没有父亲的孤儿。
- 原因：跨进程时没传 `traceparent`；异步任务、线程池里没带上当前 context（`contextvars` 在新线程里默认是空的）；父 Span 因为进程崩溃没上报。
- 排查：看孤儿 Span 的 parent span ID 在后端里能不能查到；在工具服务入口打印收到的 `traceparent`。
- 处理：HTTP 客户端和消息队列都接上 propagator；提交到线程池时显式复制 context；长任务按第 4.3 节拆成多个 Trace，用 Span Link 连起来。

**坑 2：token 总数对不上账单。**
- 现象：Trace 里加出来的 token 是账单的 2 到 3 倍，或者只有一小部分。
- 原因：多了是重复计数（Agent Span、多层推理 Span 都报了 usage）；少了是 Trace 被采样，或者把 Anthropic 的 `input_tokens` 当成了总输入。
- 排查：同一个 trace ID 下按 `gen_ai.operation.name` 和 `service.name` 分组看 token；对一次调用，把原始响应的 usage 和 Span 属性逐项比对。
- 处理：总量和成本只从计数器指标出；Trace 加总只取推理类 Span、按服务分组；归一在一处完成（第 7.1 节）。

**坑 3：缓存命中率突然下降，成本上涨。**
- 现象：输入 token 总数没变，C_r 下降、C_w 和 U 上升，每任务成本涨了几倍（第 9 节的例子是 3.9 倍）。
- 原因：system prompt 开头加了时间戳或请求 ID；工具定义的顺序变了；换了模型（缓存不跨模型）。
- 排查：按版本指纹分组画命中率曲线，下降点对应哪次发布；比对两个版本发给模型的上下文前缀。
- 处理：可变内容放到稳定前缀之后；工具定义排序固定；命中率作为发布门禁的一项指标。

**坑 4：尾部采样丢了长任务的后半段。**
- 现象：一个 5 分钟的任务，Trace 里只有前 30 秒的 Span。
- 原因：Tail Sampling Processor 的 `decision_wait` 默认 30 秒，决定做完后内存里的数据就释放了；`decision_cache` 默认大小为 0，晚到的 Span 拿不到之前的决定。
- 排查：看 Collector 的指标里有没有晚到 Span 被丢弃；对比 SDK 侧导出的 Span 数和后端收到的 Span 数。
- 处理：长任务拆成多个 Trace；按需调大 `decision_wait` 和 `decision_cache`，同时按“到达率 × `decision_wait`”核算内存。

**坑 5：回放没改代码也分叉。**
- 现象：同一份录制，同一版代码，回放在某一步报分叉。
- 原因：有未录制的外部输入：直接调了 `time.time()`、`uuid4()`、读了环境变量；遍历了顺序不稳定的集合；请求指纹里包含了每次都变的字段（时间戳、随机 ID）。
- 排查：打印分叉那一步录制的请求和回放的请求，做文本 diff，看是哪个字段不同。
- 处理：所有时间、随机数、环境读取都走可替换接口；指纹计算前去掉易变字段，JSON 用 `sort_keys`。

**坑 6：重放触发了真实副作用。**
- 现象：复盘一次 bad case 后，用户又收到一封邮件，或者订单被二次退款。
- 原因：回放层只拦截了模型调用，工具调用仍然走真实实现；或者“从第 k 步重新执行”时，工具没有切到桩。
- 排查：在回放模式下给所有有副作用的工具加告警，统计回放期间的真实调用次数。
- 处理：回放模式下默认不调任何真实工具（第 6.4 节的 `Tape` 就是这样）；工具注册时声明 `side_effect`，有副作用的工具在非线上模式下直接拒绝执行。

**坑 7：正文泄露到 Trace 后端。**
- 现象：在 Trace 后端搜到用户手机号、API Key。
- 原因：框架默认采集正文（如 OpenAI Agents SDK 的 `trace_include_sensitive_data` 默认 `True`）；脱敏 processor 抛异常后，导出器照样收到原文；把用户 ID、Key 放进了 Baggage。
- 排查：定期用敏感数据规则扫描 Trace 后端；检查脱敏 processor 的异常日志。
- 处理：显式关闭正文采集，需要时用 `*_ref` 指向受控存储；脱敏和导出放在同一个导出器里，失败就丢正文（第 8.4 节）。

**坑 8：用采样后的 Trace 算比例。**
- 现象：仪表盘上的错误率是 20%，实际只有 0.5%。
- 原因：尾部采样对错误全留、对正常请求只留 1%，留下的样本里错误被放大了（第 5.2 节）。
- 排查：对比指标算出的错误率和 Trace 算出的错误率。
- 处理：比例和总量从指标算；必须用 Trace 估时，按每类的保留率加权（每条 Trace 乘 1 / 保留率）。

## 11 对比与取舍

### 11.1 采样方式

| 方式 | 存储（第 5.2 节的假设） | 错误留存 | 实现难度 | 适用 |
|---|---|---|---|---|
| 全量 | 100% | 100% | 低 | 流量小（OTel 文档说每秒几十条以下的小 Trace 可以不采样）、法规要求不能丢 |
| 头部采样 10% | 10% | 10%，单条 bad case 90% 查不到 | 低，可在 SDK 里做 | 流量大、只看趋势、错误由指标监控 |
| 尾部采样 | 约 2.5% | 100% | 高，Collector 要有状态，按到达率配内存 | 要查个案的 Agent 平台 |
| 头部 + 尾部 | 更少 | 头部丢掉的救不回 | 高 | 流量极大的服务，先保护管道 |

### 11.2 回放方式

| 方式 | 成本 | 确定性 | 回答的问题 | 选它的时候 |
|---|---|---|---|---|
| 查看回放 | 只读存储 | 不涉及 | 当时发生了什么 | 所有复盘的第一步 |
| 确定性重放 | 不调模型和工具，几乎为零 | 完全一致，分叉可检出 | Harness 代码有没有 bug；修复是否生效 | 怀疑解析、路由、权限、压缩出错 |
| 从第 k 步重新执行 | 每次调真实模型 | 有随机性，要跑多次 | 换 prompt 或模型后这一步会不会变好 | 怀疑模型决策，要评估修复效果 |
| 直接重跑 | 全部真实调用，可能有副作用 | 不一致 | 几乎回答不了什么 | 不推荐 |

### 11.3 成本数据从哪来

| 来源 | 完整性 | 粒度 | 坑 |
|---|---|---|---|
| 厂商账单 | 最准，含服务端工具和折扣 | 粗，通常到项目或 API Key | 延迟一天以上，拆不到任务 |
| 网关计数器 | 全量，不受采样影响 | 租户、模型、功能 | 只看得到经过网关的调用 |
| Trace 推理 Span | 被采样 | 到任务、到步 | 重复计数；采样后少算 |
| Step 记录 | 全量（如果全量落库） | 到步 | 存储成本；和账单的对账要自己做 |

常见组合：网关计数器出总量和告警，每天和厂商账单对账；Trace 和 Step 记录用来解释“为什么这个任务贵”。

### 11.4 自建还是用框架自带的 Trace

| 选项 | 好处 | 代价 |
|---|---|---|
| 框架自带（如 Agents SDK 的 tracing） | 开箱即用，Span 结构现成 | 默认值可能采正文；后端常绑定框架厂商；自定义工具要手动打点 |
| OTel + GenAI 约定 | 后端可换，和其他服务的 Trace 能连上 | 约定还在 Development，属性名会变；要自己维护埋点 |
| 只记 Step 记录 | 回放和计费最直接 | 没有时序视图，跨服务调用接不上 |

Agent 平台一般两者都要：OTel Span 用来看时序和跨服务，Step 记录用来回放和计费，两边用 trace ID 和 step ID 互相引用（第 3 节）。

## 12 自测清单

- [ ] 说出 Span 的字段（Span Context、父 ID、起止时间、属性、事件、链接、状态、Kind），解释为什么 Span 只记父 ID
- [ ] 写出自身耗时的定义，说明为什么所有 Span 自身耗时之和等于根 Span 时长，并行子 Span 时怎么算
- [ ] 写出 `traceparent` 的格式，说出 sampled 标志的作用和 Baggage 里不能放什么
- [ ] 写出一条 Step 记录的字段，说出版本指纹和上下文引用缺了会怎样，推导上下文全文存储为什么近似平方增长
- [ ] 画出一次 Agent 运行的 Span 树，说出三类 Span 的名字格式、Kind、关键属性，哪些属性是 Opt-In，`*_ref` 解决什么问题
- [ ] 给定错误率和采样率，算出头部和尾部采样的存储比例，以及 k 条错误 Trace 全部漏掉的概率
- [ ] 说出尾部采样器的内存怎么估，长任务为什么会丢后半段
- [ ] 列出至少四个“重跑结果不一样”的来源和对应做法
- [ ] 说出确定性回放要录哪些接触点，回放时怎样检出分叉，分叉说明了什么
- [ ] 写出四项口径和单次成本公式，说出 Anthropic 和 OpenAI 的 `input_tokens` 差在哪
- [ ] 解释为什么不能把所有 Span 的 `gen_ai.usage.*` 加起来，为什么总量用计数器不用 Trace
- [ ] 说出 Trace 里正文的脱敏、保留期和删除怎么设计，为什么脱敏要 fail closed

## 资料

- OpenTelemetry，[Traces](https://opentelemetry.io/docs/concepts/signals/traces/)，概念文档（Span 字段、Span Link、Span Kind），查阅于 2026-11-26
- OpenTelemetry，[Context propagation](https://opentelemetry.io/docs/concepts/context-propagation/)，概念文档（Baggage 不放敏感信息），查阅于 2026-11-26
- OpenTelemetry，[Sampling](https://opentelemetry.io/docs/concepts/sampling/)，概念文档（头部与尾部采样），查阅于 2026-11-26
- OpenTelemetry，[Tail Sampling Processor](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/processor/tailsamplingprocessor)，Collector contrib 文档（策略类型、`decision_wait`、`num_traces`、`decision_cache` 默认值，稳定性 beta），查阅于 2026-11-26
- W3C，[Trace Context](https://www.w3.org/TR/trace-context/)，W3C Recommendation，2021-11-23
- OpenTelemetry，[Semantic conventions for generative AI systems](https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai)，含 Client Inference、Agent spans、Execute Tool、Inference Token Metrics（How to aggregate token usage）、Anthropic 等页，状态 Development，查阅于 2026-11-26
- OpenAI，[Tracing](https://openai.github.io/openai-agents-python/tracing/)，OpenAI Agents SDK 文档（Sensitive data、Redaction before export），查阅于 2026-11-26
- OpenAI，[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)，API 文档，查阅于 2026-11
- OpenAI，[Create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)（`seed` 与 `system_fingerprint`），API 参考，查阅于 2026-11
- Anthropic，[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)，Claude API 文档，查阅于 2026-11
- Anthropic，[Pricing](https://platform.claude.com/docs/en/about-claude/pricing)，Claude API 文档（缓存倍率、Batch 折扣、新分词器约多 30% token），查阅于 2026-11-26
- Anthropic，[Glossary](https://platform.claude.com/docs/en/about-claude/glossary)（Temperature 条目），Claude API 文档，查阅于 2026-11
- Horace He（Thinking Machines），[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，2025-09
- vLLM，[Reproducibility](https://docs.vllm.ai/en/latest/usage/reproducibility.html)，使用文档，查阅于 2026-11
