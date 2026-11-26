# 上下文装配、压缩与断点续跑

> 层级：垂类 · harness · 预计 180 分钟 · 题目：[questions.md](questions.md)
> 前置：[core/03 Prompt 与 Context Engineering](../../../core/03-prompt-context/README.md)、[harness/01 Agent Loop 与 Harness 的边界](../01-loop-and-harness/README.md)

## 1 这一章解决什么问题

Agent 跑十几步以后，每一步送给模型的输入都是 Harness 拼出来的：工具定义、系统提示、历史、工具结果、记忆、当前指令。拼什么、拼多少、放在哪，决定了效果、成本和延迟。任务再长一点，历史会超过窗口；进程会崩；用户会关掉页面明天再来。

最直接的做法是把所有历史原样带上，满了就截断开头。它有三个问题：

- 成本按步数的平方涨，因为每一步都要重送全部历史；
- 截掉的开头往往是任务目标和硬约束，模型会忘了自己在做什么；
- 上下文是唯一的状态，进程一崩，做到哪一步、哪些写操作已经生效，都说不清。

面试里常见的问法是：“窗口都百万了为什么还要压缩？”“压缩后怎么知道没丢东西？”“进程在第七步崩了，怎么不重复副作用地接着跑？”“缓存命中率突然掉了怎么查？”

学完你应该能：

- 写出上下文长度和累计输入随步数增长的式子，说清工具结果为什么是主要增长源；
- 对清理、摘要、重置、外置四种压缩，说出各自丢什么、留什么、能不能找回；
- 用事件溯源的思路讲清“上下文是从日志装配出的视图”，在任意崩溃点恢复，并保证有副作用的步骤只生效一次；
- 算出一次压缩让缓存失效的代价，判断它多少次请求后回本；
- 设计一套评测，证明压缩没有丢关键信息；
- 算清子 Agent 隔离上下文的收益和代价。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| P | 固定前缀：工具定义加系统提示 |
| U | 用户的任务描述 |
| a | 每步模型输出的 token 数，含工具调用参数 |
| r | 每步工具结果的 token 数 |
| g | 不压缩时每步新增，g = a + r |
| k, n | 第 k 次模型调用；一共 n 次 |
| C_k | 第 k 次调用时的上下文长度 |
| S_n | 累计输入 token，S_n = C_1 + … + C_n |
| W | 上下文窗口 |
| K | 清理时保留最近 K 个工具结果 |
| T, s | 摘要的触发阈值、摘要长度 |
| w, ρ | 缓存写入、读取的价格，相对基础输入价的倍数 |

## 2 上下文由哪些部分装配

### 2.1 六个组成部分

Anthropic 把 context engineering 定义为：在推理时挑选并维护最优的一组 token。它是 prompt engineering 的延伸：prompt 写一次，上下文每次调用都要重新挑。

| 部分 | 变化频率 | 怎么增长 | 排列与预算原则 |
|---|---|---|---|
| 工具定义 | 跨请求稳定 | 不增长，工具多时一次就很大 | 放最前，顺序和序列化固定 |
| 系统提示、项目规则 | 跨请求稳定 | 不增长 | 紧跟工具定义，不放时间戳、请求 ID |
| 长期记忆、用户偏好 | 变化慢 | 按检索结果注入，有上限 | 按相关性挑几条，不整段塞，见 [core/06 记忆](../../../core/06-memory/README.md) |
| 模型输出 | 每步一段 | 每步加 a，通常几百 token | 只追加 |
| 工具结果 | 每步不同 | 每步加 r，文件、网页、日志动辄上千 | 返回时就裁剪；大块内容存外部、留引用 |
| 当前指令 | 每轮更新 | 不累积 | 放末尾，靠近生成位置 |

前两行和第三行加起来是固定前缀 P。后面三行里，只有模型输出和工具结果会逐步累积。

### 2.2 增长模型：工具结果是主要增长源

不压缩时，第 k 次调用的上下文是固定部分加上前 k−1 步的输出和结果：

```text
C_k = P + U + (k − 1)·g,          g = a + r

S_n = Σ_{k=1..n} C_k
    = n·(P + U) + g·(0 + 1 + … + (n − 1))
    = n·(P + U) + g·n(n − 1)/2                   （平方项）

第一次超窗：C_k > W  ⇔  k − 1 > (W − P − U)/g
           k* = ⌈(W − P − U)/g⌉ + 1
```

两点结论：

1. 窗口什么时候撞上，只看斜率 g。P 再大也只是平移。
2. 累计输入有 g·n²/2 这一项。步数翻倍，累计输入大约翻四倍。

g 里 r 通常远大于 a：模型说一句“读 main.py”只有几十 token，读回来的文件可能是几千。第 8 节的例子里 a = 300、r = 2,500，工具结果占每步新增的 89%。所以压低 r 是第一位的，这也是 Anthropic 称工具结果清理“最安全、最轻”的原因之一。Manus 的文章报告，他们的平均输入、输出 token 之比约 100:1，一个典型任务平均约 50 次工具调用。按上面的模型推导，这个比例主要来自每步重送的历史。

**有缓存时平方项还在吗**。只追加的历史里，第 k 次请求的前缀和第 k−1 次请求完全相同，可以按读价 ρ 付费，只有新增的 g 按写价 w 付费：

```text
第 1 次：全部写入            w·(P + U)
第 k 次（k ≥ 2）：读 C_{k−1}，写 g

总成本 = w·(P + U) + ρ·Σ_{k=2..n} C_{k−1} + w·g·(n − 1)
       = w·(P + U) + ρ·(S_n − C_n) + w·g·(n − 1)
```

平方项还在，只是乘了 ρ。Anthropic 的 5 分钟缓存 ρ = 0.1，成本大约降到八分之一，但曲线的形状不变。缓存降的是常数，压缩降的是斜率，两者要一起用。代价是压缩会改写历史，破坏缓存，第 5 节算这笔账。

### 2.3 为什么不全塞进去

除了成本，还有效果。Anthropic 引用“大海捞针”类研究说明：窗口里的 token 越多，模型准确回忆其中信息的能力越差，所有模型都有这个现象，只是程度不同。文中叫 context rot，给出两个原因：n 个 token 之间有 n² 对注意力关系；训练数据里短序列更多。结论是把上下文当成边际收益递减的有限资源。

位置也有影响。Liu 等人的 Lost in the Middle 论文报告：相关信息在输入开头或结尾时，表现通常最好；在长上下文的中间时，表现明显下降，显式支持长上下文的模型也一样。这是“当前指令放末尾”的依据。Manus 的做法也是同一个思路：任务进行中不断改写一个 todo.md，把全局计划“复述”到上下文末尾，放进模型最近的注意力范围。

**按需加载**。另一种做法是不预先装入数据，只放轻量的引用（文件路径、查询语句、链接），让 Agent 用工具在需要时读取。Anthropic 文中说 Claude Code 是混合做法：项目说明文件一开始就装入，具体文件用 glob 和 grep 按需读。代价是运行时探索比预先算好的检索慢。

## 3 压缩策略：清理、摘要、重置、外置

### 3.1 四种手段各丢什么、留什么

按损失从小到大分级，先做能找回的，不够再做有损的。

| 手段 | 做法 | 留下什么 | 丢掉什么 | 能找回吗 | 对增长的作用 |
|---|---|---|---|---|---|
| 清理 | 旧工具结果换成占位符，可选连调用参数一起清 | 模型的每一步决策、调用了什么 | 结果原文 | 能，原文在日志或能重新读 | 斜率从 a + r 降到 a + 占位符 |
| 外置 | 结果原文写进文件或对象存储，上下文只留路径和一行摘要；Agent 主动写笔记、进度 | 引用和摘要行 | 原文不进上下文 | 能，按路径读回 | 斜率降到 a + 引用长度 |
| 摘要 | 把早期轮次换成一段摘要，同一个 Agent 继续 | 摘要里写到的内容 | 摘要没写到的一切，包括措辞和细节 | 只有原文另存才能 | 长度回落到 P + U + s，锯齿形 |
| 重置 | 清空上下文，开一个新会话，用结构化交接文件接上 | 交接文件里的字段 | 其余全部，包括“为什么这样做” | 只有原文另存才能 | 回到起点 |

Manus 的文章把这件事说得很直接：任何不可逆的压缩都有风险，所以他们的压缩策略总是可恢复的。网页内容可以从上下文里丢掉，只要 URL 还在；文档内容可以省略，只要它在沙箱里的路径还在。这条原则把“清理”和“外置”放在“摘要”之前。

几个出处里写明的做法：

- **API 侧的清理**。Claude API 的 context editing 提供 `clear_tool_uses_20250919` 策略：上下文超过阈值时，按时间顺序清掉最旧的工具结果，每个换成一段占位文本，告诉模型它被删了。默认 `trigger` 是 100,000 输入 token，`keep` 是最近 3 次工具调用；`exclude_tools` 列出永远不清的工具；`clear_tool_inputs` 设为 true 时连调用参数一起清。这个清理在服务端做，客户端保留完整、未改动的历史。
- **API 侧的摘要**。Claude API 的阈值压缩（`compact_20260112`）默认在 150,000 输入 token 触发，最低可设 50,000。触发后 API 生成一个 `compaction` 块，之后的请求里，这个块之前的内容全部被忽略。`instructions` 参数会完全替换默认的摘要 prompt，不是追加。默认 prompt 要求写下“状态、下一步、学到的东西”等能让任务在新上下文里继续的信息。
- **Claude Code 的压缩**。让模型总结历史，保留架构决策、未解决的 bug 和实现细节，丢掉冗余工具结果，然后带着摘要和最近访问的五个文件继续。
- **调摘要 prompt**。Anthropic 建议在复杂轨迹上先把召回做满，再提精确度。原因是两种错误代价不对称：漏了关键信息，后面要多花很多步才能补回，甚至补不回；多留一点冗余，只是多花一些 token。
- **长期规则放在哪**。Claude Agent SDK 文档提醒：压缩会把早期消息换成摘要，开头的具体指令可能保不住。长期规则应该放在每次请求都会重新注入的项目说明文件里，不要只写在第一条消息里。SDK 提供 `PreCompact` hook，可以在压缩前把完整轨迹存档。

### 3.2 对正确性的影响：丢的是哪类信息

四种手段丢的东西不同，坏掉的方式也不同。

| 丢了什么 | 典型来源 | 现象 |
|---|---|---|
| 失败过的路径和原因 | 摘要只写“已完成” | 重复尝试已经失败的方案，步数变多，通过率可能不变（第 6 节代码） |
| 用户原话里的硬约束 | 摘要把约束“改写”成更宽松的说法 | 输出格式不对、越过了用户说的范围 |
| 还没确认结果的副作用 | 摘要写成“已提交” | 恢复后重复提交，或者以为提交了其实没有 |
| 精确的标识符 | 摘要里把 ID、路径、版本号写成概称 | 后续工具调用参数错误，要重新查 |
| 旧工具结果的具体数字 | 清理 | 模型需要时会重新调工具，多花一步；如果它没意识到被清了，可能凭印象编 |

所以有一张“不能压掉”的清单：

- 任务目标和用户原话里的硬约束；
- 权限边界和输出格式要求；
- 失败过的路径和失败原因。Manus 的经验是把走错的路留在上下文里：抹掉失败就抹掉了证据，模型看到失败的动作，会降低重复同样动作的倾向；
- 未完成的步骤和还没确认结果的副作用；
- 已经引用过的来源编号、文件路径、ID。

摘要格式要固定，例如“目标 / 约束 / 已完成 / 失败过的 / 待办 / 产物引用”，方便逐项检查（第 6 节）。

### 3.3 实现：几种策略的 token 曲线

下面按第 2.2 节的模型模拟 300 步，比较五种策略。每次调用的上下文表示成片段列表，同时算两个量：累计输入 token，以及按 Anthropic 5 分钟缓存价（w = 1.25、ρ = 0.1）算的带缓存成本。

- `none`：不压缩；
- `clear`：每步都把第 K 步以前的工具结果换成占位符（K = 3）；
- `clear_batch`：攒够 20 个旧结果再一次清掉，对应 API 里 `clear_at_least` 的思路；
- `offload`：结果原文进文件，上下文只留 60 token 的引用；
- `summary`：超过 T = 100K 就把历史换成 4K 的摘要；
- `clear+summary`：两者叠加。

```python
# 上下文长度随步数的增长，以及几种压缩策略的 token 曲线和带缓存的成本
P, U = 8_000, 500        # 固定前缀（工具定义 + 系统提示）、用户任务描述
a, r = 300, 2_500        # 每步模型输出（含工具调用参数）、每步工具结果
g = a + r                # 不压缩时每步新增
n, W = 300, 200_000      # 步数、窗口（不压缩的曲线超窗后只作理论对照）
K, ph = 3, 20            # 清理：保留最近 K 个工具结果，旧的换成 ph token 的占位符
M = 20                   # 批量清理：攒够 M 个可清理的旧结果才一次清掉（对应 clear_at_least）
ref = 60                 # 外置：结果原文写进文件，上下文里只留路径和一行摘要
T, s = 100_000, 4_000    # 摘要：上下文超过 T 就把历史换成 s token 的摘要


def run(policy):
    """返回每次模型调用的上下文（片段列表），片段 = (名字, token 数)。"""
    hist, summary, n_compact, ctxs = [], None, 0, []
    for k in range(1, n + 1):
        def view():
            head = [("prefix", P + U)] + ([summary] if summary else [])
            return head + [seg for step in hist for seg in step]
        ctx = view()
        if policy in ("summary", "clear+summary") and sum(x for _, x in ctx) > T:
            summary, hist, n_compact = (f"summary{k}", s), [], n_compact + 1
            ctx = view()
        ctxs.append(ctx)
        res = (f"r{k}", ref) if policy == "offload" else (f"r{k}", r)
        hist.append([(f"a{k}", a), res])
        old = [st for st in hist[:-K] if st[1][1] == r]       # 还没清理的旧结果
        if policy in ("clear", "clear+summary") or (policy == "clear_batch" and len(old) >= M):
            for st in old:
                st[1] = (st[1][0] + "-cleared", ph)
    return ctxs, n_compact


def length(ctx):
    return sum(x for _, x in ctx)


def cached_cost(ctxs, write=1.25, read=0.1):
    """断点放在末尾：和上一次请求相同的前缀按读价，其余按 5 分钟写价。单位：基础输入价。"""
    cost, prev = 0.0, []
    for ctx in ctxs:
        hit = 0
        for x, y in zip(ctx, prev):
            if x != y:
                break
            hit += x[1]
        cost += read * hit + write * (length(ctx) - hit)
        prev = ctx
    return cost


policies = ["none", "clear", "clear_batch", "offload", "summary", "clear+summary"]
R = {p: run(p) for p in policies}
L = {p: [length(c) for c in R[p][0]] for p in policies}
S = {p: sum(L[p]) for p in policies}                          # 累计输入 token
C = {p: cached_cost(R[p][0]) for p in policies}
for p in policies:
    print(f"{p:14s} 第80步 {L[p][79]:>8,}  峰值 {max(L[p]):>8,}  累计 {S[p]:>12,}"
          f"  带缓存成本 {C[p]:>12,.0f}  摘要 {R[p][1]} 次")

# 1) 不压缩：C_k = P+U+(k-1)g，累计 S_n = n(P+U) + g·n(n-1)/2，平方量级
assert all(L["none"][k - 1] == P + U + (k - 1) * g for k in range(1, n + 1))
assert S["none"] == n * (P + U) + g * n * (n - 1) // 2 == 128_130_000
k_star = next(k for k, c in enumerate(L["none"], 1) if c > W)  # 第一次超窗
assert k_star == -(-(W - P - U) // g) + 1 == 70

# 不压缩、带缓存：第 1 次全写，之后每次读上一请求的全部（0.1 倍）、写新增的 g（1.25 倍）
# 总成本 = 1.25(P+U) + 0.1·(S_n − C_n) + 1.25·g·(n−1)，平方项还在，只是乘了 0.1
assert abs(C["none"] - (1.25 * (P + U) + 0.1 * (S["none"] - L["none"][-1])
                        + 1.25 * g * (n - 1))) < 1e-6

# 2) 清理和外置只降斜率：K 步以后每步只涨 a + ph（或 a + ref），还是线性增长
assert all(L["clear"][k] - L["clear"][k - 1] == a + ph for k in range(K + 1, n))
assert all(L["offload"][k] - L["offload"][k - 1] == a + ref for k in range(1, n))
assert max(L["clear"]) > T                                    # 300 步时清理一项不够用

# 3) 摘要：锯齿形，峰值不超过 T，周期 floor((T-P-U-s)/g) + 1 = 32 步
assert max(L["summary"]) <= T and (T - P - U - s) // g + 1 == 32 and R["summary"][1] == 9
assert R["clear+summary"][1] == 1                             # 先清理，摘要从 9 次降到 1 次

# 4) 不算缓存时，清理把累计输入降到约 15%
assert 0.14 < S["clear"] / S["none"] < 0.16

# 5) 算上缓存，结论会变：逐步清理每步都改写第 k−K 步那一段，它之后的内容每步都要重写缓存
assert S["clear"] < S["clear_batch"]          # 按 token 数，逐步清理更省
assert C["clear_batch"] < C["clear"]          # 按带缓存的成本，批量清理更省
# 只追加的两种（外置、摘要）带缓存最便宜；每步都改历史的（逐步清理）最贵
assert C["offload"] < C["summary"] < C["clear_batch"] < C["clear"] < C["none"]
assert C["summary"] < C["clear+summary"]      # 叠加逐步清理后，缓存账反而变差
print(f"带缓存：逐步清理 / 批量清理 = {C['clear'] / C['clear_batch']:.2f} 倍")
assert 1.2 < C["clear"] / C["clear_batch"] < 1.35

```

输出（截取）：

| 策略 | 第 80 步上下文 | 300 步内峰值 | 累计输入 | 带缓存成本（基础价 × token） | 摘要次数 |
|---|---|---|---|---|---|
| none | 229,700 | 845,700 | 128,130,000 | 13,785,555 | 0 |
| clear | 41,220 | 111,620 | 19,119,120 | 4,797,515 | 0 |
| clear_batch | 80,900 | 153,300 | 26,053,200 | 3,766,245 | 0 |
| offload | 36,940 | 116,140 | 18,696,000 | 2,003,161 | 0 |
| summary | 51,700 | 99,300 | 16,360,800 | 2,621,055 | 9 |
| clear+summary | 41,220 | 99,780 | 16,138,320 | 4,474,963 | 1 |

从这张表能读出四件事：

1. **清理和外置只降斜率**。斜率从 2,800 降到 320 或 360，曲线仍是直线往上。300 步时清理一项已经到了 11 万，再长的任务迟早要摘要或重置。
2. **摘要把曲线变成锯齿**。长度卡在 T 以下，周期是 ⌊(T − P − U − s)/g⌋ + 1 = 32 步。先清理再摘要，300 步里只摘要 1 次，有损操作的次数从 9 次降到 1 次。
3. **只数 token，逐步清理比批量清理省；算上缓存，结论反过来**。逐步清理每步都改写第 k − K 步那个工具结果，它后面的内容每步都要按写价重写缓存。批量清理每 20 步才改写一次，累计 token 多了 36%，带缓存成本却只有逐步清理的 1/1.27。外置和摘要两种在两次改写之间都只追加，带缓存最便宜。叠加了逐步清理的 `clear+summary` 累计 token 最少，带缓存成本却比单做摘要高 70%。
4. **压缩策略要同时看两本账**。只看上下文长度选出来的方案，上线后账单可能更高。第 5.4 节把“一次清理多久回本”写成公式。

这里的模型做了简化：每步 a、r 固定；摘要本身那次模型调用的成本没算；缓存不过期。结论的方向不依赖这些简化，具体数字要用你自己的 a、r 分布重算。

### 3.4 压缩还是重置，阈值怎么定

**压缩和重置怎么选**。Anthropic 的 Harness 设计文章说：压缩保留了连续性，但没给 Agent 一张白纸。有的模型接近上下文上限时会提前收工（context anxiety），压缩解决不了，要靠重置加结构化交接。作者在较早的模型上用重置解决了这个问题，换到更新的模型后去掉了重置，改用 SDK 的自动压缩。哪种合适取决于模型，要测。

可以按任务的形状粗分：

| 情况 | 选择 | 理由 |
|---|---|---|
| 同一阶段、后面还要用到前面的细节 | 摘要，保留最近几轮原文 | 连续性重要 |
| 进入新阶段，前一阶段的过程不再需要 | 重置，写交接文件 | 交接文件比摘要更可检查 |
| 模型在接近上限时提前收工 | 重置 | 摘要后上下文仍然“很满” |
| 工具结果占大头 | 先清理或外置 | 能找回，不必动摘要 |

**阈值**：设软、硬两个。软阈值触发摘要；硬阈值触发重置，并给本步的输出和工具结果留出余量。阈值按评测里质量开始下降的拐点定，不按窗口百分比定。换一个更大窗口的模型，拐点不一定跟着后移。

阈值也不是越低越好。第 3.3 节的表说明，摘要和清理每做一次都要重写缓存。阈值太低，压缩次数变多，省下的 token 可能被缓存写入吃掉。

## 4 Event Log、Checkpoint 与 Resume

### 4.1 问题：上下文不能是唯一的状态

压缩会丢信息，进程会崩。如果只有上下文这一份状态，两件事都无法挽回：压缩丢了的找不回来；崩溃后不知道做到了哪一步，也不知道最后那次写操作有没有生效。

解决办法来自事件溯源（Event Sourcing）。Martin Fowler 的定义是：把应用状态的所有变化记成一串事件。有了这串事件，就能把当前状态扔掉，在一个空的应用上重跑日志，重建出来（complete rebuild）；也能重跑到任意一个时间点，得到当时的状态（temporal query）。

搬到 Agent 上：

| 概念 | 是什么 | 特点 |
|---|---|---|
| Event Log | 用户、模型、工具、审批、系统事件按顺序追加的记录 | 只追加不修改，是事实来源 |
| 工作状态 | 对日志做折叠（reduce）得到的结构化状态：目标、步骤号、已完成的副作用、进行中的调用 | 是日志的函数，可以随时重算 |
| Checkpoint | 某个日志位置上的工作状态快照 | 只为加速，和全量折叠的结果必须一致 |
| Resume | 最近的 Checkpoint 加上之后的日志，折叠出工作状态，再继续 | 继续之前处理结果未知的调用 |
| Context | 下一次模型调用实际看到的内容 | 从日志和工作状态装配出来的视图 |

写成式子：

```text
state_t  = fold(apply, EMPTY, log[0:t])
         = fold(apply, snapshot[j], log[j:t])        （j ≤ t 是最近的 Checkpoint 位置）
context_t = view(log[0:t], state_t, 压缩策略)
```

关键一点：**上下文是视图，日志是事实**。压缩只换了 `view` 的实现，不碰日志，所以压缩不再是永久丢失。Anthropic 的 Managed Agents 文章就是这么做的：会话是“发生过的一切的只追加日志”，放在上下文窗口之外；Harness 用 `emitEvent(id, event)` 写事件，用 `getEvents()` 按位置取日志的片段；Harness 崩了，新实例用 `wake(sessionId)` 启动，`getSession(id)` 取回日志，从最后一个事件继续。文章还写明，取回的事件可以在 Harness 里变换后再交给模型，例如为了提高缓存命中率而重新组织。

反过来也要记住：日志完整，不代表能精确重放模型当时的行为。采样、推理侧数值差异、外部数据变化都会让重跑的轨迹分叉。回放与可复现的细节见 [06 Trace 与回放](../06-tracing-replay/README.md)。

### 4.2 重放和副作用：为什么恢复时不能重新执行

Fowler 在同一篇文章里指出了重放的陷阱：如果事件会触发发往外部系统的消息，重放时就会出错，因为外部系统分不清真实处理和重放。他的办法是把外部系统包一层网关（Gateway），重建状态时关掉网关，由网关决定是否真的发出去，领域逻辑本身不变。

Agent 的恢复有同样的结构，还多了两个麻烦：

1. **模型输出不确定**。重新调用模型，第 7 步可能选了另一个工具。所以模型输出本身也是事件，要写进日志。恢复时如果日志里已经有第 7 步的模型输出，直接用它，不重新采样。
2. **外部调用可能做了一半**。进程可以死在三个位置：

| 崩溃位置 | 日志里有什么 | 外部世界 | 恢复时怎么做 |
|---|---|---|---|
| 写调用意图之前 | 只有模型输出 | 没动 | 按模型输出正常执行这一步 |
| 写了意图，调外部之前 | 意图 | 没动 | 结果未知：先查，查不到再带同一个幂等键执行 |
| 外部生效后，结果写回之前 | 意图 | 已生效 | 结果未知：先查，查到就补记结果，不再执行 |

第二、三行在日志里看起来完全一样，Harness 分不出来。所以凡是“有意图没结果”的调用，都按结果未知处理：

- 下游支持幂等键时，带同一个键重做，下游识别为同一次。幂等键用 Task ID 加步骤编号生成，它在恢复前后不变；用随机 UUID 或时间戳，恢复后会生成新键，幂等就失效了。幂等键的服务端语义见 [04 工具编排](../04-tool-orchestration/README.md)。
- 下游不支持幂等键，但能按客户端 token 查询时，先查再决定。
- 两者都没有，交给人确认，不盲目重做。

读操作可以直接重做，但结果可能和第一次不同，要以新结果为准写进日志。

**LangGraph 里的对应物**。LangGraph 的 checkpointer 按 `thread_id` 存图状态的快照，用于人工介入、回到历史状态和故障恢复。几条和恢复直接相关的规则（以当前文档为准）：

- 同一个 super-step 里有节点失败时，其他已经成功的节点的写入会作为 pending writes 存下，恢复时不再重跑它们。
- 持久化有三种模式：`exit` 只在图执行退出时持久化，中途崩溃无法恢复；`async` 在下一步执行时异步写，崩溃时有小概率没写上；`sync` 在下一步开始前同步写完。
- 从 `interrupt()` 恢复时，节点从头重新执行，不是从 `interrupt()` 那一行接着跑。文档因此要求 `interrupt()` 之前的副作用要幂等，或者放到 `interrupt()` 之后，或者拆成单独的节点；长时间暂停后，执行写操作前要重新读外部状态。
- 内存版 saver 进程重启就丢，生产要用数据库版本；长对话的快照会不断累积，要设保留策略。

Claude Agent SDK 按 `session_id` 恢复或分叉会话，可以把会话轨迹同步到自己的存储，让另一台主机恢复。

### 4.3 实现：事件日志、重建与崩溃恢复

下面的模拟有一个四步的编程任务：读两个文件、创建合并请求、在合并请求上发一条评论。`create_pr` 支持幂等键；`post_comment` 不去重，只能按 token 查询。代码做四件事：

1. `apply` 是纯函数，工作状态只由日志折叠得到；每 4 个事件打一次 Checkpoint。
2. `Faults` 在每个“边界”计数：写日志前、调外部前、外部生效后。依次在每个边界杀一次进程，再恢复。
3. 每次恢复后 assert：工作状态、日志、装配出的上下文都和不崩溃的参照运行一致；合并请求和评论各恰好一个；重建状态的过程不调外部。
4. 反例：恢复时不先查询就重做 `post_comment`，评论发了两次。

```python
import copy
import json


class Crash(Exception):
    pass


class Faults:
    """在第 crash_at 个“边界”处抛出 Crash，模拟进程被杀。边界见 append 和 call。"""
    def __init__(self, crash_at=None):
        self.n, self.crash_at = 0, crash_at

    def tick(self):
        self.n += 1
        if self.n == self.crash_at:
            raise Crash(self.n)


class Disk:
    """持久化存储：崩溃后还在。log 只追加；snapshots 是 Checkpoint，键是它覆盖到的日志长度。"""
    def __init__(self):
        self.log, self.snapshots = [], {}


class World:
    """外部系统。create_pr 支持幂等键；post_comment 不去重，但能按客户端 token 查询。"""
    def __init__(self):
        self.prs, self.comments, self.by_key, self.calls = {}, [], {}, 0

    def create_pr(self, key, title):
        self.calls += 1
        if key not in self.by_key:
            pid = f"PR-{len(self.prs) + 1}"
            self.prs[pid], self.by_key[key] = title, pid
        return self.by_key[key]

    def post_comment(self, key, pr, text):
        self.calls += 1
        cid = f"C-{len(self.comments) + 1}"
        self.comments.append((key, pr, text, cid))
        return cid

    def query(self, key):
        if key in self.by_key:
            return self.by_key[key]
        return next((c[3] for c in self.comments if c[0] == key), None)


FILES = {"spec.md": "需求：给 /health 加超时", "main.py": "def health(): ..."}
SIDE_EFFECT = {"create_pr", "post_comment"}
TASK = "task-42"


def apply(state, ev):
    """纯函数 reducer：状态只由日志折叠得到，不碰外部世界。"""
    s = copy.deepcopy(state)
    t = ev["type"]
    if t == "user":
        s["goal"] = ev["text"]
    elif t == "model":
        s["pending"] = ev["action"]           # 模型已经决定、还没执行的动作
    elif t == "intent":
        s["pending"] = None
        s["inflight"][ev["key"]] = {"tool": ev["tool"], "args": ev["args"]}
    elif t == "result":
        it = s["inflight"].pop(ev["key"])
        if it["tool"] == "read_file":
            s["read"].append(it["args"]["path"])
        elif it["tool"] == "create_pr":
            s["pr"] = ev["result"]
        elif it["tool"] == "post_comment":
            s["comment"] = ev["result"]
        s["step"] += 1
    elif t == "finish":
        s["pending"], s["done"] = None, True
    return s                                   # "resumed" 不改工作状态


EMPTY = {"goal": None, "step": 0, "read": [], "pr": None, "comment": None,
         "inflight": {}, "pending": None, "done": False}


def fold(events, state=EMPTY):
    for e in events:
        state = apply(state, e)
    return state


def rebuild(disk):
    """Resume 的第一步：最近的 Checkpoint + 之后的日志。"""
    seq = max([q for q in disk.snapshots if q <= len(disk.log)], default=0)
    base = disk.snapshots.get(seq, EMPTY)
    return fold([json.loads(x) for x in disk.log[seq:]], base)


def decide(state):
    """模型的桩：按工作状态决定下一步。真实模型不确定，所以它的输出也要进日志。"""
    plan = [("read_file", {"path": "spec.md"}), ("read_file", {"path": "main.py"}),
            ("create_pr", {"title": "health 超时"}),
            ("post_comment", {"pr": state["pr"], "text": "已加 5s 超时"}),
            ("finish", {})]
    tool, args = plan[state["step"]]
    return {"tool": tool, "args": args}


def run(disk, world, faults, query_before_redo=True):
    def append(ev):
        faults.tick()                          # 边界 1：写日志之前崩
        disk.log.append(json.dumps(ev, ensure_ascii=False))
        if len(disk.log) % 4 == 0:             # 每 4 个事件打一次 Checkpoint
            disk.snapshots[len(disk.log)] = rebuild(disk)

    def call(key, it):
        faults.tick()                          # 边界 2：已写意图、还没调外部
        if it["tool"] == "read_file":
            r = FILES[it["args"]["path"]]
        else:
            r = getattr(world, it["tool"])(key, **it["args"])
        faults.tick()                          # 边界 3：外部已生效、结果还没记
        return r

    state = rebuild(disk)
    if disk.log:
        append({"type": "resumed"})
    for key, it in list(state["inflight"].items()):     # 结果未知的调用
        r = world.query(key) if (it["tool"] in SIDE_EFFECT and query_before_redo) else None
        if r is None:
            r = call(key, it)                   # 没生效才重做，带同一个幂等键
        ev = {"type": "result", "key": key, "result": r}
        append(ev)
        state = apply(state, ev)
    if state["goal"] is None:
        ev = {"type": "user", "text": "给 /health 加超时并提 PR"}
        append(ev)
        state = apply(state, ev)
    while not state["done"]:
        action = state["pending"]               # 日志里已有模型输出：直接用，不重新采样
        if action is None:
            action = decide(state)
            ev = {"type": "model", "step": state["step"], "action": action}
            append(ev)
            state = apply(state, ev)
        if action["tool"] == "finish":
            append({"type": "finish"})
            state = apply(state, {"type": "finish"})
            break
        key = f"{TASK}:{state['step']}"         # 幂等键 = Task ID + 步骤编号
        ev = {"type": "intent", "key": key, **action}
        append(ev)
        state = apply(state, ev)
        ev = {"type": "result", "key": key, "result": call(key, action)}
        append(ev)
        state = apply(state, ev)
    return state


def build_context(log, keep_results=None):
    """上下文是日志的视图。keep_results=K 时只展开最近 K 个工具结果，其余换成占位符。"""
    evs = [json.loads(x) for x in log]
    res_idx = [i for i, e in enumerate(evs) if e["type"] == "result"]
    keep = set(res_idx if keep_results is None else res_idx[-keep_results:])
    out = []
    for i, e in enumerate(evs):
        if e["type"] == "user":
            out.append(f"user: {e['text']}")
        elif e["type"] == "intent":
            out.append(f"call {e['tool']}({e['args']})")
        elif e["type"] == "result":
            out.append(f"result: {e['result']}" if i in keep else "result: [已清理，原文在日志]")
    return out


strip = lambda log: [x for x in log if json.loads(x)["type"] != "resumed"]

# 参照：不崩溃跑一遍
ref_disk, ref_world, f0 = Disk(), World(), Faults()
ref_state = run(ref_disk, ref_world, f0)
N = f0.n                                        # 一次完整运行经过的边界数
assert ref_state["done"] and ref_state["pr"] == "PR-1" and len(ref_world.comments) == 1

# 在每一个边界崩一次，再恢复
for k in range(1, N + 1):
    disk, world = Disk(), World()
    try:
        run(disk, world, Faults(k))
        raise AssertionError("应该崩溃")
    except Crash:
        pass
    calls_before = world.calls
    rebuild(disk)
    assert world.calls == calls_before          # 重建状态只读日志，不调外部
    state = run(disk, world, Faults())          # Resume
    assert state == ref_state                   # 工作状态一致
    assert strip(disk.log) == ref_disk.log      # 事件日志逐条一致（除恢复标记）
    assert world.prs == ref_world.prs and len(world.comments) == 1   # 副作用恰好一次
    assert build_context(disk.log) == build_context(ref_disk.log)    # 装配出的上下文一致
    assert rebuild(disk) == fold([json.loads(x) for x in disk.log])  # Checkpoint 和全量折叠一致
print(f"{N} 个崩溃点全部恢复一致")

# 反例：恢复时不先查询，直接重做结果未知的 post_comment
def first_crash_after_comment():
    for k in range(1, N + 1):
        d, w = Disk(), World()
        try:
            run(d, w, Faults(k))
        except Crash:
            st = rebuild(d)
            if len(w.comments) == 1 and any(i["tool"] == "post_comment" for i in st["inflight"].values()):
                return k
k = first_crash_after_comment()
d, w = Disk(), World()
try:
    run(d, w, Faults(k))
except Crash:
    pass
run(d, w, Faults(), query_before_redo=False)
assert len(w.comments) == 2                     # 评论发了两次：这就是不查就重做的后果

# 压缩只改视图：清理旧工具结果后上下文变短，日志一条没少
full, cleared = build_context(ref_disk.log), build_context(ref_disk.log, keep_results=1)
assert len(" ".join(cleared)) < len(" ".join(full))
assert FILES["spec.md"] in " ".join(full) and FILES["spec.md"] not in " ".join(cleared)
assert len(ref_disk.log) == len(strip(ref_disk.log))   # 压缩不写日志、不删日志
```

输出是“23 个崩溃点全部恢复一致”。这段代码里有几处设计，每一处去掉都会让某个 assert 失败：

- **模型输出进日志，恢复时不重新采样**。如果不把 `model` 事件折叠进状态，崩在“模型输出已写、意图未写”时，恢复后会再调一次模型，日志里多出一条 `model` 事件，“日志一致”的 assert 在 23 个崩溃点里有 5 个失败。真实模型每次可能选不同的动作，这个问题只会更严重。
- **先写意图，再调外部**。意图是“结果未知”的唯一证据。先调外部再写日志，崩在中间时日志里什么都没有，恢复后会当作没做过，直接再执行一遍。这和数据库的预写日志（WAL）是同一个道理。
- **结果未知时先查**。反例那段把 `query_before_redo` 关掉，`post_comment` 就发了两次。`create_pr` 不受影响，因为它带幂等键，重做也只生效一次。
- **Checkpoint 只是缓存**。每次都 assert 快照加尾部日志的折叠结果，和从头折叠的结果相同。快照和日志不一致时，以日志为准。

### 4.4 恢复流程

把上面的代码抽成步骤：

```text
resume(task_id):
    state = load_checkpoint(task_id)            # 最近的快照
    for e in log_after(state.checkpoint_seq):   # 只折叠日志，不重新执行任何调用
        state = apply(state, e)
    append(log, "resumed")                      # 恢复本身也是事件，方便审计
    for call in state.inflight:                 # 有意图、没结果
        if call.has_side_effect:
            r = query_external(call.idempotency_key)
            if r is None and supports_idempotency(call):
                r = execute(call)               # 同一个幂等键，重复也只生效一次
            elif r is None:
                r = ask_human(call)             # 查不清就交给人
        else:
            r = execute(call)                   # 读操作直接重做，以新结果为准
        append(log, result(call, r))
    verify_environment(state)                   # 工作区、依赖、模型和 prompt 版本是否还在
    if state.pending_model_output:              # 模型已决定、还没执行
        continue_from(state.pending_model_output)
    continue_loop(state)
```

`verify_environment` 要查的东西：工作区和依赖还在不在；下游服务是否可用；prompt 和模型版本有没有升级，升级了要不要按新版本继续；长时间暂停后外部数据（价格、库存、权限）是否过期。

### 4.5 长任务的跨会话交接

进程崩溃是被动中断；任务太长、要分多个会话做，是主动中断。Anthropic 的长任务文章用三样东西把多个会话接起来：一个进度文件、git 提交历史、一份 JSON 功能清单。每个会话开始时先确认工作目录，读进度文件和提交记录，挑一个未完成的功能，先跑一遍基本的端到端检查再写新代码；结束时提交代码、更新进度文件，留下可以合并的干净状态。他们选 JSON 而不是 Markdown 存清单，因为模型更少不当地改写 JSON 文件。

这套做法和事件日志是同一个思路：git 历史是只追加的事实记录；功能清单是结构化的工作状态；每个新会话的上下文从这两者装配出来，不依赖上一个会话的上下文。

## 5 稳定前缀与 KV Cache 命中

### 5.1 原理：只有从头起完全相同的部分能复用

推理时每个位置的 KV 由它前面所有 token 算出。两个请求开头完全一样，后一个可以复用前一个算好的部分。前面任何一个 token 变了，后面全部要重算。原理见 [inference/01 KV Cache 与显存](../../inference/01-kv-cache-memory/README.md)，推理引擎侧的前缀缓存和 RadixAttention 见 [inference/05 面向 Agent 的推理](../../inference/05-agent-serving/README.md)。

Manus 的文章把 KV Cache 命中率称为生产阶段 Agent 最重要的一个指标，因为它同时决定延迟和成本；文中以 Claude Sonnet 为例，缓存输入价是未缓存的十分之一。Agent 的输入远多于输出（第 2.2 节的 100:1），输入里能命中多少，基本就是账单的大头。

### 5.2 两种实现的规则

| | Anthropic API 的 prompt caching | vLLM 自动前缀缓存 |
|---|---|---|
| 缓存单位 | 到 `cache_control` 断点为止的前缀，最多 4 个断点 | 固定大小的 KV 块，按“前缀哈希 + 本块 token”算哈希 |
| 前缀顺序 | `tools` → `system` → `messages`，某一层变了，这层和之后都失效 | 按 token 序列从头匹配 |
| 怎么找到旧条目 | 从断点往回逐块查找以前写过的条目，最多查 20 个位置 | 从第一块起逐块查哈希，第一次不命中就停 |
| 不缓存的情况 | 短于模型最小可缓存长度的前缀（按模型从 512 到 4,096 token 不等），不报错，直接不缓存 | 不满的最后一块不缓存 |
| 过期 | 默认 5 分钟，被使用时免费刷新；可选 1 小时，写入价更高 | 引用数为 0 的块进空闲队列，按 LRU 淘汰 |
| 怎么看命中 | 响应里的 `cache_read_input_tokens` 和 `cache_creation_input_tokens` | 引擎的命中指标 |

价格以 Anthropic 文档为例：5 分钟缓存的写入是基础输入价的 1.25 倍，1 小时是 2 倍；读取是 0.1 倍，部分较新的模型读取倍率更低（文档脚注列出 0.05 倍和 0.025 倍）。以上倍率查阅于 2026-11，具体模型以当时的定价表为准。

两条容易漏的细节：

- **`input_tokens` 不是全部输入**。文档写明，总输入 = `cache_read_input_tokens` + `cache_creation_input_tokens` + `input_tokens`，其中 `input_tokens` 只是最后一个断点之后的部分。只看 `input_tokens` 统计用量会严重低估。成本统计口径见 [06 Trace 与回放](../06-tracing-replay/README.md)。
- **20 个位置的回看窗口**。断点每次往回最多查 20 个块。对话一次增长 20 个块以上（例如一轮里多段文本和图片交替），上一次写入的条目会落在窗口外，命中为 0。文档的建议是在更靠前的位置预先加第二个断点。

### 5.3 实现：三种操作对命中率的影响

下面按 vLLM 的规则模拟（块大小 16，只缓存满块，块哈希串着前一块的哈希）。一个 40 步的会话：12 个工具定义共 1,800 token，系统提示 1,000 token，每步追加用户消息、模型输出和工具结果。比较六种情况：

- `baseline`：只追加；
- `ts_at_end`：每轮把当前时间写进最新的用户消息；
- `ts_in_system`：每轮把当前时间写在系统提示开头；
- `tool_shuffle`：每轮按“相关度”重排工具；
- `tool_added`：第 20 步追加一个工具；
- `compact`：每 10 步把历史换成一段摘要。

```python
import random

B = 16                                   # 块大小（vLLM 风格：只缓存写满的块，块哈希串着前一块）
N_TOOLS, TOOL_LEN, SYS_LEN = 12, 150, 1_000
A, R, PH, STEPS = 300, 400, 20, 40       # 每步模型输出、工具结果、占位符长度、步数


def seg(tag, n):
    return [f"{tag}.{i}" for i in range(n)]


TOOLS = [seg(f"tool{t}", TOOL_LEN) for t in range(N_TOOLS)]
SYSTEM = seg("sys", SYS_LEN)


class PrefixCache:
    def __init__(self):
        self.blocks = set()

    def lookup_and_insert(self, toks):
        """返回命中的 token 数（从开头起连续命中的满块），再把本请求的满块写进缓存。"""
        h, hit, missed = 0, 0, False
        for i in range(0, len(toks) // B * B, B):
            h = hash((h, tuple(toks[i:i + B])))
            if not missed and h in self.blocks:
                hit += B
            else:
                missed = True
                self.blocks.add(h)
        return hit


def session(scenario, seed=0):
    """模拟一个 Agent 会话，返回每次请求的 (总长度, 命中长度)。"""
    rng, cache, hist, out = random.Random(seed), PrefixCache(), [], []
    tools = list(TOOLS)
    for k in range(STEPS):
        if scenario == "tool_shuffle":
            rng.shuffle(tools)                               # 每次按“相关度”重排工具
        if scenario == "tool_added" and k == STEPS // 2:
            tools = tools + [seg("tool_new", TOOL_LEN)]      # 中途追加一个工具
        if scenario == "compact" and k > 0 and k % 10 == 0:
            hist = [seg(f"summary{k}", 800)]                 # 历史整体换成摘要
        system = SYSTEM
        if scenario == "ts_in_system":
            system = seg(f"now{k}", 8) + SYSTEM              # 系统提示开头写“当前时间”
        new = seg(f"user{k}", 40)
        if scenario == "ts_at_end":
            new = new + seg(f"now{k}", 8)                    # 时间放进本轮的最新消息
        hist.append(new)
        toks = [t for x in tools for t in x] + system + [t for x in hist for t in x]
        out.append((len(toks), cache.lookup_and_insert(toks)))
        hist.append(seg(f"model{k}", A))                     # 模型输出和工具结果追加到历史
        hist.append(seg(f"result{k}", R))
    return out


def hit_rate(out):
    return sum(h for _, h in out) / sum(L for L, _ in out)


def cost_ratio(out, write=1.25, read=0.1):
    """缓存断点放在末尾时的输入成本 / 不开缓存的成本：命中部分按读价，其余按写价。"""
    return sum(read * h + write * (L - h) for L, h in out) / sum(L for L, _ in out)


res = {s: session(s) for s in
       ["baseline", "ts_at_end", "ts_in_system", "tool_shuffle", "tool_added", "compact"]}
for s, out in res.items():
    print(f"{s:13s} 命中率 {hit_rate(out):6.1%}  成本/不开缓存 {cost_ratio(out):.2f}")

base = res["baseline"]
# 1) 只追加时，第 k 次请求命中的正好是上一请求长度向下取整到块
assert all(h == prev // B * B for (_, h), (prev, _) in zip(base[1:], base[:-1]))
assert hit_rate(base) > 0.9
# 2) 时间戳放末尾几乎不影响；放在系统提示开头，只剩工具定义那段能命中
assert hit_rate(res["ts_at_end"]) > 0.9
tools_len = N_TOOLS * TOOL_LEN
assert all(h == tools_len // B * B for _, h in res["ts_in_system"][1:])
# 命中率从 >90% 掉到个位数，成本反而比不开缓存还高：每次都按 1.25 倍写入、几乎读不到
assert hit_rate(res["ts_in_system"]) < 0.15 and cost_ratio(res["ts_in_system"]) > 1.0
# 3) 工具每次重排：工具在最前面，整条前缀几乎全部失效
assert hit_rate(res["tool_shuffle"]) < 0.05
# 4) 中途追加工具：只有那一次请求掉到“旧工具定义”那一段，之后恢复
add = res["tool_added"]
k = STEPS // 2
assert add[k][1] == tools_len // B * B and add[k + 1][1] == add[k][0] // B * B
# 5) 压缩：压缩那一次只命中工具 + 系统提示，历史部分要重写；之后又按只追加的规律命中
comp = res["compact"]
fixed = (tools_len + SYS_LEN) // B * B
assert all(comp[j][1] == fixed for j in (10, 20, 30))
assert hit_rate(base) > hit_rate(comp)
```

输出：

| 情况 | 命中率 | 输入成本 / 不开缓存 |
|---|---|---|
| baseline | 95.4% | 0.15 |
| ts_at_end | 95.4% | 0.15 |
| ts_in_system | 10.1% | 1.13 |
| tool_shuffle | 0.7% | 1.24 |
| tool_added | 93.2% | 0.18 |
| compact | 88.1% | 0.24 |

读法：

- **时间戳放哪里差别巨大**。同样 8 个 token，放在最新消息里不影响命中；放在系统提示开头，只剩前面工具定义那一段能命中。成本比不开缓存还高 13%，因为每次都按 1.25 倍写入，几乎读不到。Manus 的文章专门点了这个错：在系统提示开头写一个精确到秒的时间戳，模型能告诉你现在几点，缓存命中率也没了。
- **工具重排最伤**。工具定义在最前面，顺序一变，整条前缀全部失效。Anthropic 文档的失效表也写明，修改工具定义会让 tools、system、messages 三层缓存全部失效。
- **中途加工具只伤一次**。追加的那次请求只命中旧工具定义那一段，下一次请求又按只追加的规律命中。Manus 的做法更进一步：任务中途尽量不增删工具，用解码时屏蔽 token logits 的方式控制哪些工具可选。他们给的另一个理由是，历史里引用了已经不在定义里的工具，模型会困惑，没有约束解码时容易产生不合法的调用。工具快照和按需加载见 [04 工具编排](../04-tool-orchestration/README.md)。
- **压缩的代价是可计算的**。每次压缩后的那次请求只能命中工具和系统提示，历史部分要重写，之后恢复正常。

### 5.4 压缩什么时候回本

压缩改写了历史，压缩点之后的缓存全部失效，下一次调用要重新写缓存。用一次清理来算：上下文长度 C，清理点之前的稳定前缀 p，清掉 X 个 token。

```text
不清理，下一次请求：读 C              → ρ·C
清理，下一次请求：  读 p，写 C − X − p → ρ·p + w·(C − X − p)
清理那次多付：      Δ = w·(C − X − p) − ρ·(C − p)
之后每次请求少读 X：  省 ρ·X

回本需要的请求数：   m* = Δ / (ρ·X)
```

这个推导告诉你两件事。X 越小越不划算：清 5K 和清 50K 的写入代价差不多，省下的却差十倍。ρ 越小越难回本：读价从 0.1 降到 0.05，缓存更值钱，回本的请求数翻一倍多。Claude API 的 `clear_at_least` 参数就是为此设计的：文档写明，如果一次清不掉指定的量，这次就不清，用它判断清理值不值得打破缓存。

```python
# 清理一次值不值：上下文 C，清理点之前的稳定前缀 p，清掉 X 个 token
# 清理那次请求多付 w(C−X−p) − ρ(C−p)；之后每次请求少读 X，省 ρ·X
def breakeven(C, p, X, w=1.25, rho=0.1):
    return (w * (C - X - p) - rho * (C - p)) / (rho * X)


def cost(reqs, w=1.25, rho=0.1):
    """reqs：每次请求的片段列表。和上一请求相同的前缀按读价，其余按写价。"""
    total, prev = 0.0, []
    for ctx in reqs:
        hit = 0
        for x, y in zip(ctx, prev):
            if x != y:
                break
            hit += x[1]
        total += rho * hit + w * (sum(n for _, n in ctx) - hit)
        prev = ctx
    return total


def recover_at(C, p, X, g=2_800, w=1.25, rho=0.1):
    """逐请求模拟：第 1 次请求做清理，返回第几次请求起累计成本不再高于不清理。"""
    before = [("p", p), ("X", X), ("rest", C - X - p)]
    after = [("p", p), ("rest", C - X - p)]
    keep, clear = [before], [before]                    # 第 0 次：缓存已经是热的
    for k in range(1, 2_000):
        tail = [(f"new{j}", g) for j in range(1, k + 1)]
        keep.append(before + tail)
        clear.append(after + tail)
        if cost(clear, w, rho) <= cost(keep, w, rho):
            return k


be = breakeven(100_000, 10_000, 50_000)
print(f"清掉 50K：清理后还要 {be:.1f} 次请求回本")
assert abs(be - 8.2) < 1e-9
assert recover_at(100_000, 10_000, 50_000) == 1 + 9     # 清理那次 + ceil(8.2) 次
assert abs(breakeven(100_000, 10_000, 5_000) - 194.5) < 1e-9
assert recover_at(100_000, 10_000, 5_000) == 1 + 195    # 只清 5K：两百次请求才回本
# 读价更低（0.05 倍）时，缓存更值钱，清理更难回本
be05 = breakeven(100_000, 10_000, 50_000, rho=0.05)
assert abs(be05 - 18.2) < 1e-9 and recover_at(100_000, 10_000, 50_000, rho=0.05) == 1 + 19
```

代入 C = 100K、p = 10K：清掉 50K，清理那次多付 41,000（以基础输入价 × token 计），之后每次省 5,000，8.2 次请求回本；只清 5K，要 194.5 次。逐请求模拟的结果是第 10 次和第 196 次请求起累计成本不再更高，多出的那 1 次是清理请求本身。

摘要也一样，只是 X 换成“被摘要替换掉的长度减去摘要长度”。Claude API 的阈值压缩文档给了一个减小 Δ 的办法：在系统提示末尾单独放一个 `cache_control` 断点。这样压缩时系统提示的缓存仍然有效，只有摘要要作为新条目写入；不加这个断点，系统提示要和摘要一起重写。

### 5.5 设计约束和排查

由此得出的设计约束：

| 约束 | 常见的破坏方式 |
|---|---|
| 系统提示里不放每次都变的内容 | 开头写“当前时间：…”、请求 ID、用户名 |
| 工具定义的顺序和序列化固定 | 按相关度给每个请求重排工具；JSON 键顺序不固定 |
| 动态内容放在稳定部分之后 | 把检索结果插在系统提示中间 |
| 历史只追加 | 每步改写早期消息的格式；逐步清理旧工具结果 |
| 压缩成批做 | 阈值太低，或者每步清一点 |
| 改参数也算改前缀 | 换 `tool_choice`、加减图片、改 thinking 配置，都会让 messages 层失效（Anthropic 失效表） |

序列化要特别小心。Manus 的文章提醒：很多语言和库不保证 JSON 对象的键顺序稳定，这会悄悄打破缓存。Python 里 `json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))` 这类固定写法，比依赖字典插入顺序可靠。

**命中率低先查什么**：比对相邻两个请求，找第一个不同的 token 在哪。常见原因是时间戳、无序序列化、工具顺序变化、参数变化，以及请求间隔超过了缓存有效期。按请求记下 `cache_read_input_tokens / 总输入`，看它从哪一次开始掉。

## 6 压缩对正确性的影响怎么评测

### 6.1 四种方法

只看最终通过率不够。通过率可能几乎不变，但步数和重复调用变多，说明丢了“已经试过什么”。

| 方法 | 做法 | 发现什么问题 |
|---|---|---|
| 关键事实保留 | 从压缩前抽出目标、约束、已得到的结论、失败过的路径，对压缩后的上下文提同样的问题 | 摘要漏掉了硬约束或失败记录 |
| 下一步行为对比 | 同一个断点，分别用压缩前和压缩后的上下文让模型选下一步，比较动作是否一致 | 压缩改变了决策 |
| 端到端对比 | 评测集上开、关压缩各跑多次，比通过率、步数、重复调用次数、成本 | 整体是否划算 |
| 上线监控 | 统计每次压缩后的重复调用和步数变化 | 线上分布和评测集不同 |

几个做法上的要求：

- 关键事实先用规则抽结构化字段（约束、ID、数字、失败的工具调用），再人工抽查一部分。用模型抽要人工校准。
- 下一步动作不一致不一定是坏事，可能换了一条等价的路。要看最终结果和步数，再人工看分歧样本。
- 构造评测要用和线上相似的长轨迹。用不相关内容填出来的长上下文，测出的结论偏乐观。
- 同一题跑多次看分布，统计方法见 [core/09 评测](../../../core/09-evaluation/README.md)。

### 6.2 实现：通过率看不出的退化

下面的模拟把第 3.2 节的第一行变成数字。任务是在 12 个候选方案里找到唯一能通过测试的那个。策略桩只会避开上下文里写着“失败过”的候选，每 4 步压缩一次。好的摘要保留失败记录，坏的摘要只留“目标 / 已完成”。

```python
import random

# 压缩对正确性的影响：一个摘要漏掉“失败过的路径”，通过率几乎不变，步数和重复调用明显变多
N_CAND, MAX_STEPS, EVERY = 12, 40, 4     # 候选方案数、步数上限、每 4 步压缩一次


def trial(keep_failures, seed):
    rng = random.Random(seed)
    answer = rng.randrange(N_CAND)           # 只有一个候选能通过测试
    ctx_failed, tried, repeats = set(), [], 0
    for step in range(1, MAX_STEPS + 1):
        options = [c for c in range(N_CAND) if c not in ctx_failed]
        c = rng.choice(options)              # 策略桩：只避开上下文里写着失败过的候选
        repeats += c in tried
        tried.append(c)
        if c == answer:
            return True, step, repeats, ctx_failed
        ctx_failed.add(c)
        if step % EVERY == 0 and not keep_failures:
            ctx_failed = set()               # 摘要只留“目标 / 已完成”，漏掉失败记录
    return False, MAX_STEPS, repeats, ctx_failed


def evaluate(keep_failures, n=4000):
    runs = [trial(keep_failures, s) for s in range(n)]
    return (sum(r[0] for r in runs) / n,              # 通过率
            sum(r[1] for r in runs) / n,              # 平均步数
            sum(r[2] for r in runs) / n)              # 平均重复调用


def fact_retention(keep_failures):
    """关键事实保留：压缩前已知的失败候选，压缩后还能“问”出来的比例。"""
    rng, ok, total = random.Random(1), 0, 0
    for _ in range(500):
        before = set(rng.sample(range(N_CAND), EVERY))
        after = before if keep_failures else set()
        ok, total = ok + len(before & after), total + len(before)
    return ok / total


good, bad = evaluate(True), evaluate(False)
print(f"保留失败记录：通过率 {good[0]:.1%}  步数 {good[1]:.2f}  重复调用 {good[2]:.2f}")
print(f"丢掉失败记录：通过率 {bad[0]:.1%}  步数 {bad[1]:.2f}  重复调用 {bad[2]:.2f}")

assert good[0] == 1.0 and bad[0] > 0.97          # 只看通过率，只差一两个点
assert good[2] == 0 and bad[2] > 3.0             # 重复调用：0 对 4 次左右
assert bad[1] > 1.5 * good[1]                    # 平均步数多出五成以上
assert abs(good[1] - (N_CAND + 1) / 2) < 0.2     # 不重复抽样的期望步数 (N+1)/2 = 6.5
assert fact_retention(True) == 1.0 and fact_retention(False) == 0.0
```

输出：

| 摘要 | 通过率 | 平均步数 | 平均重复调用 | 失败记录保留率 |
|---|---|---|---|---|
| 保留失败记录 | 100.0% | 6.43 | 0.00 | 100% |
| 丢掉失败记录 | 98.2% | 10.50 | 4.02 | 0% |

只看通过率，两者差不到两个点，单次评测的噪声就可能盖过去。步数多了六成，重复调用从 0 到 4 次，这才是摘要坏了的信号。关键事实保留率直接指出了原因：失败记录一条没留下。保留失败记录时，平均步数是不放回抽样的期望 (12 + 1)/2 = 6.5，代码也 assert 了这一点。

这个桩很简化，真实模型不会完全随机重试。但方向是一样的：漏掉失败记录，代价体现在步数和成本上，不一定体现在通过率上。

**压缩 prompt 改了一版怎么上线**：在固定评测集上跑门禁，通过率不降、步数和重复调用不升、关键事实保留率不降，才放量，并保留回滚开关。线上回归和发布门禁见 [evaluation/05 线上评测、A/B 与回归门禁](../../evaluation/05-online-regression/README.md)。

## 7 用子 Agent 隔离上下文

### 7.1 做法和出处

主 Agent 把一个子任务交给子 Agent。子 Agent 用一个干净的上下文去探索，可能花掉几万 token，只交回一段精简的总结。Anthropic 文中说这段总结通常是 1,000 到 2,000 token。Claude Agent SDK 文档也写明，子 Agent 只把最终回复交回，主 Agent 的上下文只增长一段总结。

从第 2.2 节的模型推导：子 Agent 做的事是把一条长曲线切成几段短曲线。累计输入的平方项 g·n²/2 变成 J 段各自的 g·m²/2，m = n/J，加起来是 J·g·(n/J)²/2 = g·n²/(2J)，原来的 1/J。主上下文只增长 J 段派单和总结。

### 7.2 实现：什么时候隔离反而更贵

下面比较两种做法：主 Agent 自己做 5 个子任务、每个 15 步；或者每个子任务派给一个子 Agent。子 Agent 也要装配自己的固定前缀，还可能要重读主 Agent 已经读过的材料。

```python
# 子 Agent 隔离：同样的探索工作，放在主 Agent 里做和交给子 Agent 做，各花多少 token
P, a, r = 8_000, 300, 2_500     # 固定前缀、每步模型输出、每步工具结果
J, m = 5, 15                    # 5 个子任务，每个要探索 15 步
brief, back = 500, 1_500        # 交给子 Agent 的任务说明、交回的总结


def inline():
    """主 Agent 自己探索：所有工具结果都进主上下文。"""
    ctx, calls = P, []
    for _ in range(J * m):
        calls.append(ctx)
        ctx += a + r
    return ctx, calls


def isolated(reread=0):
    """每个子任务派给子 Agent。reread：子 Agent 要重新读一遍主 Agent 已经读过的内容。"""
    main, calls = P, []
    for _ in range(J):
        calls.append(main)              # 主 Agent 发出派单
        main += brief
        sub = P + brief + reread        # 子 Agent 自己的上下文从头装配
        for _ in range(m):
            calls.append(sub)
            sub += a + r
        main += back                    # 只有总结回到主上下文
    calls.append(main)                  # 主 Agent 汇总
    return main, calls


main_in, calls_in = inline()
main_iso, calls_iso = isolated()
print(f"主上下文：自己探索 {main_in:,}，派给子 Agent {main_iso:,}")
print(f"累计输入：自己探索 {sum(calls_in):,}，派给子 Agent {sum(calls_iso):,}")

assert main_in == P + J * m * (a + r) == 218_000      # 已经超过 200K 窗口
assert main_iso == P + J * (brief + back) == 18_000   # 主上下文只多了 J 段派单和总结
assert max(calls_iso) < 0.25 * max(calls_in)

# 推导：同样的工作，平方项被切成 J 段，累计输入反而更少
assert sum(calls_iso) < sum(calls_in)
# 但每个子 Agent 都要从头装配：固定前缀重付一次，还可能重读主 Agent 已有的材料。
# 重读量 x 让每个子 Agent 的 m 次调用都多 x，盈亏平衡点 x* = 差额 / (J·m)
x_star = (sum(calls_in) - sum(calls_iso)) / (J * m)
print(f"子 Agent 每个重读超过 {x_star:,.0f} token，累计输入就超过自己探索")
assert 80_000 < x_star < 85_000
assert sum(isolated(reread=int(x_star) + 1)[1]) > sum(calls_in)
```

输出：主上下文从 218,000 降到 18,000；累计输入从 8,370,000 降到 2,185,500。但每个子 Agent 要重读的材料超过约 8.2 万 token 时，累计输入就超过主 Agent 自己做。

这个模型只数了输入 token，还漏了两项真实成本：

- 子 Agent 之间不共享缓存前缀时，每个子 Agent 的固定前缀都要按写价付一次；
- 主 Agent 汇总、检查子 Agent 结果的步数。

所以多 Agent 系统的实际 token 用量往往比单 Agent 高得多，量级和出处见 [core/08 多 Agent](../../../core/08-multi-agent/README.md) 的成本一节。上面的模拟说明的是隔离本身的账：子任务之间越独立、要重读的共享材料越少，隔离越划算。

### 7.3 代价和适用条件

- **成本**：每个子 Agent 都是一次独立装配加多步调用，见上一节。
- **信息丢失**：子 Agent 看不到主 Agent 的完整上下文，可能误解任务；总结在层层传递中会走样。Anthropic 多 Agent 研究系统文章的做法是让子 Agent 把产物写到外部存储，只给主 Agent 传轻量的引用。

什么时候拆：子任务能独立完成、中间过程很长、主 Agent 只需要结论时才拆。子任务之间强依赖时，拆开反而要反复交接。Anthropic 的建议是：来回对话多的任务适合压缩，有明确阶段目标的迭代开发适合结构化笔记，能并行探索的复杂研究适合多 Agent。多 Agent 的编排和失败处理见 [core/08 多 Agent](../../../core/08-multi-agent/README.md)。

**交接要显式**：传过去的是目标、约束、输出格式、完成标准；交回来的是约定好字段的结构化结果（结论、证据引用、未解决问题），方便主 Agent 用代码汇总和检查。子 Agent 的权限不能比主 Agent 更宽，花费计入 Task 总预算。

## 8 数值例子：一个 80 步的任务

用第 3.3 节的参数手算一遍：固定前缀和任务描述 P + U = 8,500，每步模型输出 a = 300、工具结果 r = 2,500，窗口 200K，Anthropic 5 分钟缓存价。

**1. 什么时候撞窗口**。g = 2,800。k* = ⌈(200,000 − 8,500) / 2,800⌉ + 1 = ⌈68.4⌉ + 1 = 70。第 69 次调用是 198,900，第 70 次是 201,700。

**2. 80 步累计输入多少**。S_80 = 80 × 8,500 + 2,800 × 80 × 79 / 2 = 680,000 + 8,848,000 = 9,528,000。前一项是固定前缀，后一项是平方项，后者占 93%。（第 70 步以后其实已经超窗，这里按理论值算。）

**3. 开缓存、只追加，能省多少**。总成本 = 1.25 × 8,500 + 0.1 × (9,528,000 − 229,700) + 1.25 × 2,800 × 79 = 10,625 + 929,830 + 276,500 ≈ 1,216,955（基础输入价 × token），约为不开缓存的 12.8%。缓存把账单降到八分之一，但没解决撞窗口。

**4. 清理旧工具结果**。工具结果占每步新增的 2,500 / 2,800 = 89.3%。保留最近 3 个、其余换成 20 token 的占位符后，斜率从 2,800 降到 320，是原来的 1/8.75。80 步时上下文 41,220，离窗口还很远。

**5. 只用摘要**。T = 100K、s = 4K：第一次摘要发生在第 ⌊(100,000 − 8,500) / 2,800⌋ + 1 = 33 次调用，之后每 32 次一次。

```python
# 第 8 节的手算：P+U = 8,500，每步新增 2,800（工具结果 2,500），80 步
PU, g, r, n, W = 8_500, 2_800, 2_500, 80, 200_000
w, rho = 1.25, 0.1

C = [PU + (k - 1) * g for k in range(1, n + 1)]
S = n * PU + g * n * (n - 1) // 2
k_star = -(-(W - PU) // g) + 1
assert sum(C) == S == 9_528_000
assert k_star == 70 and C[68] == 198_900 and C[69] == 201_700

# 只追加 + 缓存：第 1 次全写，之后读上一请求全部、写新增 g
cached = w * PU + rho * (S - C[-1]) + w * g * (n - 1)
assert round(cached) == 1_216_955 and 0.12 < cached / S < 0.13

# 工具结果占每步新增的比例，以及清理（K=3、占位符 20）后的斜率
assert round(r / g, 3) == 0.893 and g / (g - r + 20) == 8.75

# 摘要（T=100K，s=4K）：第一次触发的步数和之后的周期
T, s = 100_000, 4_000
first, period = (T - PU) // g + 1, (T - PU - s) // g + 1
assert (first, period) == (33, 32)
print(k_star, S, round(cached), first, period)
```

## 9 工程细节与常见坑

每条写现象、原因、排查和处理。

**1. 长期规则只写在第一条消息里**

- 现象：任务前半段遵守“不改测试文件”，压缩之后开始改测试；或者输出格式在某一步之后变了。
- 原因：摘要把第一条消息里的约束改写或丢掉了。阈值压缩之后，`compaction` 块之前的内容全部被忽略。
- 排查：对比压缩前后的上下文，搜约束原文还在不在；看违规是否都发生在第一次压缩之后。
- 处理：长期规则放进每次请求都重新注入的位置（系统提示、项目说明文件）。自定义摘要 prompt 要显式列出“约束原文照抄”。注意 `instructions` 会完全替换默认 prompt，默认 prompt 里要求保留的内容要自己写回去。

**2. 摘要漏了失败记录**

- 现象：通过率几乎不变，平均步数和重复调用在开启压缩后明显上升；轨迹里能看到同一个失败的命令被跑了好几次。
- 原因：摘要格式里没有“失败过的”这一栏，模型写摘要时倾向于只写进展。
- 排查：第 6.2 节的指标，按压缩前后分开统计重复调用。
- 处理：摘要模板固定加“失败过的路径和原因”；或者把失败记录外置成文件，每次压缩后读回。

**3. 清理之后模型凭印象编数字**

- 现象：模型引用了一个早先读过的数值，和文件里的实际值不一致；时间点总在那个工具结果被清理之后。
- 原因：结果被换成占位符，模型没有重新读，而是按记忆复述。
- 排查：找出被引用的数值出自哪次工具调用，查那次结果是否已被清理。
- 处理：占位符写清“已清理，需要时重新调用某工具”；关键工具放进 `exclude_tools`；或者用外置，占位符里留路径。

**4. 恢复后重复副作用**

- 现象：崩溃恢复后，同一条评论、同一封邮件、同一个工单出现两次。
- 原因：恢复时重新执行了日志里有意图没结果的调用；或者幂等键用了随机值，恢复后变了；或者先调外部、后写日志，日志里没有意图记录。
- 排查：在日志里找这一步的意图和结果事件，看幂等键前后是否一致；看崩溃点在调外部前还是后。
- 处理：第 4.3 节的三条：先写意图再调外部；幂等键 = Task ID + 步骤编号；结果未知时先查。用故障注入在每个边界杀一次进程做回归测试。

**5. 恢复后模型换了一条路**

- 现象：崩溃前日志显示模型打算调 A，恢复后调了 B；两次运行的审计记录对不上。
- 原因：恢复时重新调用了模型，没有使用日志里已经写下的模型输出。
- 排查：看日志里崩溃点之前最后一条是不是模型输出事件，恢复后是否又出现了一条。
- 处理：模型输出也是事件，折叠进工作状态的“待执行动作”；恢复时有它就直接用。

**6. 框架从节点开头重跑**

- 现象：用 LangGraph 的 `interrupt()` 做审批，审批通过后，审批之前的那次写操作又执行了一遍。
- 原因：恢复时节点从头执行，`interrupt()` 之前的代码会再跑一次。
- 排查：看副作用代码是否在 `interrupt()` 之前、同一个节点里。
- 处理：副作用放到 `interrupt()` 之后或单独的节点；做不到就让它幂等。持久化模式选 `sync` 或 `async`，`exit` 模式中途崩溃无法恢复。

**7. 时间戳或无序序列化打破缓存**

- 现象：`cache_read_input_tokens` 每次都只等于工具定义那么长，或者干脆是 0；成本比预期高出几倍。
- 原因：系统提示开头有时间、请求 ID；工具定义的 JSON 键顺序不固定；每次按相关度重排工具。
- 排查：把相邻两次请求序列化后逐字节比对，找第一个不同的位置。
- 处理：动态内容移到最新消息；序列化固定键顺序和分隔符；一个 Task 内冻结工具列表。

**8. 压缩太频繁，账单反而上升**

- 现象：上下文长度降下来了，平均每任务成本却上升；缓存写入 token 占比很高。
- 原因：每次压缩都要重写压缩点之后的缓存，压缩量太小时回不了本（第 5.4 节）。逐步清理每一步都在改写历史。
- 排查：按请求画 `cache_creation_input_tokens`，看尖峰是否和压缩事件对齐；按第 5.4 节的公式算回本请求数。
- 处理：批量清理（`clear_at_least`）；软阈值别设太低；系统提示末尾单独加断点。

**9. 回看窗口外的断点**

- 现象：对话一切正常，某一轮之后命中率突然为 0，下一轮又恢复。
- 原因：这一轮新增了 20 个以上的块（例如多段文本和图片交替），上一次写入的条目落到了 20 个位置的回看窗口之外。
- 排查：数这一轮新增的块数。
- 处理：按文档建议，在更靠前的位置预先加第二个断点。

**10. Checkpoint 和日志不一致**

- 现象：从 Checkpoint 恢复和从头重放日志得到的状态不同；恢复后某一步被跳过或做了两次。
- 原因：快照在日志写入之前打；或者 reducer 不是纯函数，读了时钟、随机数、外部状态。
- 排查：定期 assert “快照 + 尾部日志”的折叠结果等于“从头折叠”。
- 处理：快照只在日志写入成功之后打，并记下它覆盖到的日志位置；reducer 里不调外部，需要的外部信息先作为事件写进日志。

**11. 子 Agent 返回全部原文**

- 现象：用了子 Agent，主上下文还是涨得很快；或者主 Agent 拿不到结论，只拿到一堆片段。
- 原因：没有约定交回的格式，子 Agent 把探索过程原样返回。
- 排查：看主上下文里每次子 Agent 返回的长度。
- 处理：约定结构化字段（结论、证据引用、未解决问题），设长度上限；大块产物写到外部存储，只传引用。

## 10 对比与取舍

### 10.1 四种压缩手段

| | 清理 | 外置 | 摘要 | 重置 |
|---|---|---|---|---|
| 损失 | 结果原文，能找回 | 原文不在上下文，能找回 | 摘要没写到的，原文另存才能找回 | 交接文件以外的，原文另存才能找回 |
| 对增长 | 降斜率 | 降斜率 | 锯齿，有上限 | 回到起点 |
| 对缓存 | 逐步做每步都失效；批量做可控 | 只追加，最友好 | 每次摘要失效一次 | 新会话，只剩固定前缀 |
| 额外调用 | 无 | 写文件、读回各一次工具调用 | 一次生成摘要的模型调用 | 一次写交接文件 |
| 适合 | 工具结果大、用过就不再看 | 结果以后可能还要查 | 同一阶段，需要连续性 | 新阶段；模型接近上限时提前收工 |

一般顺序：返回时就裁剪 → 外置或批量清理 → 接近软阈值时摘要 → 进入新阶段或摘要后仍超硬阈值时重置。

### 10.2 状态保存方式

| | 只存上下文 | 只存快照 | 只存日志 | 日志 + 快照 |
|---|---|---|---|---|
| 崩溃后恢复 | 不能 | 能，回到快照时点 | 能，从头折叠，慢 | 能，快 |
| 结果未知的副作用 | 不知道 | 快照之后的都不知道 | 有意图记录就知道 | 有意图记录就知道 |
| 审计和复盘 | 压缩后丢失 | 只有时点状态 | 完整 | 完整 |
| 压缩可逆 | 否 | 否 | 是 | 是 |
| 存储代价 | 小 | 中 | 随步数线性增长 | 线性增长加快照，要设保留策略 |

### 10.3 把问题对到章节

| 你遇到的问题 | 先看 |
|---|---|
| 上下文涨得快、撞窗口 | 第 2.2、3 节 |
| 压缩后行为变了 | 第 3.2、6 节，坑 1、2、3 |
| 崩溃恢复后重复副作用 | 第 4.2、4.3 节，坑 4、5、6 |
| 缓存命中率低、账单高 | 第 5 节，坑 7、8、9 |
| 子 Agent 不省 | 第 7 节，坑 11 |
| 线上 bad case 复现 | [06 Trace 与回放](../06-tracing-replay/README.md) |
| 工具超时与幂等键的服务端语义 | [04 工具编排](../04-tool-orchestration/README.md) |

## 11 自测清单

- [ ] 写出 C_k 和 S_n 的式子，算出第一次超窗的步数，说清为什么工具结果是主要增长源
- [ ] 说明缓存为什么只降常数、不降平方项，写出只追加时带缓存的总成本
- [ ] 对清理、外置、摘要、重置，各说出丢什么、留什么、能不能找回、对缓存的影响
- [ ] 列出不能压掉的内容，说出漏掉失败记录时哪些指标会变、哪些不会
- [ ] 讲清 Event Log、工作状态、Checkpoint、Resume、Context 五者的关系，写出折叠的式子
- [ ] 说出进程可能死在哪三个位置，以及每种情况恢复时怎么做
- [ ] 解释幂等键为什么用 Task ID 加步骤编号，不用随机值
- [ ] 说明为什么模型输出也要写进日志
- [ ] 说出时间戳、工具重排、中途加工具、压缩四种操作对命中率的影响
- [ ] 写出一次清理回本需要的请求数，说明 `clear_at_least` 的作用
- [ ] 设计一套评测，证明压缩没有丢关键信息
- [ ] 算出子 Agent 隔离在什么条件下反而更贵

## 资料

- Anthropic（Prithvi Rajasekaran 等），[Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)，2025-09
- Anthropic（Justin Young），[Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)，2025-11
- Anthropic（Prithvi Rajasekaran），[Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)，2026-03
- Anthropic（Lance Martin 等），[Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents)，2026-04-08
- Anthropic，[How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)，2025-06
- Anthropic，[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)，Claude API 文档，2026-11 查阅
- Anthropic，[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)，Claude API 文档，2026-11 查阅
- Anthropic，[Compaction at a token threshold](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)，Claude API 文档，2026-11 查阅
- Anthropic，[How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)，Claude Agent SDK 文档，2026-11 查阅
- Yichao Ji，[Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)，Manus 博客，2025-07-18
- Martin Fowler，[Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)，2005-12-12
- Nelson F. Liu 等，[Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)，arXiv 2307.03172，2023-07
- vLLM，[Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html)，设计文档，2026-11 查阅
- LangChain，[Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)、[Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)、[Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)，LangGraph 文档，2026-11 查阅
