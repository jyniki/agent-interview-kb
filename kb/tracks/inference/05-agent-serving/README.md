# 面向 Agent 的推理：前缀缓存、多轮 KV 复用与流式工具输出

> 层级：垂类 · inference · 预计 150 分钟 · 题目：[questions.md](questions.md)
> 前置：[01 KV Cache 与显存估算](../01-kv-cache-memory/README.md)（KV 公式、PagedAttention 的块与引用计数）

## 1 这一章解决什么问题

聊天请求是“短 prompt、长回答、一轮结束”。Agent 请求不一样：一次任务调用模型几十轮，每轮都带着越来越长的历史，中间还要停下来等工具。按聊天的方式部署 Agent，大部分算力花在重复 Prefill 上，工具调用还常常因为输出格式不对而失败。

推理侧有四件事要做对：

1. **前缀复用**：第 k 轮的输入就是第 k−1 轮的全部内容再加一点。引擎要认出这段重复，跳过重算。vLLM 用块哈希，SGLang 用基数树，命中粒度不同。
2. **路由和保留**：缓存只在一个引擎实例里有效，还会被淘汰。多副本时要把会话送回有缓存的那台；等工具的几十秒里，这段 KV 是留着、换出还是丢掉，要算账。
3. **流式工具调用**：客户端收到的是一个个字符片段，要拼回 id、name、arguments，还要知道什么时候才能执行。
4. **约束解码**：保证输出一定能被解析。正则用有限状态机，嵌套 JSON 要用带栈的下推自动机，XGrammar 把大部分检查挪到了预处理。

KV Cache 本身、每 token KV 的大小、PagedAttention 的块表和写时复制，都在 [01 章](../01-kv-cache-memory/README.md) 讲过，这里直接用。

学完你应该能：

- 推导多轮任务的 Prefill 成本：不复用时按轮数平方增长，复用后按线性增长，并说出注意力那部分为什么还高一阶；
- 讲清 vLLM 块哈希的组成、只缓存满块、逆序释放和 LRU 淘汰；讲清 RadixAttention 的基数树、叶子优先淘汰和最长前缀优先调度；说出两者命中粒度的差别，并用模拟验证；
- 比较轮询、会话粘滞、前缀感知三种路由，说出只看缓存不看负载会出什么事；
- 算清等工具期间留 KV、换出、重算三种做法的代价；
- 写出流式工具调用的服务端解析和客户端聚合，证明任意切分都拼出相同结果，说出何时可以执行；
- 从状态机和下推自动机讲清约束解码，说出 XGrammar 为什么要区分上下文无关和上下文相关 token；
- 说出生产里前缀命中率低、TTFT 抖动、工具参数不合法这几类问题的排查方法。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| S | 系统提示加工具定义的 token 数，每轮都一样 |
| o | 每轮模型输出的 token 数（工具调用或回答） |
| r | 每轮写回上下文的工具结果 token 数（含模板包装） |
| D | 每轮新增的 token 数，D = o + r |
| n | 一次任务的模型调用轮数；k 表示第 k 轮 |
| P_k | 第 k 轮的输入长度 |
| B | 块大小（vLLM 默认 16 个 token） |
| N、L、n_h、d_h | 参数量、层数、查询头数、每头维度，同 [01 章](../01-kv-cache-memory/README.md#1-这一章解决什么问题) |

## 2 Agent 请求有什么不同

### 2.1 负载特点

| 维度 | 聊天 | Agent |
|---|---|---|
| 轮数 | 一轮或几轮 | 一次任务几十轮模型调用 |
| 输入 | 短 | 长：系统提示 + 工具定义 + 全部历史 + 工具结果 |
| 输出 | 长文本 | 往往很短：一次工具调用、一段 JSON |
| 前缀 | 轮与轮之间部分重叠 | 每轮前缀就是上一轮的全部内容 |
| 中间停顿 | 等用户 | 等工具，可能几百毫秒到几分钟 |
| 格式要求 | 人读，容错高 | 程序解析，错一个括号就失败 |

输入长、输出短，意味着成本集中在 Prefill。第 [02 章](../02-batching-pd/README.md) 讲过 Prefill 受算力限制、Decode 受带宽限制；Agent 负载里 Prefill 占比高，所以第一优化点是别重复算它。

### 2.2 推导：不复用按平方增长，复用后按线性增长

假设每轮新增的长度固定为 D。第 k 轮的输入是系统提示加前 k−1 轮的全部内容：

```text
P_k = S + (k − 1)·D
```

**不复用**：每轮都把整个输入 Prefill 一遍。

```text
C_no(n) = Σ_{k=1..n} P_k
        = n·S + D·(0 + 1 + … + (n−1))
        = n·S + D·n(n−1)/2
```

第二项是 n 的二次式。轮数翻倍，这一项变成约 4 倍。

**完美复用**：第 1 轮算 S；之后每轮的前 P_{k−1} + o 个 token 都已经在缓存里（上一轮的输入和它生成的输出），只需要算新来的部分。先按“新增的 D 个 token 都要 Prefill”算：

```text
C_reuse(n) = S + (n − 1)·D = P_n
```

总量恰好等于最后一轮的输入长度，是 n 的一次式。两者之比：

```text
C_reuse / C_no = (S + (n−1)D) / (nS + D·n(n−1)/2)  →  2/n    （n 很大时）
```

所以轮数越多，复用省得越多。10 轮约省 85%，100 轮约省 98%。

**还能再少一点**：上一轮的模型输出 o 是 Decode 时一个个写进 KV 的。引擎会把写满的块一起放进缓存（vLLM 文档写明运行中的请求块写满就进缓存），所以第 k 轮真正要 Prefill 的只有工具结果 r。理想下限是 S + (n−1)·r。这个结论有前提：下一轮的模板没有改写上一轮的输出。第 9 节会讲 Qwen3 去掉思考内容时这个前提怎么被破坏。

### 2.3 注意力那一项高一阶

上面数的是 token。每个 token 的计算量并不一样：[01 章第 2.2 节](../01-kv-cache-memory/README.md#22-每步的计算量有缓存和没缓存) 推过，线性层每 token 约 2N FLOPs，注意力里位置 t 的查询要看 t 个键，约 4·L·n_h·d_h·t FLOPs。

- 不复用：第 k 轮的注意力是 Σ_{t=1..P_k} t ≈ P_k²/2，再对 k 求和，约 D²n³/6，按 n 的三次方增长。
- 复用：每个 token 只算一次，而且算的时候看的位置数和“一次性 Prefill 整段”时一样。所以总量恰好等于一次 Prefill 完 P_n 个 token 的注意力，约 P_n²/2，按 n 的平方增长。

推论：复用把线性层成本从平方降到线性，把注意力成本从三次降到平方，两者都降了一阶。但注意力在复用后的占比更高，因为留下来要算的新 token 都要看很长的前缀。这也意味着 FLOPs 省的比例略小于 token 省的比例。

### 2.4 实现：算一遍并验证增长阶

下面的代码用一个虚构的编码 Agent 任务（S = 6,000，D = 1,500，10 轮）和 Qwen3-8B 的形状，验证闭式、增长阶和注意力占比，并估第 10 轮 TTFT 的理想下限。

```python
# 多轮 Agent 的 Prefill 成本：不复用 vs 完美前缀复用
# 记号：S = 系统提示 + 工具定义，D = 每轮新增 token，n = 轮数
# 线性层每 token 约 2N FLOPs；注意力每个查询 token 对 T 个位置约 4·L·n_h·d_h·T FLOPs（inference/01 第 2.2 节）

N = 8.2e9                     # Qwen3-8B 参数量（模型卡）
L, n_h, d_h = 36, 32, 128     # Qwen3-8B config：层数、查询头数、head_dim


def prefill_tokens(S, D, n, reuse):
    """返回每轮要 Prefill 的 token 数列表。"""
    out = []
    for k in range(1, n + 1):
        length = S + D * (k - 1)                  # 第 k 轮输入长度
        new = length if (k == 1 or not reuse) else D
        out.append(new)
    return out


def attn_flops(prefix, new):
    """在长度为 prefix 的已缓存前缀后面追加 new 个 token，注意力打分 + 加权的 FLOPs。"""
    # 第 i 个新 token 要看 prefix + i 个位置：Σ_{i=1..new} (prefix + i)
    positions = new * prefix + new * (new + 1) // 2
    return 4 * L * n_h * d_h * positions


def total_flops(S, D, n, reuse):
    lin = att = 0
    for k, new in enumerate(prefill_tokens(S, D, n, reuse), 1):
        length = S + D * (k - 1)
        lin += 2 * N * new
        att += attn_flops(length - new, new)
    return lin, att


S, D, n = 6000, 1500, 10
no = prefill_tokens(S, D, n, reuse=False)
yes = prefill_tokens(S, D, n, reuse=True)
assert sum(no) == n * S + D * n * (n - 1) // 2 == 127_500      # 闭式：nS + D·n(n−1)/2
assert sum(yes) == S + D * (n - 1) == 19_500                   # 闭式：最后一轮的长度
print(f"省掉 {1 - sum(yes) / sum(no):.1%} 的 Prefill token")    # 84.7%

# 1) 增长阶：轮数翻倍，不复用的 token 数趋近 4 倍（平方），复用后趋近 2 倍（线性）
for n1 in (100, 1000):
    r_no = sum(prefill_tokens(S, D, 2 * n1, False)) / sum(prefill_tokens(S, D, n1, False))
    r_yes = sum(prefill_tokens(S, D, 2 * n1, True)) / sum(prefill_tokens(S, D, n1, True))
    print(n1, round(r_no, 3), round(r_yes, 3))
assert abs(r_no - 4) < 0.05 and abs(r_yes - 2) < 0.01

# 2) 注意力那一项高一阶：不复用 ~n³，复用后 ~n²（新 token 仍要看完整前缀）
a1 = total_flops(S, D, 1000, False)[1]; a2 = total_flops(S, D, 2000, False)[1]
b1 = total_flops(S, D, 1000, True)[1];  b2 = total_flops(S, D, 2000, True)[1]
assert abs(a2 / a1 - 8) < 0.1 and abs(b2 / b1 - 4) < 0.05

# 3) 10 轮这个例子里，注意力占 Prefill FLOPs 的两成左右，token 数是成本的主体
lin_no, att_no = total_flops(S, D, n, False)
lin_yes, att_yes = total_flops(S, D, n, True)
share_no, share_yes = att_no / (lin_no + att_no), att_yes / (lin_yes + att_yes)
print(f"注意力占比：不复用 {share_no:.0%}，复用 {share_yes:.0%}")
assert 0.15 < share_no < 0.25 and 0.2 < share_yes < 0.3
# 完美复用时每个 token 只算一次、看的位置数和一次性 Prefill 整段时相同：
# 注意力总量恰好等于“把最后一轮的输入一次 Prefill 完”
T_final = S + D * (n - 1)
assert att_yes == attn_flops(0, T_final)
print(f"FLOPs 比：{(lin_yes + att_yes) / (lin_no + att_no):.1%}，token 比：{sum(yes) / sum(no):.1%}")
# FLOPs 比 token 比更接近 1：复用省掉的是“短前缀”的重算，留下的新 token 都要看长前缀
assert (lin_yes + att_yes) / (lin_no + att_no) > sum(yes) / sum(no)

# 4) 第 10 轮 TTFT 的理想下限：H100 稠密 BF16 约 990 TFLOPS，按 100% 利用率算
peak = 990e12
t_no = (2 * N * no[-1] + attn_flops(0, no[-1])) / peak
t_yes = (2 * N * yes[-1] + attn_flops(S + D * (n - 2), yes[-1])) / peak
print(f"第 10 轮 Prefill 至少 {t_no * 1e3:.0f} ms（不复用） vs {t_yes * 1e3:.0f} ms（复用）")
assert t_no / t_yes > 10

# 5) 更细一点：上一轮的模型输出是 Decode 时写进 KV 的，引擎会把它一起缓存。
#    D = o（模型输出）+ r（工具结果），完美复用时第 k≥2 轮只需 Prefill r 个 token
o, r = 300, 1200
assert S + (n - 1) * r == 16_800 < sum(yes)
```

运行结果：

```text
省掉 84.7% 的 Prefill token
100 3.869 1.971
1000 3.986 1.997
注意力占比：不复用 20%，复用 26%
FLOPs 比：16.5%，token 比：15.3%
第 10 轮 Prefill 至少 436 ms（不复用） vs 42 ms（复用）
```

几点读法：

- 100 轮时比值是 3.87 和 1.97，还没到 4 和 2，因为 S 那一项还占一部分。轮数越多越接近。
- 10 轮的例子里，注意力占 Prefill FLOPs 的两成左右，token 数仍是成本的主体，所以用 token 估算误差不大。上下文到几万以上时注意力占比会继续上升。
- 436 ms 是按 H100 稠密 BF16 约 990 TFLOPS（[02 章第 4 节](../02-batching-pd/README.md#4-prefill-与-decode-的计算特点)）、100% 利用率算的下界，实际 MFU 达不到 100%，真实 TTFT 更长。差 10 倍这个比例更有参考价值。
- 这个会话最后占约 19,500 × 147,456 B ≈ 2.9 GB KV（每 token 144 KiB 的推导见 [01 章第 3.3 节](../01-kv-cache-memory/README.md#33-数值例子一张卡能放几条会话)）。

这个结论有两个前提：每一轮的前缀都和上一轮逐 token 一致；等工具的这段时间里，缓存没被淘汰。第 3、4 节讲引擎怎么找出重复前缀，第 5 节讲怎么让缓存留住，第 8 节把这些损耗放回同一个例子里算一遍。

## 3 前缀缓存：按块哈希

**问题**：两个请求的开头相同，KV 就相同（[01 章第 2.1 节](../01-kv-cache-memory/README.md#21-从注意力出发为什么缓存-kv不缓存-q) 推过：位置 j 的 K、V 只取决于前 j 个 token），没必要重算。引擎要能快速回答“这个新 prompt 的最长已缓存前缀有多长、在哪些物理块里”。

### 3.1 vLLM 的做法

以 vLLM 官方设计文档为准：

1. **块的身份是哈希**。块哈希由三部分算出：前一块的哈希、本块的 token id 元组、额外信息（LoRA ID、多模态输入的哈希、用于多租户隔离的 cache salt）。文档说放进完整 token 是为了降低哈希冲突。因为包含前一块哈希，一个块的哈希代表了“从开头到这一块的整段前缀”，所以只要查到第 i 块的哈希，前 i 块就都对上了。
2. **哈希算法**。`--prefix-caching-hash-algo` 可选 `sha256`、`sha256_cbor`、`xxhash`、`xxhash_cbor`，当前默认 `sha256`（先用 pickle 序列化）。文档说 v0.11 起默认改成 sha256 来处理冲突风险；xxhash 更快但不是密码学哈希，多租户下冲突可能导致未定义行为甚至泄露数据。
3. **只缓存写满的块**。块写满时分配哈希，被淘汰时清掉哈希。没写满的尾块还会变化，不进缓存。文档的例子：块大小 4，两个请求共享 10 个 token，只有前 2 块（8 个 token）命中，第 3 块只对上 4 个里的 2 个。
4. **查找**：新请求从第一块开始算哈希查表，命中就“touch”这个块：引用计数加 1，从空闲队列里摘出来。第一次不命中就停，后面的块哈希都依赖它，不可能再命中。
5. **至少留一个 token 现算**。Prefill 要输出最后一个位置的 logits，所以整段都命中时也要重算最后一个 token。vLLM 源码（`kv_cache_manager.py`）的注释说，因为命中长度要按块对齐，这可能导致重算最后一整块。
6. **释放不等于删除**。请求结束后，引用计数为 0 的块进入空闲队列，但保留哈希，后面的请求仍可能命中。
7. **淘汰**：空闲队列是双向链表，队头最久未用。要分配新块时从队头取；如果它带着缓存哈希，就从哈希表里删掉（淘汰）。请求释放的块**按逆序**放到队尾，所以一个请求最后面的块最先被淘汰。文档的理由：最后一块哈希了最长的前缀，最不可能被别的请求复用。
8. **cache salt**：请求可以带 `cache_salt` 字段，它被注入第一块的哈希。只有 salt 相同的请求才能共享块，用来防御“通过延迟推断别人缓存了什么”的时序攻击。

vLLM 当前版本的 `CacheConfig` 里 `enable_prefix_caching` 默认是 True（main 分支源码，2026-11 查阅）。

### 3.2 推论：块粒度损失和失效位置

- **尾部损失**：一个请求的命中长度是 B 的整数倍。和缓存里的序列共享 m 个 token 时，命中 ⌊m/B⌋·B，最多丢 B−1 个 token。完全相同的 prompt 再来一次，还要再丢最后一块。
- **失效从块开始**：前缀第 i 个 token 改了，命中长度是 ⌊i/B⌋·B，改动位置所在的整块和之后全部失效。
- **重复块**：文档还提到一个限制：V1 的块表只能追加，一个请求自己算出来的块即使和缓存里已有的块哈希相同，也不能换成已有的那块，同一内容会暂时存两份，请求释放后才消除。

缓存和正在运行的请求争同一个块池。负载高时空闲队列短，缓存块很快被淘汰，命中率下降。

## 4 RadixAttention：基数树

SGLang 论文（Zheng 等，2023-12）提出 RadixAttention：请求结束后不丢弃 prompt 和生成结果的 KV，而是放进一棵基数树（radix tree）里管理。

### 4.1 数据结构

- 基数树是压缩过的前缀树：一条边可以标一整段 token 序列，而不是一个 token。树上从根出发的每条路径对应一段缓存的前缀，节点指向对应的 KV。
- KV 按页存储，论文实现的页大小等于 1 个 token。所以匹配可以停在任意 token 上，没有块对齐的损失。SGLang 当前的 `--page-size` 参数默认仍是 1，可以调大。
- 新请求在树上做最长前缀匹配，命中部分直接复用；匹配停在一条边中间时，把这条边拆成两段。论文图 3 的例子：第二个聊天会话到来时，原来“系统提示 + 第一个会话”的节点被拆成“共享的系统提示”和“各自的对话”两段。
- 树结构放在 CPU 上，维护开销很小。论文在没有任何复用机会的 ShareGPT 负载上测：100 个请求共 74.3 秒，管理树结构只用 0.2 秒，不到 0.3%。

### 4.2 淘汰与引用计数

- 淘汰用 LRU，**先淘汰最久未用的叶子**。公共祖先要等它的子节点都被淘汰、自己变成叶子后才会被淘汰。效果是共享的系统提示活得最久。
- 每个节点有引用计数，记录有多少运行中的请求在用它；计数为 0 才可淘汰。
- 不预先划一块固定的缓存区。缓存的 token 和运行中的请求共用同一个显存池；等待的请求多时，系统会把缓存全部让出来换更大的 batch。
- 当前 SGLang 的 `--radix-eviction-policy` 默认 `lru`，还提供 `lfu` 等选项。

两种实现在淘汰上的共同点：都优先保留前缀短、被共享多的部分。vLLM 靠“逆序放回队尾”，SGLang 靠“只淘汰叶子”。

### 4.3 缓存感知调度

论文把命中率定义为“命中的 prompt token 数 / prompt token 总数”。等待队列里请求的执行顺序会影响命中率：调度器在不相关的请求间来回切换，会导致缓存抖动。

论文的做法：按匹配前缀长度排序，最长共享前缀优先（longest-shared-prefix-first），而不是先来先服务。论文证明（定理 3.1）：离线情况下，对一批请求按基数树的深度优先顺序访问，在缓存大小不小于最长请求长度时能达到最优命中率；最长共享前缀优先等价于深度优先顺序。在线时顺序会被打乱，但仍近似深度优先。

代价：贪心调度可能让前缀不常见的请求一直排在后面。论文自己写明会饿死，把和公平调度的结合留作以后的工作。

### 4.4 实现：两种缓存的命中对比

下面的代码实现两种缓存，喂同一批多轮请求：4 个会话共用 100 个 token 的系统提示，每轮追加 20 到 59 个 token，交错到达，共 24 个请求。两者用同样的 token 容量。

- `BlockHashCache`：链式 sha256 块哈希、只缓存满块、第一次不命中就停、释放时逆序放回队尾、从队头淘汰、cache salt。
- `RadixCache`：页大小 1 的基数树，边中间分叉时拆节点，容量满时淘汰最久未用的叶子，正在使用的路径受保护。
- 两者都“至少留 1 个 token 现算”。

验证的结论：块大小为 1 时两者命中相同；块越大，块哈希命中越少，差距每个请求不到一块；基数树命中等于“每个请求和之前所有请求的最长公共前缀”之和；改一个 token 时两者失效的位置；salt 隔离；容量紧张时两者都下降。

```python
# 两种前缀缓存的最小模拟：按块哈希（vLLM 式）与基数树（RadixAttention 式）
# 两者共用“容量 = 可放的 token 槽位数”，比较同一批多轮请求的命中 token 数
import hashlib
import random
from collections import OrderedDict


class BlockHashCache:
    """块大小 B。只缓存写满的块；块哈希链式包含前一块哈希；空闲块按 LRU 淘汰。"""

    def __init__(self, capacity_tokens, B, salt=""):
        self.B, self.num_blocks = B, capacity_tokens // B
        self.salt = salt
        self.cached = OrderedDict()   # 已缓存块的哈希；顺序即空闲队列（队头最久未用）

    def _hashes(self, tokens):
        h, out = self.salt, []
        for i in range(len(tokens) // self.B):          # 尾部不满一块的部分不算哈希
            blk = tokens[i * self.B:(i + 1) * self.B]
            h = hashlib.sha256(repr((h, tuple(blk))).encode()).hexdigest()
            out.append(h)
        return out

    def serve(self, tokens):
        """处理一个请求（Prefill 后立刻结束），返回命中的 token 数。"""
        hs = self._hashes(tokens)
        hit = 0
        while hit < len(hs) and hs[hit] in self.cached:   # 第一次不命中就停：后面的哈希都依赖它
            hit += 1
        mine = set(hs)                                  # 运行中的请求引用着这些块，不能被淘汰
        for h in hs[hit:]:                              # 新写满的块进缓存
            if len(self.cached) >= self.num_blocks:
                victim = next((k for k in self.cached if k not in mine), None)
                if victim is None:                      # 整个池都被这个请求占着：不再缓存
                    break
                del self.cached[victim]                 # 从空闲队列队头淘汰（最久未用）
            self.cached[h] = True
        # 请求结束：块按逆序放回队尾。第一块最后放、离队头最远；最后一块（前缀最长）最先被淘汰
        for h in reversed(hs):
            if h in self.cached:
                self.cached.move_to_end(h)
        # 至少要留 1 个 token 现算，才能得到最后一个位置的 logits。vLLM 要求命中长度按块对齐，
        # 所以整段都命中时会重算最后一整块（kv_cache_manager.py 的注释）
        return min(hit, (len(tokens) - 1) // self.B) * self.B


class Node:
    def __init__(self, parent=None):
        self.children = {}            # 首个 token → (边上的 token 序列, 子节点)
        self.parent = parent
        self.last = 0                 # 最近一次访问的时刻
        self.edge_len = 0


class RadixCache:
    """基数树，页大小 1 token。最长前缀匹配；容量满时淘汰最久未用的叶子。"""

    def __init__(self, capacity_tokens):
        self.root, self.cap, self.size, self.clock = Node(), capacity_tokens, 0, 0

    def _match(self, tokens):
        node, i, path = self.root, 0, []
        while i < len(tokens) and tokens[i] in node.children:
            edge, child = node.children[tokens[i]]
            j = 0
            while j < len(edge) and i + j < len(tokens) and edge[j] == tokens[i + j]:
                j += 1
            if j < len(edge):                           # 在边中间分叉：拆节点
                mid = Node(node)
                mid.edge_len = j
                node.children[tokens[i]] = (edge[:j], mid)
                mid.children[edge[j]] = (edge[j:], child)
                child.parent, child.edge_len = mid, len(edge) - j
                mid.last = child.last
                path.append(mid)
                return mid, i + j, path
            node, i = child, i + j
            path.append(node)
        return node, i, path

    def _evict(self, need, protect):
        while self.size + need > self.cap:
            leaves = []
            stack = [self.root]
            while stack:
                n = stack.pop()
                for _, c in n.children.values():
                    stack.append(c)
                if not n.children and n is not self.root and n not in protect:
                    leaves.append(n)
            if not leaves:
                return False
            victim = min(leaves, key=lambda n: n.last)  # LRU 叶子；祖先变成叶子后才轮到它
            for k, (e, c) in list(victim.parent.children.items()):
                if c is victim:
                    del victim.parent.children[k]
            self.size -= victim.edge_len
        return True

    def serve(self, tokens):
        self.clock += 1
        node, hit, path = self._match(tokens)
        for n in path:
            n.last = self.clock
        rest = tokens[hit:]
        if rest and self._evict(len(rest), protect=set(path)):
            leaf = Node(node)
            leaf.edge_len, leaf.last = len(rest), self.clock
            node.children[rest[0]] = (list(rest), leaf)
            self.size += len(rest)
        return min(hit, len(tokens) - 1)               # 同样留 1 个 token 现算


def agent_sessions(n_sessions, turns, S, seed=0):
    """虚构负载：所有会话共用系统提示（S 个 token），每轮追加长度不定的模型输出 + 工具结果。"""
    rng = random.Random(seed)
    system = [rng.randrange(1000) for _ in range(S)]
    sess = [list(system) for _ in range(n_sessions)]
    reqs = []
    for _ in range(turns):
        for s in sess:                                  # 会话交错到达
            s += [rng.randrange(1000, 50000) for _ in range(rng.randrange(20, 60))]
            reqs.append(list(s))
    return reqs


reqs = agent_sessions(n_sessions=4, turns=6, S=100)
total = sum(len(r) for r in reqs)

# 1) 容量充足：基数树按 token 命中，块哈希每次最多丢掉尾部不满一块的部分
big = 10 ** 6
for B in (1, 16, 64):
    bh = BlockHashCache(big, B)
    rx = RadixCache(big)
    hb = sum(bh.serve(r) for r in reqs)
    hr = sum(rx.serve(r) for r in reqs)
    print(f"B={B:2d}  块哈希命中 {hb:5d}  基数树命中 {hr:5d}  / prompt 总量 {total}")
    if B == 1:
        assert hb == hr                                # 块大小为 1 时两者命中相同
    else:
        assert hb <= hr and hr - hb <= B * len(reqs)   # 差距来自块粒度，每个请求不到一块

# 基数树命中 = 每个请求和之前所有请求的最长公共前缀之和
def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k

want = sum(min(len(r) - 1, max([lcp(r, p) for p in reqs[:i]] or [0])) for i, r in enumerate(reqs))
rx = RadixCache(big)
assert sum(rx.serve(r) for r in reqs) == want

# 2) 前缀里改一个 token（例如系统提示里的时间戳），从那一块往后全部失效
bh = BlockHashCache(big, 16)
a = list(range(200))
bh.serve(a)
b = list(a); b[40] = -1                                # 第 40 个 token 落在第 2 块（0 起算）
assert bh.serve(b) == 32                               # 只有前两块（32 token）命中
rx = RadixCache(big); rx.serve(a)
assert rx.serve(b) == 40                               # 基数树命中到改动位置为止

# 3) cache salt：不同租户的相同 prompt 不共享
t1 = BlockHashCache(big, 16, salt="tenant-1")
assert t1.serve(a) == 0 and t1.serve(a) == 192         # 同租户第二次命中 12 个满块
other = BlockHashCache(big, 16, salt="tenant-2")
other.cached = t1.cached                               # 共用同一个块池
assert other.serve(a) == 0

# 4) 完全相同的 32 token prompt 再来一次：块哈希只能用 1 块，基数树能用 31 个 token
bh, rx = BlockHashCache(big, 16), RadixCache(big)
p32 = list(range(1000, 1032))
bh.serve(p32); rx.serve(p32)
assert bh.serve(p32) == 16 and rx.serve(p32) == 31

# 5) 容量紧张：两种缓存都靠 LRU 淘汰，命中都下降
small = 600
bh, rx = BlockHashCache(small, 16), RadixCache(small)
hb = sum(bh.serve(r) for r in reqs)
hr = sum(rx.serve(r) for r in reqs)
print(f"容量 {small} token：块哈希命中 {hb}，基数树命中 {hr}")
assert hb < 4240                                       # 容量充足、B=16 时是 4240
assert hr < want
```

运行结果：

```text
B= 1  块哈希命中  4417  基数树命中  4417  / prompt 总量 5400
B=16  块哈希命中  4240  基数树命中  4417  / prompt 总量 5400
B=64  块哈希命中  3648  基数树命中  4417  / prompt 总量 5400
容量 600 token：块哈希命中 3120，基数树命中 3173
```

这组数说明：在多轮 Agent 这种前缀一直增长的负载上，块大小 16 时块哈希只比基数树少命中 4%（4,240 对 4,417）。每轮新增的 token 远多于一块，块粒度的损失摊得很薄。差距在“很多短请求共享一段不对齐的前缀”时才明显，比如块大小 64 时少了 17%。容量紧张时两者都靠 LRU，差别不大。

模拟和真实引擎的区别：这里请求一进来就结束，不模拟并发运行；真实 vLLM 把块缓存在请求运行过程中就做了，同一批里的其他请求也能用；真实 SGLang 还有缓存感知调度，这里按到达顺序处理。

### 4.5 两者对比

| | 按块哈希（vLLM） | 基数树（SGLang） |
|---|---|---|
| 匹配粒度 | 块（默认 16 token），命中长度是 B 的整数倍 | token（页大小默认 1） |
| 数据结构 | 哈希表 + 双向链表空闲队列 | 基数树，放在 CPU |
| 查找代价 | 每块一次哈希查表 | 沿树走一遍，按边比较 |
| 共享方式 | 多个块表指向同一物理块 | 树上的公共祖先节点 |
| 淘汰 | 空闲队列队头（LRU），请求的块逆序放回 | 最久未用的叶子 |
| 调度配合 | 主要靠路由和 prompt 设计 | 论文提出最长共享前缀优先 |
| 隔离 | cache salt 进第一块哈希 | 按引擎配置 |

两者思想相同：复用最长公共前缀的 KV。差别在数据结构、粒度和调度。它们省的是重复计算和重复存储，每个 token 的 KV 大小一样。

## 5 会话粘滞与 KV 复用

**问题**：前缀缓存只在单个引擎实例里有效。多副本部署时，同一个会话的第 5 轮如果被负载均衡发到了另一台机器，就要从头 Prefill。另外，等工具的这段时间里，这个会话没有运行中的请求，它的块随时可能被淘汰。

### 5.1 三种路由策略

| 策略 | 怎么路由 | 优点 | 代价 |
|---|---|---|---|
| 轮询 / 最少负载 | 不看内容 | 负载均匀 | 多轮会话的历史几乎不命中，只能命中各副本都有的系统提示 |
| 会话粘滞 | 同一会话 ID 固定到一台 | 实现简单，多轮必命中 | 会话少或有热点长会话时单机过载；机器故障时会话缓存全部失效 |
| 前缀感知 | 网关维护各实例的前缀索引，选缓存了最长前缀的实例 | 不同会话共享系统提示也能命中；不需要客户端传会话 ID | 网关要分词、维护索引；必须和负载一起权衡 |

SGLang 论文附录 A.4 描述了数据并行下的做法：每个副本维护自己的子树，路由器维护一棵元树（meta-tree），记录每个子树在哪个设备上；新请求先在元树上做前缀匹配，按共享前缀长度决定发给谁；副本发生淘汰时把淘汰提交到队列，路由器在空闲时更新元树。

SGLang 当前的路由组件（sgl-router README，2026-11 查阅）：

- `cache_aware`：选缓存了最长前缀的引擎，要求匹配至少 `--cache-affinity-min-matched-tokens` 个 token（默认 1,024），也可以要求匹配占 prompt 的最小比例。需要分词器，前缀匹配来自路由器本地的基数树。
- 亲和与负载怎么取舍：默认的 `prefer` 模式下，偏好的引擎只要通过准入就选它；`balanced` 模式下再抽一个备选，只有偏好引擎的负载同时超过“备选 × `--affinity-load-factor`（默认 2）”和“备选 + `--affinity-load-gap`”时才换。负载默认按 prefill token 数算。
- `session_aware`：按请求头里的会话 ID（默认 `x-session-id`）绑定引擎。绑定关系存在每个路由器副本的内存里，路由器有多个副本时，同一会话还要先粘到同一个路由器副本上。
- 两种亲和策略在没有偏好引擎或它没通过准入时，都退回 `power_of_two`：抽两个引擎选负载低的。

### 5.2 实现：三种路由的命中和负载

下面模拟 4 个副本，8 个会话共用 4,000 token 的系统提示，每轮追加 200 到 599 个 token，会话随机交错。命中按“这个副本服务过的序列里的最长公共前缀”算（容量充足，不淘汰），负载按每个副本实际要 Prefill 的 token 数算。`balanced` 一行照 sgl-router 的规则：偏好副本的负载超过备选 2 倍且多出一个绝对差时，放弃亲和。

```python
# 多副本路由：轮询 / 会话粘滞 / 前缀感知，比较前缀命中和负载
# 每个副本的缓存用“已服务过的序列集合 + 最长公共前缀”近似（容量充足，不淘汰）
import random
import zlib


def lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def workload(n_sessions, turns, S, seed, hot=None):
    """所有会话共用 S 个 token 的系统提示；hot=(会话号, 倍数) 时该会话轮数是别人的若干倍。"""
    rng = random.Random(seed)
    system = [rng.randrange(1000) for _ in range(S)]
    hist = {s: list(system) for s in range(n_sessions)}
    todo = {s: turns * (hot[1] if hot and s == hot[0] else 1) for s in hist}
    reqs = []
    while any(todo.values()):
        s = rng.choice([k for k, v in todo.items() if v])           # 会话随机交错到达
        todo[s] -= 1
        hist[s] += [rng.randrange(1000, 10 ** 6) for _ in range(rng.randrange(200, 600))]
        reqs.append((s, list(hist[s])))
    return reqs


def simulate(reqs, R, policy, min_match=1024, gap=8000):
    served = [[] for _ in range(R)]       # 每个副本缓存过的序列
    load = [0] * R                        # 每个副本处理的 prompt token 数（代替负载）
    hit = total = 0
    for i, (sess, toks) in enumerate(reqs):
        if policy == "round_robin":
            r = i % R
        elif policy == "session":
            r = zlib.crc32(str(sess).encode()) % R                  # 按会话 ID 哈希绑定
        else:                                                        # cache_aware / balanced
            best = [max([lcp(toks, p) for p in served[k]] or [0]) for k in range(R)]
            r = max(range(R), key=lambda k: best[k])
            alt = min(range(R), key=lambda k: load[k])
            if best[r] < min_match:                                  # 匹配太短：退回最少负载
                r = alt
            elif policy == "balanced" and load[r] > 2 * load[alt] and load[r] > load[alt] + gap:
                r = alt                                              # 偏好的副本太忙：放弃亲和
        h = max([lcp(toks, p) for p in served[r]] or [0])
        hit, total = hit + h, total + len(toks)
        load[r] += len(toks) - h                                     # 真正要 Prefill 的量
        served[r].append(toks)
    return hit / total, max(load) / (sum(load) / R)


reqs = workload(n_sessions=8, turns=8, S=4000, seed=1)
res = {p: simulate(reqs, 4, p) for p in ("round_robin", "session", "cache_aware", "balanced")}
for p, (h, imb) in res.items():
    print(f"{p:12s} 命中率 {h:.1%}  最忙副本 Prefill 量 / 平均 {imb:.2f}")
# 1) 轮询也能命中共享的系统提示，但会话历史只有落到同一副本才命中，所以命中率最低
assert res["round_robin"][0] < res["session"][0] < res["cache_aware"][0]
# 2) 只看缓存不看负载：所有请求都共享 4000 token 的系统提示（超过 1024 的门槛），
#    第一个副本永远匹配最长，全部流量压到一台，最忙副本的量是平均的 R 倍
assert abs(res["cache_aware"][1] - 4) < 1e-9
# 3) 加上负载门槛（偏好副本的负载超过备选的 2 倍且多出一个绝对差才切换）后，
#    负载基本均衡，命中率仍高于轮询
assert res["balanced"][1] < 1.5 and res["balanced"][0] > res["round_robin"][0]

# 4) 会话数少、又有一个热点长会话时，按会话哈希会把负载压到个别副本
reqs = workload(n_sessions=5, turns=4, S=4000, seed=2, hot=(0, 6))
h_s, imb_s = simulate(reqs, 4, "session")
h_r, imb_r = simulate(reqs, 4, "round_robin")
print(f"热点场景：粘滞 命中 {h_s:.1%} 不均衡 {imb_s:.2f}；轮询 命中 {h_r:.1%} 不均衡 {imb_r:.2f}")
assert imb_s > imb_r
```

运行结果：

```text
round_robin  命中率 77.0%  最忙副本 Prefill 量 / 平均 1.12
session      命中率 88.7%  最忙副本 Prefill 量 / 平均 1.04
cache_aware  命中率 91.9%  最忙副本 Prefill 量 / 平均 4.00
balanced     命中率 86.9%  最忙副本 Prefill 量 / 平均 1.40
热点场景：粘滞 命中 90.3% 不均衡 2.06；轮询 命中 78.4% 不均衡 1.06
```

读法：

- 轮询的命中率也有 77%，因为系统提示很长，每台副本都缓存了它。差的那部分是会话历史。
- 纯前缀感知把所有请求都压到一台：所有请求和第一台副本的匹配都超过 1,024，谁也抢不走。这就是 sgl-router 要有负载门槛和退回策略的原因。这个模拟故意放大了效果（共享前缀很长、没有准入），真实系统里准入检查会拦下一部分。
- 会话少、有一个热点长会话时，粘滞的负载不均衡到 2 倍。会话多、长度均匀时粘滞的不均衡很小。

### 5.3 等工具时缓存会不会丢

会。工具执行期间这个会话没有运行中的请求，它的块引用计数为 0，处在可淘汰状态。负载高、工具又慢时，下一轮回来可能已经被淘汰。三种处理方式的账：

```python
# 等工具期间这段 KV 怎么办：留在显存、换出到 CPU、丢掉重算，三种做法的账
# 会话取第 2 节的例子：第 10 轮结束时上下文 19,500 token，模型是 Qwen3-8B
N, L, n_h, n_kv, d_h = 8.2e9, 36, 32, 8, 128
kv_per_token = 2 * L * n_kv * d_h * 2            # BF16，147,456 B（inference/01 第 3.3 节）
T = 19_500
kv = T * kv_per_token
assert round(kv / 1e9, 1) == 2.9

PEAK = 990e12                                     # H100 SXM BF16 稠密，按 100% 利用率算（下界）
PCIE_ONE_WAY = 64e9                               # 官方标称 PCIe Gen5 128 GB/s 为双向合计，单向取一半（假设）

# 1) 重算：把 19,500 token 从头 Prefill 一遍
flops = 2 * N * T + 4 * L * n_h * d_h * T * (T + 1) // 2
t_recompute = flops / PEAK
# 2) 换入：把 2.9 GB 从 CPU 内存拷回显存
t_swap_in = kv / PCIE_ONE_WAY
print(f"重算 ≥ {t_recompute * 1e3:.0f} ms，换入 ≈ {t_swap_in * 1e3:.0f} ms")
assert t_recompute > 5 * t_swap_in               # 长上下文时换入比重算便宜得多

# 3) 一直留在显存：占着的显存不干活。一个会话每轮“模型在算”的时间是 t_model，
#    “在等工具”的时间是 t_tool，它的 KV 空闲的比例是 t_tool / (t_model + t_tool)
t_model, t_tool = 6.0, 30.0                       # 假设：每轮生成约 6 s，工具约 30 s
idle = t_tool / (t_model + t_tool)
assert abs(idle - 5 / 6) < 1e-9
budget = 51.6e9                                   # 单卡留给 KV 的字节（inference/01 第 3.3 节的假设）
pinned_sessions = int(budget // kv)               # 全部钉住时最多容纳的会话数
active = pinned_sessions * (1 - idle)             # 其中平均只有这么多在跑
print(f"钉住 {pinned_sessions} 个会话，平均只有 {active:.1f} 个在用 GPU")
assert pinned_sessions == 17 and active < 3

# 4) 短上下文时账反过来：2,000 token 的会话重算只要几十毫秒，留着或换出都不划算
T2 = 2_000
t_re2 = (2 * N * T2 + 4 * L * n_h * d_h * T2 * (T2 + 1) // 2) / PEAK
assert t_re2 < 0.05
```

运行结果：

```text
重算 ≥ 436 ms，换入 ≈ 45 ms
钉住 17 个会话，平均只有 2.8 个在用 GPU
```

由此得到的判断（推导，数字依赖上面的假设）：

- **长上下文、工具慢**：换出到 CPU 内存最划算。换入 2.9 GB 约 45 ms，比重算的 436 ms 下界快一个数量级。代价是占 CPU 内存，且拷贝和计算争 PCIe。
- **短上下文**：重算几十毫秒，直接让它被淘汰就行。
- **一直钉在显存**：等工具占 5/6 时间时，钉住的 17 个会话平均只有 2.8 个在用 GPU，显存利用率很低。只适合工具很快（几百毫秒）的场景。

引擎侧的支持（文档 2026-11 查阅）：

- vLLM：`--kv-offloading-size` 设置 KV 卸载到 CPU 的缓冲大小（GiB，TP 大于 1 时是各 rank 的总和），设了才启用；`--kv-offloading-backend` 可选 `native`（默认）和 `lmcache`。显存不足时的抢占默认用重算而不是换出（见 [01 章第 5.4 节](../01-kv-cache-memory/README.md#54-收益与代价)）。
- SGLang：`--enable-hierarchical-cache` 开启分层缓存，默认关闭；`--hicache-ratio` 是主机 KV 池和显存池的大小之比，默认 2.0；写策略默认 `write_through`；还可以再接一层存储后端。

其他可以做的事：

- 给工具设超时上限，长耗时工具接受一次重算；
- 托管 API 也有同样的问题。以 Anthropic 的 Prompt Caching 为例，缓存默认 5 分钟有效、每次命中刷新，可选 1 小时；两种 TTL 的计价不同，工具耗时经常超过 5 分钟时要比较总成本，倍率见 [harness/02](../../harness/02-context-compaction/README.md) 的缓存一节。

### 5.4 Harness 怎么配合

1. 稳定的内容放最前：系统提示、工具定义，按固定顺序序列化。
2. 动态内容放后面：时间戳、请求 ID、检索结果不要插进系统提示里。
3. 历史只追加，不改写。做上下文压缩时一次性重写，接受那一次失效（见 [harness/02 上下文压缩](../../harness/02-context-compaction/README.md)）。
4. 注意聊天模板的隐式改写。例如 Qwen3 模型卡建议多轮时历史里不保留思考内容，官方模板会自动去掉。这意味着上一轮生成的思考部分的 KV 在下一轮用不上，下一轮要从这里开始重算。第 8 节算了这笔账。
5. 多副本时在请求里带稳定的会话 ID，让网关能做粘滞或至少有一个退路。

## 6 流式工具调用输出的解析

**问题**：Agent 需要流式输出降低首字延迟，但工具调用是结构化的。客户端拿到的是一个个文本片段，要边收边判断“这是普通文本还是工具调用”“参数写完了没有”。

### 6.1 两层解析

1. **服务端把模型原始文本解析成工具调用**。不同模型用不同的标记包住工具调用，例如 Hermes 格式用 `<tool_call>` 和 `</tool_call>`，中间是 `{"name": ..., "arguments": {...}}`。vLLM 用 `--tool-call-parser` 选模型对应的解析器（Qwen2.5 也用 `hermes`）。流式时，解析器要“扣住”可能是半个起始标签的后缀，不能先当普通文本吐给客户端。
2. **客户端把增量拼回完整调用**。OpenAI 兼容的流式格式里，每个 chunk 的 `delta.tool_calls` 是一个数组，每项有 `index`，以及可选的 `id`、`type`、`function.name`、`function.arguments`。arguments 是分片下发的字符串。结束时 `finish_reason` 为 `"tool_calls"`。

服务端的状态机只有两个状态：

- **文本态**：在缓冲区里找 `<tool_call>`。找到了，前面的内容作为 content 发出，切到调用态，index 加 1。没找到，就把缓冲区发出去，但末尾如果是 `<`、`<to`、`<tool_ca` 这种可能是标签开头的后缀，先扣住，等下一片再说。
- **调用态**：累积到 `"arguments":` 出现时，前面的 `{"name": "..."` 已经完整，解析出 name，连同 id 发一个增量。之后 arguments 按到达的字符下发，同样要扣住末尾可能属于外层对象的 `}` 和可能是 `</tool_call>` 开头的后缀。遇到结束标签，回到文本态。

### 6.2 何时可以执行

- **arguments 什么时候到齐**：arguments 是一个 JSON 对象。一个合法 JSON 对象的任何真前缀都不是合法 JSON 对象：最外层的 `}` 是最后一个非空白字符，在它之前括号没配平，或者字符串没闭合。所以“缓冲区第一次能整体解析成对象”的那一刻就是到齐的那一刻。下面代码的第 4 部分验证了这一点。
- **工程上还是等结束信号**：`finish_reason == "tool_calls"`，或者服务端已经开始发下一个 index。并行调用时，每个 index 到齐就可以先执行，不用等全部。
- **执行前要校验**：到齐不等于正确。vLLM 文档写明，`auto` 模式下没有文法约束时，参数偶尔会不合法或不符合 schema；还提到 Llama 3.x 会把数组序列化成字符串。执行前按 schema 校验，失败时把错误作为工具结果返回模型重试，而不是直接崩溃（见 [harness/04 工具编排](../../harness/04-tool-orchestration/README.md)）。
- **想提前展示**：用容错的增量 JSON 解析器展示半截参数，但执行一定等完整参数。

### 6.3 实现：任意切分都拼出相同结果

下面的代码实现了服务端解析器和客户端聚合。原始文本里故意放了干扰项：普通文本里的 `<b>` 和 `<2`、参数值里的 `</tool_cal>` 和 `{x}`。验证：

1. 一次性喂完，得到正确的文本和两个调用；
2. 随机切 2,000 次（每次切 0 到 39 刀）和逐字符切，拼出的文本、name、arguments 原始字符串完全相同，content 里永远不会漏出标签碎片；
3. 逐字符流式时，绝大多数 arguments 分片单独 `json.loads` 会失败；
4. arguments 只有完整的那一刻才能解析成对象。

```python
# 流式工具调用的两层解析
# 服务端：把模型原始文本（Hermes 式 <tool_call>{...}</tool_call>）增量切成 content 与 tool_calls 增量
# 客户端：按 index 聚合 id / name / arguments，finish_reason 到了再整体 json.loads
import json
import random

OPEN, CLOSE = "<tool_call>", "</tool_call>"


def partial_suffix(buf, tag):
    """buf 末尾有多长可能是 tag 的开头（需要扣住不发）。"""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if tag.startswith(buf[-k:]):
            return k
    return 0


class ServerParser:
    """每次喂一段原始文本，返回 OpenAI 风格的 delta 列表。"""

    def __init__(self):
        self.buf, self.in_call, self.index = "", False, -1
        self.call_text, self.sent_name = "", False
        self.sent_args = 0

    def feed(self, text, final=False):
        self.buf += text
        out = []
        while True:
            if not self.in_call:
                i = self.buf.find(OPEN)
                if i >= 0:
                    if i:
                        out.append({"content": self.buf[:i]})
                    self.buf, self.in_call = self.buf[i + len(OPEN):], True
                    self.index += 1
                    self.call_text, self.sent_name, self.sent_args = "", False, 0
                    continue
                keep = 0 if final else partial_suffix(self.buf, OPEN)   # 半个起始标签先扣住
                if len(self.buf) > keep:
                    out.append({"content": self.buf[:len(self.buf) - keep]})
                    self.buf = self.buf[len(self.buf) - keep:]
                return out
            j = self.buf.find(CLOSE)
            keep = 0 if j >= 0 else partial_suffix(self.buf, CLOSE)
            upto = j if j >= 0 else len(self.buf) - keep
            self.call_text += self.buf[:upto]
            self.buf = self.buf[upto:]
            out += self._emit_call_deltas(done=j >= 0)
            if j >= 0:
                self.buf, self.in_call = self.buf[len(CLOSE):], False
                continue
            return out

    def _emit_call_deltas(self, done):
        """name 一旦完整就发（连同 id）；arguments 按到达的原始字符分片下发。"""
        out = []
        t = self.call_text
        if not self.sent_name:
            k = t.find('"arguments"')
            colon = t.find(":", k + len('"arguments"')) if k >= 0 else -1
            if colon < 0:                            # "arguments": 还没到齐，name 也先不发
                return out
            head = json.loads(t[:k].rstrip().rstrip(",") + "}")   # {"name": "..."} 已经完整
            out.append({"tool_calls": [{"index": self.index, "id": f"call_{self.index}",
                                        "type": "function", "function": {"name": head["name"]}}]})
            self.sent_name = True
            self.args_start = colon + 1
        body = t[self.args_start:]
        if done:                                     # 去掉外层对象的右花括号
            body = body.rstrip()[:-1]
        else:                                        # 末尾的空白和 } 可能属于外层对象，先扣住
            body = body.rstrip()
            if body.endswith("}"):
                body = body[:-1]
        if len(body) > self.sent_args:
            piece = body[self.sent_args:]
            if self.sent_args == 0:
                piece = piece.lstrip()
            if piece:
                out.append({"tool_calls": [{"index": self.index, "function": {"arguments": piece}}]})
            self.sent_args = len(body)
        return out


def _parses(text):
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def client_aggregate(deltas):
    content, calls = "", {}
    for d in deltas:
        content += d.get("content", "")
        for tc in d.get("tool_calls", []):
            c = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
            if tc.get("id"):
                c["id"] = tc["id"]
            fn = tc.get("function", {})
            c["name"] += fn.get("name", "")
            c["arguments"] += fn.get("arguments", "")
    return content, calls


def run(raw, cuts):
    p, deltas, last = ServerParser(), [], 0
    for c in cuts + [len(raw)]:
        deltas += p.feed(raw[last:c], final=(c == len(raw)))
        last = c
    finish = "tool_calls" if p.index >= 0 else "stop"
    content, calls = client_aggregate(deltas)
    assert finish == "tool_calls"                    # 只有在这之后才解析、校验、执行
    parsed = [(c["id"], c["name"], json.loads(c["arguments"])) for _, c in sorted(calls.items())]
    raw_args = [c["arguments"] for _, c in sorted(calls.items())]
    return content, parsed, deltas, raw_args


raw = ('先查一下天气和航班 <b>注意</b> 1<2。'
       '<tool_call>{"name": "get_weather", "arguments": {"city": "上海", "days": [1, 2]}}</tool_call>'
       '<tool_call>{"name": "search", "arguments": {"q": "a</tool_cal> b {x}"}}</tool_call>')
want_text = '先查一下天气和航班 <b>注意</b> 1<2。'
want_calls = [("call_0", "get_weather", {"city": "上海", "days": [1, 2]}),
              ("call_1", "search", {"q": "a</tool_cal> b {x}"})]

# 1) 一次性喂完
text, calls, _, args0 = run(raw, [])
assert text == want_text and calls == want_calls

# 2) 任意切分（包括逐字符、把标签切成两半）都拼出同样的结果
rng = random.Random(0)
for trial in range(2000):
    k = rng.randrange(0, 40)
    cuts = sorted(rng.sample(range(1, len(raw)), k))
    text, calls, deltas, args = run(raw, cuts)
    assert text == want_text and calls == want_calls and args == args0, cuts   # 连原始字符串都一样
    # 扣住半个标签：任何 content 增量里都不会漏出 "<tool" 这类标签碎片
    assert all("<tool" not in d.get("content", "") for d in deltas)
text, calls, _, _ = run(raw, list(range(1, len(raw))))   # 逐字符
assert text == want_text and calls == want_calls

# 3) 中途的 arguments 片段一般不是合法 JSON：不能逐片 json.loads，更不能提前执行
_, _, deltas, _ = run(raw, list(range(1, len(raw))))
pieces = [tc["function"]["arguments"] for d in deltas for tc in d.get("tool_calls", [])
          if "arguments" in tc.get("function", {})]
bad = 0
for p in pieces:
    try:
        json.loads(p)
    except json.JSONDecodeError:
        bad += 1
assert bad > 0.9 * len(pieces)
print(f"{len(pieces)} 个 arguments 分片里 {bad} 个单独解析会失败")

# 4) 什么时候能执行：arguments 是 JSON 对象，最外层 } 一出现，后面再接任何非空白字符都不合法。
#    所以“缓冲区第一次能整体解析成对象”的那一刻，这个调用的参数就到齐了
for full in args0:
    ok = [k for k in range(1, len(full) + 1) if _parses(full[:k])]
    assert ok == [len(full)]
```

运行结果：

```text
39 个 arguments 分片里 37 个单独解析会失败
```

模拟和真实解析器的区别：真实服务端拿到的是 token 而不是字符，一个 token 可能横跨标签边界，所以扣住的逻辑一样要有；真实解析器还要处理模型没写结束标签就停了（`max_tokens` 截断）的情况。vLLM 文档描述 `hf` 解析器的做法是：截断的调用按已生成的文本解析，解析不了就丢弃。

## 7 结构化输出与约束解码

**问题**：靠提示词要求“只输出 JSON”，模型偶尔仍会多一个逗号、少一个括号，或者输出 schema 外的字段。

### 7.1 原理：每步一个合法 token 集合

约束解码在每一步做三件事：

1. 根据目标语法和已生成的内容，维护一个解析状态；
2. 算出词表里哪些 token 接在当前状态后面仍可能得到合法输出，得到一个 0/1 掩码；
3. 把不合法 token 的 logits 置为负无穷，再采样，然后用采到的 token 推进状态。

只要每一个可达状态至少有一个合法 token（包括“合法结束”时的 EOS），并且 EOS 只在文法完成时合法，那么只要生成自然结束，输出一定符合文法。这是归纳论证：每一步都停在“某个合法串的前缀”上，结束时这个前缀本身就是合法串。

难点在第 2 步。词表有十几万个 token，每步逐个检查太慢。Outlines 论文（Willard、Louf，2023-07）指出，逐 token 扫词表的做法每步代价是 O(N)（N 是词表大小）；他们把正则编译成有限状态机（FSM），预先算好“每个状态下哪些 token 合法”的索引，运行时查表，平均每步 O(1)。

### 7.2 FSM 够不够：嵌套需要栈

FSM 的状态数是有限的，它记不住任意深的括号嵌套。JSON 允许对象里套数组再套对象，要判断 `}` 合不合法，必须知道当前最内层是对象还是数组。这需要栈，也就是下推自动机（PDA）执行上下文无关文法（CFG）。Outlines 论文也讨论了把索引扩展到 CFG 和 PDA。

实际系统的两种处理：

- 把 JSON schema 展开成正则，嵌套限制在有限深度内。schema 固定时通常够用。
- 直接用 CFG 和 PDA，支持任意嵌套、代码语法和自定义 DSL。代价是每步的合法集合依赖整个栈，没法简单地按“状态”预计算。

### 7.3 SGLang 的压缩 FSM

SGLang 论文第 4 节：已有系统每步只解码一个 token。但很多时候下一段是确定的，例如输出 `{"summary": "` 时，每一步都只有一个合法的下一个 token，却仍要一个 token 一次前向。

做法：在字符级 FSM 上，把“源节点只有一个后继、边上只接受一个字符串”的连续边（singular transition edge）合并成一条压缩边。走到压缩边时，一次前向就能把整段确定的文本写进去（论文叫 Jump Forward）。

坑在分词：压缩边上的文本要转回 token。论文的例子是 `{"summary": "` 按分词器只能切成 `{"`、`summary`、`":`、` "`，不能随便切成 `{"`、`summa`、`ry`……。所以要用原分词器把之前的全部文本和压缩边的文本一起重新分词（retokenization）。论文附录还指出概率扭曲的问题：choice 的选项被压缩后，模型并不知道还有哪些选项。

### 7.4 XGrammar：上下文无关和上下文相关 token

XGrammar（Dong 等，2024-11）用字节级的下推自动机执行 CFG。字节级是为了处理不规则的 token 边界和含半个 UTF-8 字符的 token。它的核心观察是：PDA 匹配一个 token 时，栈的变化分三种：

1. 进入子规则，压栈；
2. 在当前规则内前进，只改栈顶；
3. 走到当前规则末尾，返回父规则，出栈。

前两种只看栈顶节点（当前规则内的位置）就能判定，论文称为**上下文无关 token**；第三种要看整个栈，称为**上下文相关 token**。对 PDA 的每个节点，上下文无关 token 的合法性可以在预处理时算好，以栈顶节点为键缓存起来（自适应 token 掩码缓存）。运行时直接查缓存，只对少量上下文相关 token 跑完整的 PDA。

论文报告的数字（Llama-3.1 词表、JSON 文法）：

- 上下文相关 token 不到 1%（128k 里 1,134 个）；
- 再加上“上下文扩展”（预先算出每个规则结束后可能接的后缀，提前排除不可能的 token），上下文相关 token 减少 90%（1,134 到 120）；
- 掩码缓存按“存接受集合还是拒绝集合”自适应选格式，内存从 160 MB 降到 0.46 MB；
- 用持久化栈管理多条并行栈和历史状态，回滚只改指针，常数时间，方便 Jump Forward 的重新分词和投机解码回退；
- 掩码生成只用 CPU、只依赖之前的 token，可以和 GPU 上的前向重叠，采样前同步；预处理也和 Prefill 重叠。

系统指标（论文报告）：每 token 掩码生成在 JSON schema 和 JSON CFG 上低于 40 µs；CFG 上比已有最好方案快 100 倍以上；和 MLC-LLM 集成后 TPOT 开销接近零。

### 7.5 实现：小文法约束解码

下面实现一个 JSON 子集文法（顶层是对象，值可以是对象、数组、字符串、无前导零的非负整数）的 PDA，用随机 logits 代替模型，每步按掩码把非法 token 置为负无穷。词表里故意放了多字符 token（`"},`、`]}`、`[1`、`1,`），一个 token 可能跨越好几个文法状态。验证：

1. 300 次生成只要采到 EOS，输出都能被 `json.loads` 解析，且顶层是对象；
2. `max_tokens` 截断不归文法管，截得太短照样不完整；
3. 按 XGrammar 的定义分类：枚举所有深度不超过 4 的栈，同一个（模式, 栈顶）下结论随更深的栈变化的 token 就是上下文相关的。结果是只有少数几个、全是包含闭括号的 token。

```python
# 约束解码的最小实现：小 JSON 文法 + 下推自动机（栈）+ 每步 token 掩码
# 文法（JSON 子集）：顶层必须是对象；值 = 对象 | 数组 | 字符串（a-z 和空格）| 非负整数（无前导零）；
# 字符串之外不允许空白
import json
import numpy as np

EOS = "<eos>"
VOCAB = ['{', '}', '[', ']', ',', ':', '"', '{"', '":', '"}', '"]', '},', '],', '}}', ']}', '}]',
         '"a', 'a', 'b', 'ab', ' ', '1', '12', '0', '1,', '1}', '1]', 'name', 'id', '":"', '","',
         '":[', '":{', '"},', 'x"', '[1', '[[', '{{', EOS]
DIG, LET = set("0123456789"), set("abcdefghijklmnopqrstuvwxyz ")


def complete(stack):
    return (stack, "AFTER")                      # 一个值结束了；接下来看栈顶决定能接什么


def step(state, ch):
    """读一个字符。返回新状态，非法返回 None。状态 = (栈, 模式)，栈元素 'O' 对象 / 'A' 数组。"""
    stack, mode = state
    if mode == "START":
        return ((("O",), "OBJ_FIRST") if ch == "{" else None)
    if mode in ("VALUE", "ARR_FIRST"):
        if ch == "{":
            return (stack + ("O",), "OBJ_FIRST")
        if ch == "[":
            return (stack + ("A",), "ARR_FIRST")
        if ch == '"':
            return (stack, "STR")
        if ch == "0":
            return complete(stack)               # JSON 不允许前导零：0 自己就是一个完整整数
        if ch in DIG:
            return (stack, "INT")
        if ch == "]" and mode == "ARR_FIRST":
            return complete(stack[:-1])
        return None
    if mode == "STR":
        return (stack, "STR") if ch in LET else (complete(stack) if ch == '"' else None)
    if mode == "KEY":
        return (stack, "KEY") if ch in LET else ((stack, "COLON") if ch == '"' else None)
    if mode == "INT":
        return (stack, "INT") if ch in DIG else step(complete(stack), ch)   # 整数在非数字处结束
    if mode == "OBJ_FIRST":
        return (stack, "KEY") if ch == '"' else (complete(stack[:-1]) if ch == "}" else None)
    if mode == "OBJ_KEY":
        return (stack, "KEY") if ch == '"' else None
    if mode == "COLON":
        return (stack, "VALUE") if ch == ":" else None
    if mode == "AFTER":
        if not stack:
            return None                          # 顶层对象已闭合，后面什么都不能接
        top = stack[-1]
        if ch == ",":
            return (stack, "OBJ_KEY" if top == "O" else "VALUE")
        if (ch, top) in (("}", "O"), ("]", "A")):
            return complete(stack[:-1])          # 出栈：这里要看栈的更深一层
        return None
    raise ValueError(mode)


def accepting(state):
    return state == ((), "AFTER")


def advance(state, tok):
    for ch in tok:
        state = step(state, ch)
        if state is None:
            return None
    return state


def mask(state):
    """词表里每个 token 现在能不能接。EOS 只在文法完成时合法。"""
    return np.array([accepting(state) if t == EOS else advance(state, t) is not None for t in VOCAB])


def generate(seed, max_tokens=400):
    rng = np.random.default_rng(seed)
    state, out = ((), "START"), []
    closing = np.array([any(c in t for c in '}]"') for t in VOCAB])
    opening = np.array([any(c in t for c in "{[") for t in VOCAB])
    for t in range(max_tokens):
        logits = rng.normal(size=len(VOCAB)) * 2              # 代替模型：随机 logits
        logits += 0.03 * t * closing - 0.03 * t * opening      # 让随机“模型”越写越想收尾
        m = mask(state)
        assert m.any()                                         # 文法保证任何可达状态都有出路
        logits[~m] = -np.inf                                   # 非法 token 置为负无穷
        p = np.exp(logits - logits.max()); p /= p.sum()
        tok = VOCAB[rng.choice(len(VOCAB), p=p)]
        if tok == EOS:
            return "".join(out), True
        out.append(tok)
        state = advance(state, tok)
    return "".join(out), False                                 # 被 max_tokens 截断


# 1) 只要正常结束（采到 EOS），输出一定能被 json.loads 解析，且顶层是对象
lens = []
for seed in range(300):
    text, finished = generate(seed)
    assert finished, seed
    obj = json.loads(text)
    assert isinstance(obj, dict)
    lens.append(len(text))
print("300 条全部合法，平均长度", np.mean(lens), "例子：", generate(7)[0])

# 2) 坑：max_tokens 截断不归文法管。截得太短，约束解码也会给出不完整的 JSON
bad = 0
for seed in range(300):
    text, finished = generate(seed, max_tokens=6)
    if not finished:
        try:
            json.loads(text)
        except json.JSONDecodeError:
            bad += 1
assert bad > 0
print("max_tokens=6 时被截断且不合法：", bad, "/ 300")

# 3) XGrammar 的分类：只看“当前模式 + 栈顶”就能判定的 token 是上下文无关的，可以预先算好缓存；
#    要看更深栈的是上下文相关的，运行时再查。枚举深度 ≤ 4 的所有栈来分类
modes = ["VALUE", "ARR_FIRST", "STR", "KEY", "INT", "OBJ_FIRST", "OBJ_KEY", "COLON", "AFTER"]
stacks = [s for d in range(1, 5) for s in __import__("itertools").product("OA", repeat=d)]
toks = [t for t in VOCAB if t != EOS]
dependent = set()
pairs = 0
for mode in modes:
    for top in "OA":
        same_top = [s for s in stacks if s[-1] == top]
        for t in toks:
            pairs += 1
            verdicts = {advance((s, mode), t) is not None for s in same_top}
            if len(verdicts) > 1:                              # 同一个栈顶，结论随更深的栈变化
                dependent.add((mode, top, t))
print(f"(模式, 栈顶, token) 共 {pairs} 组，上下文相关 {len(dependent)} 组：",
      sorted({t for _, _, t in dependent}))
assert len(dependent) < 0.1 * pairs
# 上下文相关的 token 都包含一次出栈后还要继续匹配的闭括号
assert all(any(c in t for c in "}]") for _, _, t in dependent)
```

运行结果：

```text
300 条全部合法，平均长度 21.31 例子： {"b":"ax"}
max_tokens=6 时被截断且不合法： 194 / 300
(模式, 栈顶, token) 共 684 组，上下文相关 21 组： ['"},', '],', ']}', '},', '}]', '}}']
```

第 3 部分的结果和 XGrammar 的观察一致：上下文相关的都是“闭合一层之后还要继续匹配”的 token。`},` 在对象里闭合了一个子对象之后，`,` 合不合法要看再往上一层是对象还是数组；单独一个 `}` 只看栈顶就够了。这个玩具文法里比例是 3%；真实词表大部分 token 是普通单词片段，只在字符串里合法，比例会更低。

### 7.6 引擎里的用法和代价

vLLM 支持的约束类型有 choice、regex、json（JSON schema）、grammar（EBNF 文法）、structural_tag（在指定标签内遵循 JSON schema）。工具调用上（vLLM Tool Calling 文档，2026-11 查阅）：

- `tool_choice` 为 `required` 或指定函数名时，用结构化输出约束；
- `auto` 模式下，只有至少一个工具设了 `strict: true`，或服务端 `--tool-strict-level` 设为 `function` / `parameter` 时才加文法；否则只用解析器从原始文本里抽取，参数偶尔可能不合法；
- `--tool-strict-level parameter` 等于给每个工具都设 `strict: true`；文档建议严格 schema 里每层对象设 `additionalProperties: false`，所有字段都列为 required。

**代价**：

- 合法不等于正确。vLLM 文档的原话是，指定函数时得到的是“可解析的调用，不一定是高质量的调用”。
- 首次编译语法有延迟，复杂 schema 的第一次请求可能明显变慢，需要缓存编译结果。SGLang 论文的消融也提到：状态机要预处理一次给一批请求复用，每个请求重做预处理吞吐反而更低。
- 约束会改变模型本来的概率分布。schema 设计和模型训练时见过的格式差太多，质量会下降；可以让模型先在约束外思考，再在约束内输出结果。
- 和投机解码、思考模式一起用时，要确认引擎在草稿验证和思考段上对约束的处理方式。草稿 token 也要过文法检查，验证失败回退时文法状态要同步回滚（投机解码见 [03 投机解码](../03-speculative-decoding/README.md)）。

## 8 数值例子：一个 10 轮会话实际要 Prefill 多少

把第 2 节的例子拆细：S = 6,000，每轮模型输出 o = 300、工具结果 r = 1,200（D = 1,500），10 轮，vLLM 式块缓存，B = 16。手算几个关键数：

```text
不复用：Σ P_k = 10 × 6,000 + 1,500 × 45 = 127,500
理想复用（上一轮输出也在缓存）：6,000 + 9 × 1,200 = 16,800
第 2 轮：P_2 = 7,500；上一轮缓存了 6,000 + 299 个 token（最后一个采样出的 token 没喂回去过），
        写满的块是 ⌊6,299 / 16⌋ = 393 块 = 6,288 token → 要 Prefill 7,500 − 6,288 = 1,212
```

下面的代码把每一轮都算出来，再加两种损耗：Qwen3 式模板去掉了每轮输出里 200 个思考 token；第 6 轮开始前，这个会话在等工具时被整体淘汰。

```python
# 数值例子：一个 10 轮会话在 vLLM 式块缓存（B=16）下每轮实际要 Prefill 多少
# 条件：S = 6,000（系统提示 + 工具定义），每轮模型输出 o = 300、工具结果 r = 1,200（含模板包装）
S, o, r, n, B = 6000, 300, 1200, 10, 16


def run(strip_thinking=0, evicted_turns=()):
    """strip_thinking：每轮输出里有多少思考 token 在下一轮被模板去掉（Qwen3 式）。
    evicted_turns：哪几轮开始前，这个会话的缓存在等工具时被整体淘汰。"""
    prompt, cached, per_turn = S, 0, []
    for k in range(1, n + 1):
        if k in evicted_turns:
            cached = 0
        hit = min(cached, (prompt - 1) // B * B)        # 命中按块对齐，且至少留 1 个 token 现算
        per_turn.append(prompt - hit)
        # 这一轮结束后显存里有 KV 的 token：prompt + 输出，最后一个采样出的 token 没被喂回去过
        kv_len = prompt + o - 1
        # 下一轮的前缀和这一轮一致到哪里：去掉思考时，历史在“输出开头”就分叉了
        same = prompt + (o if strip_thinking == 0 else 0)
        cached = min(kv_len, same) // B * B             # 只有写满的块进缓存
        prompt = prompt + (o - strip_thinking) + r
    return per_turn


ideal = S + (n - 1) * r                                # 理想：上一轮输出也已在缓存里
base = run()
print("每轮 Prefill：", base, "合计", sum(base))
assert ideal == 16_800 and 0 < sum(base) - ideal < n * B   # 块对齐每轮最多多算不到一块

think = run(strip_thinking=200)                        # 300 个输出里 200 是思考
print("去掉思考后合计", sum(think))
# 第 2 轮起，每轮要把这一轮保留的 100 个回答 token 重新算一遍，外加块对齐
assert sum(think) - sum(base) >= (n - 1) * (o - 200) - n * B

evict = run(evicted_turns=(6,))                        # 第 6 轮前等工具时被淘汰
print("第 6 轮被淘汰后合计", sum(evict), "，第 6 轮 Prefill", evict[5])
assert evict[5] == S + 5 * (o + r)                     # 那一轮从头算：6,000 + 5 × 1,500
no_reuse = sum(S + (k - 1) * (o + r) for k in range(1, n + 1))
assert no_reuse == 127_500
print(f"相对不复用：基线 {sum(base) / no_reuse:.1%}，去思考 {sum(think) / no_reuse:.1%}，"
      f"淘汰一次 {sum(evict) / no_reuse:.1%}")
```

运行结果：

```text
每轮 Prefill： [6000, 1212, 1208, 1204, 1216, 1212, 1208, 1204, 1216, 1212] 合计 16892
去掉思考后合计 17748
第 6 轮被淘汰后合计 29180 ，第 6 轮 Prefill 13500
相对不复用：基线 13.2%，去思考 13.9%，淘汰一次 22.9%
```

结论：

- 块对齐每轮只多算 4 到 16 个 token，合计比理想值多 92 个，可以忽略。
- 去掉思考让每轮多算 100 个左右（保留下来的回答部分要重算），这个例子里影响也不大；思考很长、回答很长时影响会变大。
- 一次淘汰就让总量从 13.2% 涨到 22.9%，这一轮的 TTFT 也回到不复用的水平。对多轮 Agent 来说，**缓存留不留得住比块粒度重要得多**。这和第 4.4 节模拟的结论一致。

## 9 工程细节与常见坑

**1. 前缀被悄悄改写，命中率低**

- 现象：引擎的前缀命中率远低于按轮数估的值；TTFT 随轮数线性上涨，而不是基本持平。
- 原因：系统提示里有时间戳或请求 ID；工具定义每次按相关度重排，或 JSON 键顺序不固定；历史消息每轮被重新格式化。第 3.2 节说过，改动位置所在的块和之后全部失效。
- 排查：把相邻两轮实际发给引擎的 token id 序列（应用聊天模板之后的）对比，找第一个不同的位置。只比较消息 JSON 不够，模板会改写内容。
- 处理：按第 5.4 节的规则整理 prompt；把“第一个不同 token 的位置 / prompt 长度”做成监控。

**2. 聊天模板去掉思考内容**

- 现象：思考模型上命中率比非思考模型低，每轮新算的 token 比工具结果多出一截。
- 原因：Qwen3 模型卡建议多轮时历史里不保留思考内容，官方模板会自动去掉。上一轮 Decode 时写进 KV 的是“思考 + 回答”，下一轮的前缀里只有回答，从思考开始的位置就分叉了，后面的回答也要重算。
- 排查：看第 8 节那种逐轮 Prefill 量，和工具结果长度对比。
- 处理：接受这部分重算（模型卡的建议有质量上的理由）；思考很长时，评估保留思考是否影响质量，再决定。不要为了命中率私自改模板而不做评测。

**3. 等工具时被淘汰**

- 现象：TTFT 的 P99 很差，且集中在工具慢的那些轮次；负载高的时段更明显。
- 原因：工具执行期间会话的块引用计数为 0，空闲队列短时很快被淘汰（第 5.3 节）。
- 排查：按“距上一轮的间隔”分桶看命中率；看引擎的命中指标和抢占计数是否随负载变化。
- 处理：长上下文开 KV 卸载（vLLM `--kv-offloading-size`、SGLang 分层缓存）；给工具设超时；容量规划时把“等工具的会话”占的 KV 算进去。

**4. 多副本下会话漂移**

- 现象：单实例压测命中率高，上线多副本后掉一大截；扩缩容、发版之后的一段时间 TTFT 变差。
- 原因：负载均衡不看内容（第 5.2 节轮询那一行）；粘滞绑定存在路由器内存里，路由器有多个副本时会话被打散；实例重启后缓存全空。
- 排查：日志里记录每个请求被路由到哪个实例，按会话看是否稳定；按实例看命中率。
- 处理：带稳定的会话 ID，用 `session_aware` 或 `cache_aware`；路由器多副本时前面再加一层粘滞；发版时逐台滚动，接受预热期。

**5. 只看缓存不看负载**

- 现象：开了前缀感知路由后，某一台副本排队很长，其他副本空闲；整体吞吐反而下降。
- 原因：所有请求共享一段很长的系统提示，第一台缓存了它的副本永远匹配最长（第 5.2 节模拟里不均衡到 4 倍）。
- 排查：按副本看排队长度和 prefill token 量。
- 处理：设负载门槛（sgl-router 的 `balanced` 模式）和匹配下限（默认 1,024 token）；共享前缀各副本都会有，让路由只按会话历史的匹配区分。

**6. 最长前缀优先导致饿死**

- 现象：大部分请求很快，少数前缀独特的请求等待时间异常长。
- 原因：缓存感知调度是贪心的，SGLang 论文自己写明会饿死。
- 排查：按“匹配前缀长度”分桶看排队时间。
- 处理：限制重排的窗口或最长等待时间，超时的请求按先来先服务插队。

**7. 多租户共享缓存的风险**

- 现象：安全审查指出可以通过 TTFT 推断别的租户是否发过某段 prompt。
- 原因：命中时 TTFT 明显更短，这是一个时序侧信道。
- 处理：vLLM 按租户设 `cache_salt`，只有同 salt 的请求共享块；多租户下不要用非密码学哈希（文档警告 xxhash 冲突可能导致未定义行为甚至泄露数据）。代价是租户之间不再共享系统提示的缓存。

**8. 跨实例的哈希不一致**

- 现象：自己做跨实例的前缀索引或 KV 迁移时，同样的 prompt 在两台机器上算出的块哈希不同。
- 原因：vLLM 默认的 `sha256` 先用 pickle 序列化，文档说跨 Python 或 vLLM 版本可能不同；用非密码学哈希且没设 `PYTHONHASHSEED` 时，源码会给每个进程随机一个起始种子。
- 处理：需要可复现时用 `sha256_cbor`；各实例的版本保持一致。

**9. 流式工具调用的解析错误**

- 现象：客户端偶尔把 `<tool_call>` 的碎片显示给用户；或者拿到第一个带 name 的分片就执行了工具，参数是空的；并行调用时两个调用的参数串在一起。
- 原因：服务端没扣住半个标签；客户端没等参数到齐；没按 index 聚合。
- 排查：用第 6.3 节的方法，对录下来的原始流随机切分重放，断言结果不变。
- 处理：按 index 聚合，等 `finish_reason` 或下一个 index 出现再执行，执行前按 schema 校验。

**10. 约束解码的输出仍不完整**

- 现象：开了 JSON 约束，偶尔还是解析失败。
- 原因：`max_tokens` 截断不归文法管（第 7.5 节第 2 部分）；`auto` 模式下没设 `strict`，根本没加约束。
- 排查：看失败样本的 `finish_reason` 是不是 `length`；看请求的 `tool_choice` 和工具的 `strict`。
- 处理：给结构化输出留足 `max_tokens`；需要保证合法时用 `required`、指定函数或 `strict: true`。

**11. 第一次请求很慢**

- 现象：新 schema 的第一个请求 TTFT 高很多，之后恢复正常。
- 原因：文法编译和掩码缓存的预处理。
- 处理：启动时预热常用 schema；schema 不要每个请求动态生成；确认引擎缓存了编译结果。

## 10 对比与取舍

### 10.1 前缀缓存的两种实现

| 场景 | 块哈希（vLLM） | 基数树（SGLang） |
|---|---|---|
| 多轮 Agent，前缀一直增长 | 损失每轮不到一块，和基数树差别很小（第 4.4 节：4%） | 同左 |
| 大量短请求共享不对齐的前缀（few-shot、模板化任务） | 块越大损失越多 | 按 token 命中，占优 |
| 树状分叉（多次采样、思维树） | 能共享，靠调度和路由凑在一起 | 树结构天然表达，配合最长前缀优先调度 |
| 跨实例索引 | 块哈希可以直接当键（选可复现的哈希算法） | 路由器维护近似的树 |

### 10.2 等工具期间的 KV

| 做法 | 什么时候选 | 代价 |
|---|---|---|
| 留在显存 | 工具很快（几百毫秒），会话少 | 显存被不干活的会话占着，第 5.3 节例子里钉住 17 个只有 2.8 个在跑 |
| 换出到 CPU | 长上下文、工具慢 | CPU 内存、PCIe 带宽；换入延迟和 KV 大小成正比 |
| 丢掉重算 | 短上下文，或负载低时不会被淘汰 | 下一轮 TTFT 回到不复用水平 |

### 10.3 输出格式的保证程度

| 做法 | 保证 | 代价 |
|---|---|---|
| 只靠提示词 + 解析器（`auto`，无 `strict`） | 不保证，偶尔不合法 | 无额外开销；要有校验和重试 |
| 正则 / FSM（Outlines、压缩 FSM） | 正则能表达的格式；JSON 嵌套展开到有限深度 | 编译开销；正则状态可能很多 |
| CFG / PDA（XGrammar） | 任意嵌套的 JSON、代码、DSL | 实现复杂；首次编译 |
| 以上任何一种 | 都只保证语法，不保证值对、调用的工具对 | 仍要业务校验 |

### 10.4 相关章节

| 问题 | 在哪讲 |
|---|---|
| KV 为什么能复用、每 token 多大、PagedAttention | [01 KV Cache 与显存估算](../01-kv-cache-memory/README.md) |
| 命中前缀后剩下的 Prefill 怎么和 Decode 混跑、要不要 PD 分离 | [02 批处理与 PD 分离](../02-batching-pd/README.md) |
| 约束解码和投机解码一起用 | [03 投机解码](../03-speculative-decoding/README.md) |
| 托管 API 的 prompt caching、压缩对缓存的影响 | [harness/02 上下文压缩](../../harness/02-context-compaction/README.md) |
| 工具参数校验、失败重试 | [harness/04 工具编排](../../harness/04-tool-orchestration/README.md) |
| 整体服务怎么设计 | [设计题：多租户 Agent 推理服务](../../../design/05-agent-serving/README.md) |

## 11 自测清单

- [ ] 说出 Agent 请求和聊天请求在输入、输出、前缀、停顿上的区别
- [ ] 推导不复用 nS + D·n(n−1)/2 和复用 S + (n−1)D，说出比值为什么趋近 2/n，并算出 10 轮例子的 127,500 和 19,500
- [ ] 说出注意力那一项为什么复用前后都高一阶
- [ ] 讲清 vLLM 块哈希的三部分、只缓存满块、逆序放回、LRU 淘汰、cache salt，以及为什么整段命中也要重算最后一块
- [ ] 讲清 RadixAttention 的拆节点、叶子优先淘汰、最长共享前缀优先调度和它的饿死问题
- [ ] 说出两种缓存的命中粒度差别，以及在多轮 Agent 负载上这个差别为什么不大
- [ ] 对比三种路由，说出只看缓存不看负载会怎样，sgl-router 用什么规则兜底
- [ ] 算出等工具期间留在显存、换出、重算的代价，说出各自适合的场景
- [ ] 写出流式工具调用的服务端扣住逻辑和客户端按 index 聚合，说出何时可以执行
- [ ] 从 FSM 到 PDA 讲清约束解码，说出 XGrammar 上下文无关和上下文相关 token 的定义和为什么能预计算
- [ ] 说出约束解码的代价，以及开了约束还会解析失败的两个原因

## 资料

- Zheng 等，[SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)，arXiv 2312.07104，2023-12（第 3 节 RadixAttention、定理 3.1、第 4 节压缩 FSM、附录 A.4 数据并行、附录 B）
- Dong 等，[XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100)，arXiv 2411.15100，2024-11（第 3.1 节自适应 token 掩码缓存、3.2 节上下文扩展、3.3 节持久化栈、3.5 节重叠）
- Willard、Louf，[Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702)，arXiv 2307.09702，2023-07
- Kwon 等，[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，arXiv 2309.06180，2023-09
- vLLM，[Automatic Prefix Caching 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching.html)，2026-11 查阅
- vLLM，[Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)（`--prefix-caching-hash-algo`、`--kv-offloading-size`、`--kv-offloading-backend`），2026-11 查阅
- vLLM 源码，[`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)（最后一个 token 必须重算）、[`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)（NONE_HASH 种子）、[`vllm/config/cache.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py)（`enable_prefix_caching` 默认值），main 分支，2026-11 查阅
- vLLM，[Tool Calling 文档](https://docs.vllm.ai/en/latest/features/tool_calling.html)，2026-11 查阅
- vLLM，[Structured Outputs 文档](https://docs.vllm.ai/en/latest/features/structured_outputs.html)，2026-11 查阅
- SGLang，[sgl-router README](https://github.com/sgl-project/sglang/blob/main/sgl-router/README.md)，2026-11 查阅
- SGLang，[Server Arguments 文档](https://docs.sglang.io/advanced_features/server_arguments.html)（`--page-size`、`--radix-eviction-policy`、分层缓存），2026-11 查阅
- OpenAI，[Chat Completions 流式事件参考](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)，2026-11 查阅
- Anthropic，[Prompt caching 文档](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)，2026-11 查阅
- Qwen 团队，[Qwen3-8B 模型卡](https://huggingface.co/Qwen/Qwen3-8B)，Hugging Face，2025-04
- NVIDIA，[H100 Tensor Core GPU 产品页](https://www.nvidia.com/en-us/data-center/h100/)（PCIe Gen5 128 GB/s、显存带宽 3.35 TB/s），2026-11 查阅
