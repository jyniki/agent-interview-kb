# 手撕题怎么用

> 题目：[questions.md](questions.md)，共 20 道。`code-model-` 为模型侧，`code-eng-` 为工程侧。

## 怎么练

1. 只看标题，自己写。标题里给了要实现什么、输入输出形状或函数签名。先写骨架，再补边界条件，限时 15 到 25 分钟。
2. 写完再对照“一句话答案”和“要点”。要点里列的是关键步骤、形状和边界条件，逐条检查自己的代码有没有覆盖。
3. 写一个小测试验证。模型侧的题大多有一个现成的检验方法，例如 RoPE 检查内积只依赖相对距离，KV Cache 解码检查和整段前向结果一致，GAE 检查 λ = 0 和 λ = 1 的特例。
4. 最后过一遍“追问链”和“易错点”。面试官常在你写完后顺着追问改需求，比如把 MHA 改成 GQA、给 LRU 加 TTL。

题目只给要点和关键几行伪代码，不给完整实现。写不出来时，先回到对应章节把原理看懂，再回来写。

## 推荐顺序

先打基础，再写组合题。同一行内的题互相独立，可以任选顺序。

| 阶段 | 题目 | 说明 |
|---|---|---|
| 1 张量基本功 | code-model-006 RMSNorm、code-model-001 MHA、code-model-012 GQA 头复制 | 练形状变换和广播 |
| 2 位置与解码 | code-model-005 RoPE、code-model-008 top-k / top-p 采样、code-model-007 KV Cache 解码循环 | 007 会用到 001、005、008 |
| 3 训练 loss | code-model-013 loss mask 交叉熵、code-model-003 LoRA、code-model-004 DPO | 先把 SFT 的 mask 写对 |
| 4 RL | code-model-010 GAE、code-model-009 PPO loss、code-model-002 GRPO | 先 GAE 再 PPO，最后对比 GRPO 省掉了什么 |
| 5 分词 | code-model-011 BPE 训练 | 纯 Python 字典和计数 |
| 6 工程基础 | code-eng-004 LRU、code-eng-002 令牌桶、code-eng-003 指数退避重试 | 数据结构和并发 |
| 7 Agent 工程 | code-eng-001 Agent Loop、code-eng-005 流式工具调用拼接、code-eng-006 检索选工具、code-eng-007 语义缓存 | 005 是 001 的流式版本 |

只有一两天准备时间，优先练 001、005、007、008、009、010、013 和 eng-001、eng-005。

## 每题对应的章节

| 题号 | 题目 | 先读 |
|---|---|---|
| code-model-001 | Multi-Head Attention | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md) |
| code-model-002 | GRPO 组内优势与 loss | [GRPO 及其变体](../tracks/training/03-grpo-family/README.md) |
| code-model-003 | LoRA 线性层 | [后训练入门](../core/11-post-training-basics/README.md) |
| code-model-004 | DPO loss | [后训练入门](../core/11-post-training-basics/README.md) |
| code-model-005 | RoPE | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md) |
| code-model-006 | RMSNorm | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md)（章节未展开归一化层，以题目要点为准） |
| code-model-007 | 带 KV Cache 的解码循环 | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md)、[KV Cache 与显存估算](../tracks/inference/01-kv-cache-memory/README.md) |
| code-model-008 | top-k / top-p 采样 | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md) |
| code-model-009 | PPO clip loss 与价值 loss | [强化学习入门](../core/12-rl-basics/README.md) |
| code-model-010 | GAE | [强化学习入门](../core/12-rl-basics/README.md) |
| code-model-011 | BPE 训练 | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md) |
| code-model-012 | GQA 的 KV 头复制 | [LLM 与 Transformer 基础](../core/01-llm-foundations/README.md)、[KV Cache 与显存估算](../tracks/inference/01-kv-cache-memory/README.md) |
| code-model-013 | 带 loss mask 的交叉熵 | [后训练入门](../core/11-post-training-basics/README.md) |
| code-eng-001 | 最小 Agent Loop | [Agent 架构与设计模式](../core/02-agent-architecture/README.md) |
| code-eng-002 | 令牌桶限流器 | [模型网关、流式、缓存与熔断降级](../tracks/harness/05-gateway-reliability/README.md) |
| code-eng-003 | 带抖动的指数退避重试 | [工具注册、路由、并发与错误恢复](../tracks/harness/04-tool-orchestration/README.md)、[模型网关、流式、缓存与熔断降级](../tracks/harness/05-gateway-reliability/README.md) |
| code-eng-004 | LRU 缓存 | [面向 Agent 的推理](../tracks/inference/05-agent-serving/README.md)（前缀缓存按 LRU 淘汰） |
| code-eng-005 | 流式工具调用增量拼接 | [面向 Agent 的推理](../tracks/inference/05-agent-serving/README.md) |
| code-eng-006 | 按工具描述检索选工具 | [工具注册、路由、并发与错误恢复](../tracks/harness/04-tool-orchestration/README.md) |
| code-eng-007 | 语义缓存 | [模型网关、流式、缓存与熔断降级](../tracks/harness/05-gateway-reliability/README.md) |
