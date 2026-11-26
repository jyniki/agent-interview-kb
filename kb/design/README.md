# 系统设计整题

设计题考的是把前面各章的知识组合起来：先澄清和估算，再给出最简单可行的架构，然后挑两三处深挖，最后讲评测与上线。答题框架见 [core/13 Agent 系统设计方法](../core/13-system-design/README.md)，它用一个售后客服 Agent 演示了完整过程，这里不再重复客服题。

题目清单和对应方向在 [design.json](design.json)。学习教练会在计划末尾排上和你所选方向相关的设计题，复习日也会推荐一道。

| 题目 | 主要方向 | 重点考什么 |
|---|---|---|
| [01 企业知识库问答 Agent](01-enterprise-kb-qa/README.md) | harness、evaluation | 路由、权限过滤检索、引用校验、评测集构造 |
| [02 仓库级 Coding Agent](02-coding-agent/README.md) | harness、evaluation | 代码上下文、沙箱、测试验证、并行工作区 |
| [03 Deep Research 研究助手](03-deep-research/README.md) | harness、evaluation | 研究计划、并行子 Agent、停止条件、事实核查 |
| [04 Agent 评测平台](04-agent-eval-platform/README.md) | evaluation、harness | 任务与环境管理、评分器、显著性、回归门禁 |
| [05 多租户 Agent 推理服务](05-agent-serving/README.md) | inference、harness | 前缀缓存、调度与 SLO、会话粘滞、容量估算 |
| [06 端云协同的手机个人助理](06-on-device-assistant/README.md) | inference、harness | 端侧模型选型、端云路由、GUI 操作、隐私 |
| [07 Agentic RL 训练平台](07-agentic-rl-platform/README.md) | training、inference | rollout 与训练协同、环境规模化、奖励、稳定性 |
| [08 让开源模型学会调用企业内部工具](08-tool-agent-post-training/README.md) | training、evaluation | 数据合成、SFT 到 RL、奖励设计、上线评测 |

## 每道题的格式

`README.md`：

1. **题面**：面试官会怎么问，一两句话。
2. **澄清**：该问的问题，以及本题采用的假设（数字都标明是假设）。
3. **估算**：流量、token、成本或显存，写出算式。
4. **架构**：最多一张 mermaid 图，加模块职责表。先给最简单可行的版本，再说什么情况下升级。
5. **深挖**：两到三个关键点，每个按“问题 → 方案 → 代价”讲，链接到对应章节。
6. **评测与上线**：离线评测、灰度、线上指标、回滚。
7. **常见追问**：面试官最可能追的方向，各一两句回答。
8. **评分要点**：面试官心里的打分表，学习教练用它给白板作答打分。
9. **资料**。

`questions.md`：3-5 道追问题，格式和章节题目一样，题号前缀 `design-`。
