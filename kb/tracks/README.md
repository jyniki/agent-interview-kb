# 垂类方向是怎么定的

截至 2026-11。垂类方向按“招聘需求里反复出现、面试会单独考”来定，而不是按技术领域平均切分。这一页记录依据，方便以后按新数据调整。

## 依据

1. **国内 Agent 算法岗 JD 关键词统计**。2026-10 至 2026-11 收集的上海、杭州约 36 份有正文的 Agent 算法岗 JD（来自公司招聘页、招聘平台和第三方聚合站，多数没有在官方站逐条复核），人工合并粗略计数。前几位：

   | 要求词 | 约出现次数 |
   |---|---|
   | 工具调用 / Function Calling | 23 |
   | 强化学习 / Agentic RL | 22 |
   | SFT | 18 |
   | 任务规划 / 多步推理 | 16 |
   | 评测 / 评估体系 | 14 |
   | 记忆 | 14 |
   | 多 Agent | 11 |
   | RAG | 11 |
   | GRPO / DPO / PPO / RLHF | 10 |
   | Harness / 上下文工程 | 9 |
   | Reward / Verifier / 过程奖励 | 约 8 |
   | 数据构建 / 合成 | 约 8 |
   | 用户模拟器 / 环境构建 | 约 6 |
   | Search / Deep Research | 约 6 |

   同一批 JD 里能分出的岗位类型：Agent 应用算法、Agent 后训练 / Agentic RL、Agent 评测、Agentic Search、Agent 工程（Harness）。

2. **公开面经整理**：[2026 国内 Agent 面经指南](https://keepkeen.github.io/blog/china-agent-interview-guide-2025-2026/)。它按岗位把 Agent 面试分成四类：应用 / 全栈、算法 / 后训练、Runtime / Harness / Infra、Coding / GUI / Browser / Computer-Use Agent，并认为 Runtime / Harness 是 2026 年增长最明显的方向。它引用的开源面经数据集（767 篇帖子筛出 248 篇，948 道题合并为 638 道）里，最高频的主题是 Agent 评测、指标构建、多 Agent、上下文溢出、工具参数幻觉、记忆。作者说明这批样本偏校招、实习和一面。

3. **开源题库的组织方式**：[agent-interview-hub](https://github.com/Zchary1106/agent-interview-hub) 按公司和知识层次组织，它的岗位能力矩阵分为应用、算法、Runtime、Coding Agent、平台五类。

## 由此得出的方向

| 方向 | 为什么单列 | 证据强度 |
|---|---|---|
| training | JD 里 RL、SFT、GRPO 系、reward、数据合成、环境合计出现次数最多；有大量专门的后训练岗 | 最强 |
| harness | 有专门的 Harness / Runtime 岗；面试指南认为增长最快；上下文压缩、沙箱、长任务终止是多家公司的面试侧重 | 强 |
| evaluation | 有专门的 Agentic 评估岗；在面经数据集里是最高频主题；对其他方向也是加分项 | 强 |
| inference | 不是独立的 Agent 岗位方向，但作为算法基础（KV Cache、显存、量化、Prefill / Decode）和 Runtime 内容出现；模型公司和端侧岗要求单独准备 | 中（作为配套） |

## 考虑过、没有单列的方向

- **Agentic Search**：有独立的 AI 搜索、搜推 Agent 岗位（JD 里 RAG 约 11 次、Search / Deep Research 约 6 次），证据中等。没有单列：检索基础放在通用层 [07 RAG、搜索与 Deep Research](../core/07-rag-search/README.md)，Search Agent 的训练放在 [training/04](training/04-agentic-rl/README.md)。
- **GUI / 端侧 Agent**：面试指南把 GUI / Computer-Use 列为独立方向，但可核实的 2026 样本少。没有单列：Computer-Use 的交互、校验和授权问题并入 [harness/07](harness/07-coding-agent/README.md)，端侧模型选型和端云协同并入 [inference/06](inference/06-on-device/README.md)。

以后这两类岗位的样本多了，可以再拆出来。

## 和最初设想的差别

最初按技术领域设想了四个方向：harness、training、端侧、推理 infra。按上面的证据做了以下调整：

- **新增 evaluation**：在 JD 和面经里都是独立、高频的需求，原来放在通用层，讲不深。
- **端侧不单列**：只做“端侧”的 Agent 岗位样本很少，相关内容分到 harness 和 inference。
- **推理 infra 收窄为 inference**：专职推理引擎岗（CUDA kernel、引擎开发）是另一个职业方向，不在本库范围。这里只保留 Agent 岗位会考到的推理知识。
- **Coding Agent 并入 harness**：面试内容（上下文选择、工作区隔离、回滚、验证）和 Harness 高度重叠。
- **记忆、多 Agent、规划留在通用层**：它们在 JD 里出现次数高，但所有方向都会问，没有对应的独立岗位。

## 已知局限

- JD 样本只覆盖上海、杭州，且大部分来自第三方聚合站，数量也只有几十份。
- 面经样本偏校招和一面，社招深挖轮的题目覆盖不足。
- Computer-Use 和端侧部署的题目可核实样本少，补内容时请优先找可核实的面经。

欢迎提交新的统计数据来修正这些方向。
