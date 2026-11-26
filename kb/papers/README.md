# 论文精读

面试里常被追问到原文细节的 13 篇论文。每篇精读都打开原文核对过公式、超参数和章节图表编号，只作阅读材料，不出题。题目在对应章节的 `questions.md` 里。

| 编号 | 论文 | 一句话 | 对应章节 | 相关方向 |
|---|---|---|---|---|
| 01 | [ReAct](01-react.md)（2210.03629） | 把“思考”当成不改变环境的动作，和工具调用交替出现 | [core/02](../core/02-agent-architecture/README.md) | core、harness |
| 02 | [DeepSeekMath / GRPO](02-deepseekmath-grpo.md)（2402.03300） | 用同一问题多个回答的平均奖励做基线，去掉价值模型 | [training/03](../tracks/training/03-grpo-family/README.md) | training |
| 03 | [DeepSeek-R1](03-deepseek-r1.md)（2501.12948） | 基座上直接用规则奖励做 GRPO，推理行为自然涌现；再用多阶段流水线做成可用模型 | [training/03](../tracks/training/03-grpo-family/README.md)、[training/05](../tracks/training/05-reward-verifier/README.md) | training |
| 04 | [DAPO](04-dapo.md)（2503.14476） | Clip-Higher、动态采样、token 级 loss、超长奖励整形四项改进 | [training/03](../tracks/training/03-grpo-family/README.md) | training |
| 05 | [Dr. GRPO](05-dr-grpo.md)（2503.20783） | GRPO 的长度归一化和 std 归一化带来偏差，去掉即可；基座本来就会反思 | [training/03](../tracks/training/03-grpo-family/README.md) | training |
| 06 | [GSPO](06-gspo.md)（2507.18071） | 重要性比率和 clip 改到序列级，和序列级奖励对齐，MoE 上更稳 | [training/03](../tracks/training/03-grpo-family/README.md) | training |
| 07 | [DPO](07-dpo.md)（2305.18290） | 从 KL 约束的奖励最大化推出分类 loss，不用奖励模型也不用 RL | [core/11](../core/11-post-training-basics/README.md)、[training/02](../tracks/training/02-preference-optimization/README.md) | training、core |
| 08 | [Search-R1](08-search-r1.md)（2503.09516） | 用结果奖励和检索 token mask 训练模型多轮调用搜索 | [training/04](../tracks/training/04-agentic-rl/README.md) | training、harness |
| 09 | [SWE-bench](09-swe-bench.md)（2310.06770） | 用真实 GitHub issue 和 PR 构造任务，FAIL_TO_PASS 与 PASS_TO_PASS 测试判对错 | [core/09](../core/09-evaluation/README.md)、[evaluation/04](../tracks/evaluation/04-benchmarks/README.md) | evaluation、harness |
| 10 | [τ-bench](10-tau-bench.md)（2406.12045） | Agent 与模拟用户多轮对话、调 API、守政策，按数据库终态判分，并提出 pass^k | [core/09](../core/09-evaluation/README.md)、[evaluation/04](../tracks/evaluation/04-benchmarks/README.md) | evaluation、harness |
| 11 | [PagedAttention / vLLM](11-pagedattention-vllm.md)（2309.06180） | 借鉴操作系统分页管理 KV Cache，消除预留和碎片，支持按块共享 | [inference/01](../tracks/inference/01-kv-cache-memory/README.md) | inference |
| 12 | [Speculative Decoding](12-speculative-decoding.md)（2211.17192） | 小模型猜、大模型并行验证，接受规则保证输出分布不变 | [inference/03](../tracks/inference/03-speculative-decoding/README.md) | inference |
| 13 | [AWQ](13-awq.md)（2306.00978） | 按激活幅度找显著通道，用等价缩放代替混合精度来保护它们 | [inference/04](../tracks/inference/04-quantization/README.md) | inference |

## 怎么读

1. **先读对应章节**。章节讲清这个问题在面试里怎么问、几种做法的对比和代价，是答题的主线。
2. **再读精读**。精读比章节细：完整的公式和符号、超参数、消融结论、论文自己承认的局限，以及面试官常见的追问。每篇的第 5 节是追问和回答要点，适合自测。
3. **最后有时间再读原文**。每篇精读的第 6 节列了原文里最值得看的章节、图和表，按那个顺序读，比从头读快得多。

精读里不写任何模型在 Benchmark 上的分数和排名；系统类论文的加速比和显存节省都标明“论文报告”。GRPO 系四篇建议按 02 → 05 → 04 → 06 的顺序读，和 [training/03](../tracks/training/03-grpo-family/README.md) 第 6 节“问题 → 修法”的讲法一致。
