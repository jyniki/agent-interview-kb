# 题目：连续批处理与 Prefill / Decode 分离

> 对应知识章：[README.md](README.md)。先闭卷作答，再看答案。

## infer-batch-001 连续批处理比静态批好在哪
- level: 1
- tags: [inference]

**一句话答案**：静态批要等整批最长的请求结束才返回、才接新请求；连续批处理按迭代调度，每生成一步就让结束的请求离开、让新请求加入，GPU 不再陪长请求空转，新请求也不用排整批的队。

**要点**：
- 静态批的两种浪费：早结束的请求空占位置；新请求等整批结束。
- Orca 提出迭代级调度：调度器每次只让模型跑一次迭代。
- Prefill 和 Decode 请求 token 数不同，Orca 用 selective batching：非 Attention 算子拉平成 `[ΣL, H]` 一起算，Attention 按请求拆开。
- Orca 论文报告：GPT-3 175B 上同等延迟下吞吐是 FasterTransformer 的 36.9 倍。

**追问链**：
1. 为什么 Attention 拆开算损失不大？→ Attention 不涉及模型参数，拼 batch 也省不了读权重；省读权重的是 Linear 这类算子，它们仍然拼在一起。
2. 新请求随时加入，显存怎么办？→ 需要按需分配 KV，例如 PagedAttention；显存不够时要抢占正在跑的请求。
3. 被抢占的请求怎么恢复？→ 把 KV 换到 CPU 再换回，或丢掉 KV 把已生成内容拼回 prompt 重算；长上下文重算很贵。

**易错点**：
- 把连续批处理说成“batch 越大越好”。batch 受 KV 显存限制，而且 batch 大了每步变慢，TPOT 会变长。

## infer-batch-002 现场估算：稠密模型 BF16 Decode，GPU 稠密算力约 990 TFLOPS、显存带宽 3.35 TB/s（H100 SXM），batch 多大才接近吃满算力
- level: 2
- tags: [inference]

**一句话答案**：稠密模型 Decode 的算术强度约等于 batch；H100 稠密 BF16 约 990 TFLOPS、带宽 3.35 TB/s，比值约 295，batch 约 300 才接近吃满，读 KV 更难。

**要点**：
- 稠密模型每 token 约 2N FLOPs，一步读 2N 字节权重。
- 计算过程：
  - 一步计算 ≈ 2 × N × B，读权重 ≈ 2 × N 字节，强度 ≈ B
  - 官方 BF16 1,979 TFLOPS 标注 with sparsity，稠密取一半 ≈ 990 TFLOPS
  - 990e12 / 3.35e12 ≈ 295
- Prefill 2,000 token 的强度约 2,000，远超 295，受算力限制。
- 每个请求的 KV 都要单独读，batch 越大 KV 读取越多，强度被拉低。

**追问链**：
1. 用 8B 模型算 Prefill 2,000 token 的时间下界？→ 2 × 8.2e9 × 2,000 / 990e12 ≈ 33 ms，按满利用率算，实际更慢。
2. batch 受什么限制到不了 300？→ KV 显存。按[第 01 章](../01-kv-cache-memory/README.md)，Qwen3-8B 在 80 GB 卡上只能放约 35 万 token 的 KV，300 个请求平均只能各约 1,100 token。
3. 那增加 batch 之外还能怎么提高 Decode 效率？→ 减少每步读的字节：权重量化、KV 量化、GQA；或者一步验证多个 token，即投机解码。

**易错点**：
- 直接用产品页带 sparsity 的 1,979 TFLOPS 算，比值会高估一倍。

## infer-batch-003 TTFT、TPOT 和 goodput 分别是什么，调度怎么影响它们
- level: 2
- tags: [inference, evaluation]

**一句话答案**：TTFT 是到首个 token 的时间，主要看排队和 Prefill；TPOT 是之后每个 token 的间隔，主要看 Decode 单步；goodput 是两者都达标的最大请求速率。

**要点**：
- DistServe 用 goodput 代替单纯吞吐。
- 每步 token 预算（vLLM 的 `max_num_batched_tokens`）是主要旋钮。
- 预算小：Decode 受 Prefill 干扰少，TPOT 稳，但长 prompt 的 TTFT 变长。
- 预算大：Prefill 推得快，TTFT 短，吞吐高，但 TPOT 波动大。
- vLLM V1 调度先放全部 Decode，再用剩余预算放 Prefill。

**追问链**：
1. Agent 场景更在乎哪个？→ 两个都在乎；Agent 每轮输出短、轮数多，TTFT 会被乘以轮数，常常比 TPOT 更影响端到端时延。
2. 只看平均值够吗？→ 不够，要看 P90/P99；长 Prefill 造成的卡顿集中在尾部。

**易错点**：
- 用“总吞吐”评价服务好坏。吞吐很高但大量请求超出延迟目标，goodput 很低。

## infer-batch-004 Chunked Prefill 解决什么问题，代价是什么
- level: 2
- tags: [inference]

**一句话答案**：把长 Prefill 切块，每步放一块、其余给 Decode，长 Prefill 不再卡住 Decode；代价是总计算不减，后面的块要重读前面的 KV，块太小时长 prompt 的 TTFT 变长。

**要点**：
- 块大小相近，每步放一块 Prefill，其余位置填 Decode。
- Sarathi 的 decode-maximal batching：一个 Prefill 块吃算力，Decode 搭便车共用权重读取。
- Sarathi-Serve 的 stall-free 调度：加入新请求时不暂停正在进行的 Decode。
- 论文报告：Sarathi-Serve 在 Mistral-7B、单张 A100、尾延迟约束下服务容量是 vLLM 的 2.6 倍。
- vLLM V1 默认开启。

**追问链**：
1. 为什么 Prefill 和 Decode 混在一步里更省？→ Prefill 受算力限制，Decode 受带宽限制，放一起两种资源都更满。
2. 块大小怎么定？→ 由每步 token 预算决定；看 TPOT 稳定性和长 prompt TTFT 哪个更重要，压测决定。
3. 它能替代 PD 分离吗？→ 不能完全替代。两阶段仍共用一套并行配置和扩缩，干扰只是变小。

**易错点**：
- 说“Chunked Prefill 减少了 Prefill 计算量”。它只改变计算的时间分布。

## infer-batch-005 PD 分离为什么能提升 goodput，什么时候不值得
- level: 3
- tags: [inference]

**一句话答案**：两阶段瓶颈不同，分开后各自选并行方式和扩缩，长 Prefill 不再打断 Decode，TTFT、TPOT 可同时达标；代价是多一次 KV 传输和两套运维；负载低、prompt 短、前缀命中高时不值。

**要点**：
- 跨节点带宽低时 KV 传输太慢，也不值得。
- DistServe：按 TTFT、TPOT 目标为两阶段分别分配资源和并行方式，按带宽放置；论文报告相比当时最好的系统多服务 7.4 倍请求或满足严 12.6 倍的 SLO。
- Splitwise：两阶段可用不同型号的机器，Decode 可用更低功耗、更低成本的硬件。
- Mooncake：以 KV 为中心，用 CPU、DRAM、SSD 组成分布式 KV 池，过载时提前拒绝。
- Prefill 实例可以用更大 TP 缩短 TTFT；Decode 实例可以开大 batch 提吞吐。

**追问链**：
1. KV 传输会成为瓶颈吗？→ 会。DistServe 举例：OPT-66B 每条 512 token 请求 KV 约 1.13 GB，10 请求/秒要约 90 Gb/s；所以优先同节点内走 NVLink。
2. 能不能把传输藏起来？→ 可以和 Prefill 按层流水重叠，边算边传；但仍占网络带宽。
3. 两边各配多少卡？→ 按目标负载下 Prefill 和 Decode 各自的单卡处理能力和 SLO 求解，DistServe 就是做这种搜索；负载结构变了要重新配。
4. Agent 多轮场景怎么看？→ 前缀缓存命中后每轮只需 Prefill 新增的工具结果，Prefill 很轻；先做缓存和会话粘滞，再评估分离。

**易错点**：
- 说“PD 分离一定更快”。它优化的是有 SLO 约束时的 goodput，低负载时可能更慢。

## infer-batch-006 现场估算：PD 分离时，每 token KV 144 KiB（Qwen3-8B）的 32K 请求要传多少 KV，在 100 Gb/s 和 400 Gb/s 网络上各要多久
- level: 2
- tags: [inference]

**一句话答案**：以 Qwen3-8B 计，每 token KV 约 144 KiB，32K 约 4.83 GB；100 Gb/s 约 0.39 s，400 Gb/s 约 0.10 s，不重叠就直接加到 TTFT 上。

**要点**：
- 计算过程：
  - 每 token：2 × 36 层 × 8 个 KV 头 × 128 × 2 B = 147,456 B
  - 32K：147,456 × 32,768 ≈ 4.83 GB
  - 100 Gb/s = 12.5 GB/s：4.83 / 12.5 ≈ 0.39 s
  - 400 Gb/s = 50 GB/s：4.83 / 50 ≈ 0.10 s
- 带宽按理想值算，没算协议开销。
- 每秒请求数 × 每条 KV 大小 = 需要的持续带宽。

**追问链**：
1. 怎么降低传输量？→ KV 量化（FP8 减半）、GQA/MLA 这类结构、只传前缀缓存未命中的部分。
2. 8K 请求呢？→ 约 1.21 GB，400 Gb/s 下约 24 ms，影响小得多；长上下文请求才是问题。

**易错点**：
- 把 Gb/s 当 GB/s，差 8 倍。

## infer-batch-007 为什么平均 TPOT 达标时用户仍会感到输出卡顿，SLO 应该怎么定
- level: 2
- tags: [inference, evaluation]

**一句话答案**：TPOT 是每个请求的平均间隔，长 Prefill 插队造成的一次停顿会被几百个正常间隔平均掉；卡顿只在 ITL 的尾部可见，SLO 应定在 P99 ITL 或每个请求的最长间隔上。

**要点**：
- vLLM 压测脚本的 TPOT = (总时延 − TTFT) / (输出长度 − 1)；ITL 记录每一个相邻间隔。
- 停顿来源：连续批处理里一整段长 Prefill 和 Decode 放在同一步，这一步所有 Decode 都要等它，Sarathi-Serve 叫 generation stall。
- 本章模拟器（Qwen3-8B、H100 roofline、20% 请求带 8K prompt）：Orca 式连续批处理 TPOT p99 约 8 ms，每个请求最长停顿的 p99 约 145 ms。
- 同一模拟器里，SLO 用平均 TPOT 时 Chunked Prefill 和整段 Prefill 的 goodput 差得不多；改用最长停顿后差一个数量级以上。
- Sarathi-Serve 的 SLO 定在 P99 TBT 上。

**追问链**：
1. 怎么定位停顿是谁造成的？→ 把停顿时刻和同一时刻进入的长 prompt 对上；看 `vllm:inter_token_latency_seconds` 直方图的尾部。
2. 怎么消掉？→ 开 Chunked Prefill，把每步 token 预算设在拐点附近；或者做 PD 分离，让 Decode 实例上不出现 Prefill。
3. 为什么 vLLM 的 `--goodput tpot:…` 不够？→ 它用的是每个请求的平均 TPOT，同样会把停顿平均掉；而且它是一个负载点上每秒的达标请求数，不是最大可承受速率。

**易错点**：
- 只看 TPOT 的 P99 就说“没有卡顿”。P99 是在请求之间取的，每个请求内部已经平均过一次。

## infer-batch-008 每步 token 预算怎样同时影响 TTFT 和 TPOT，为什么存在一个拐点
- level: 3
- tags: [inference]

**一句话答案**：一步耗时取算力和带宽下界中较大的那个。预算低于拐点时一步受带宽限制，多塞 Prefill 不加时间，TTFT 随预算反比下降；超过拐点后一步随预算线性变长，TTFT 不再缩短，TPOT 变差。

**要点**：
- 混跑一步：t ≈ max(2N·(b + c) / F, (W + E_kv·Σctx) / BW)，b 是 Decode 数，c 是 Prefill 块大小，预算 C = b + c。
- 拐点：b + c* ≈ (F / BW)·(1 + E_kv·Σctx / W)，H100 上 F / BW ≈ 295；KV 读取越多，拐点越高。
- 长 prompt 要 ⌈P / (C − b)⌉ 步做完，Decode 的 batch 大时留给 Prefill 的预算更少。
- 本章推算（Qwen3-8B、63 条 Decode 各 1,000 上下文）：拐点约 465；预算 128 时 8K prompt 的 TTFT 约 1 s，512 时约 0.18 s，整段约 0.16 s。
- vLLM 文档：预算小 ITL 好，预算大 TTFT 好，追求吞吐建议 8192 以上。

**追问链**：
1. 实测的拐点为什么比 roofline 估的高？→ 有 kernel 启动等固定开销；Sarathi-Serve 报告高张量并行度下要 500 到 600 token 才进入算力受限。
2. 预算还要考虑什么？→ tile 量化（块加 Decode 数落在 tile 整数倍上），流水线并行时每步计算量要均匀，以避免气泡。
3. 负载变了预算要不要变？→ 拐点随 Decode batch 和上下文长度变化，按目标负载压测定，必要时分时段配置。

**易错点**：
- 以为预算越小越好。低于拐点后 Decode 并没有变快，只是长 prompt 的 TTFT 白白变长。

## infer-batch-009 现场估算：Qwen3-8B（每 token KV 144 KiB、8.2B 参数）在 H100（稠密 990 TFLOPS）上做 PD 分离，跨节点链路至少多快才能用逐层传输把 KV 传输藏住
- level: 3
- tags: [inference]

**一句话答案**：每层传输要快于每层计算。Prefill 时间至少为 2N·P/F，KV 是 E_kv·P，P 约掉后带宽门槛是 E_kv·F/(2N)，约 71 Gb/s。低于它，KV 就产出得比传出去快。

**要点**：
- 计算过程：
  - 满算力时每秒 Prefill 的 token 数 = 990e12 / (2 × 8.2e9) ≈ 6.0 万
  - 每秒产出 KV = 6.0e4 × 147,456 B ≈ 8.9 GB/s ≈ 71 Gb/s
- 满足门槛时露在外面的只有最后一层的传输：32K prompt 在 100 Gb/s 上只露约 11 ms，一次性传则要 0.39 s。
- 这是用满算力下界推的；实际 Prefill 更慢，门槛会更低。没算注意力，长 prompt 的 Prefill 更慢，也会让门槛更低。
- Splitwise 论文报告：逐层传输后露出的传输时间约为常数，A100 约 8 ms、H100 约 5 ms。

**追问链**：
1. 换成 MHA（32 个 KV 头）呢？→ KV 是 4 倍，门槛约 284 Gb/s。GQA、MLA 直接降低了 PD 分离对网络的要求。
2. 带宽不够怎么办？→ P、D 放同一节点走 NVLink（DistServe 的低亲和放置）；KV 用 FP8 传；只传前缀缓存未命中的部分；或者不分离。
3. 逐层传输有什么代价？→ 每层都要同步，小 prompt 时同步开销可能超过收益，所以 Splitwise 对小 prompt 仍用一次性传输。

**易错点**：
- 只算单条请求的传输时间。逐层流水之后要看的是持续速率能不能跟上 Prefill 的产出。

## infer-batch-010 用 M/D/1 模型解释：Prefill 实例的利用率从 50% 升到 90% 时 TTFT 怎么变
- level: 2
- tags: [inference]

**一句话答案**：平均 TTFT = D + D·ρ/(2(1−ρ))，D 是 Prefill 时间，ρ 是利用率。排队从 0.5D 涨到 4.5D，TTFT 从 1.5D 到 5.5D，有延迟目标就不能跑满。

**要点**：
- 假设：泊松到达、每个请求的 Prefill 时间固定为 D、一次只做一个、先来先服务；ρ = R·D < 1。
- 排队项只取决于 ρ：0.5 时 0.5D，0.9 时 4.5D，0.95 时 9.5D。
- DistServe 用同一模型比较两卡并行：张量并行把执行时间缩为 D/K（1 < K < 2），低负载时更好；流水线并行容量翻倍，高负载时更好。
- Little 定律：系统内平均请求数 = 到达率 × 平均停留时间；用在 Decode 上，batch ≈ 到达率 × 输出长度 × TPOT。

**追问链**：
1. 真实负载比泊松更突发会怎样？→ 到达方差更大，同样利用率下排队更长；要留更多余量或做限流。
2. 加副本有什么用？→ 把 R 摊到 N 份，ρ 降为 R·D/N，排队项随之下降；代价是每份都要存权重。
3. 过载时怎么办？→ 提前拒绝或降级，不要接了再超时；Mooncake 在请求到达时就按 Prefill 和 Decode 两侧较重的负载决定是否接收。

**易错点**：
- 以为单个请求的 Prefill 时间没变，TTFT 就不会变。高利用率下 TTFT 的增长几乎全部来自排队。
