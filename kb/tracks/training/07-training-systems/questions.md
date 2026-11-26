# 题目：RL 训练框架、并行与显存估算

> 对应知识章：[README.md](README.md)。先闭卷作答，再看答案。“训练为什么比推理多 8 倍显存”的入门题见 [inference/01 的 infer-kv-006](../../inference/01-kv-cache-memory/questions.md#infer-kv-006-推理显存和训练显存差在哪)，这里往下问。

## train-sys-001 现场估算：14B 模型（40 层、hidden 5120、40 头、词表 10 万）在 16 张 80 GB 卡上纯数据并行、混合精度 Adam 全参训练，序列 8192、micro-batch 1，每层激活按 sbh(34 + 5as/h) 字节算，显存够不够
- level: 3
- tags: [training, inference]

**一句话答案**：不切分放不下（224 GB）；ZeRO-3 每卡约 14 GB，ZeRO-2 约 40 GB；FlashAttention 下激活约 57 GB。ZeRO-3 勉强够，加重计算才稳。

**要点**：
- ZeRO-2 每卡约 40 GB，加上激活放不下，必须配重计算。
- 假设配置：Ψ = 14e9，L = 40，h = 5120，a = 40，s = 8192，b = 1，16 卡纯数据并行，混合精度 Adam，词表 V = 100000。
- 模型状态（ZeRO 口径，每参数 16 字节）：

```
不切分： 16Ψ           = 224 GB
ZeRO-1：4Ψ + 12Ψ/16    = 56 + 10.5   = 66.5 GB
ZeRO-2：2Ψ + 14Ψ/16    = 28 + 12.25  = 40.25 GB
ZeRO-3：16Ψ/16         = 14 GB
```

- 激活（Korthikanti 等，每层 `sbh(34 + 5as/h)` 字节）：

```
sbh = 8192 × 1 × 5120 = 41,943,040
不去注意力矩阵：sbh × (34 + 5 × 40 × 8192 / 5120) = sbh × 354 ≈ 14.8 GB / 层
FlashAttention：sbh × 34 ≈ 1.43 GB / 层，× 40 层 ≈ 57 GB
全量重计算：  2 × sbh × 40 ≈ 3.4 GB，再加反向时一层的完整激活 1.43 GB ≈ 4.8 GB
logits（FP32）：8192 × 100000 × 4 ≈ 3.3 GB
```

- 合计：ZeRO-3 + FlashAttention ≈ 14 + 57 + 3.3 ≈ 74 GB，再加通信缓冲和碎片，80 GB 很紧；开全量重计算后 ≈ 14 + 4.8 + 3.3 ≈ 22 GB，宽松。ZeRO-2 + 重计算 ≈ 40 + 4.8 + 3.3 ≈ 48 GB，也可行。

**追问链**：
1. 选 ZeRO-2 加重计算，还是 ZeRO-3 不重计算？→ ZeRO-2 不增加通信；重计算多约一次前向。ZeRO-3 通信是基线的 1.5 倍。卡间带宽高时两者都行，要实测吞吐。
2. 激活公式里的 34 和 5as/h 分别是什么？→ 34 是线性层、LayerNorm、dropout 等的中间结果；5as/h 来自 QKᵀ、softmax、dropout 和对 V 的注意力，和 s² 成正比。FlashAttention 不存注意力矩阵，所以这一项去掉。
3. 如果是 GRPO 的 RL 训练，还要加什么？→ reference 模型的推理权重 2Ψ（可分片或卸载）、rollout 引擎的权重和 KV Cache；PPO 还要再加一个 critic 的 16Ψ。

**易错点**：
- 只算 2Ψ 的参数就说“28 GB 一张卡够了”。
- 忘了激活和 logits，或把激活当成和参数量成正比。

## train-sys-002 ZeRO-1、ZeRO-2、ZeRO-3 各切了什么，通信量怎么变
- level: 2
- tags: [training]

**一句话答案**：ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 再切参数；前两级通信量和普通数据并行一样是 2Ψ，ZeRO-3 多了收集参数的 all-gather，变成 3Ψ，是 1.5 倍。

**要点**：
- 每卡模型状态：`4Ψ + 12Ψ/N_d`、`2Ψ + 14Ψ/N_d`、`16Ψ/N_d`。
- 论文例子 7.5B、N_d = 64：120 GB、31.4 GB、16.6 GB、1.9 GB。
- 普通 DP 的 all-reduce 等于 reduce-scatter 加 all-gather，共 2Ψ。
- 优化器状态 12Ψ 是大头，所以 ZeRO-1 收益最大。

**追问链**：
1. ZeRO-2 为什么不增加通信？→ 梯度本来就要 reduce-scatter，每卡只留自己负责那部分的梯度，更新后再 all-gather 参数，总量仍是 2Ψ。
2. ZeRO-3 多的 Ψ 从哪来？→ 前向和反向都要按层 all-gather 参数，加上梯度的 reduce-scatter，共 3Ψ。
3. ZeRO 能省激活吗？→ 模型状态的三级不能。ZeRO 论文另有 ZeRO-R 处理激活、临时缓冲和碎片，例如在模型并行时切分激活。

**易错点**：
- 说 ZeRO-1 切的是参数。

## train-sys-003 张量并行和流水线并行各自的代价是什么，怎么组合
- level: 2
- tags: [training, inference]

**一句话答案**：TP 切层内矩阵，每层前向反向都有 all-reduce，适合放在节点内；PP 按层切段，段间只传激活，但有 (p − 1)/m 的气泡；一般 TP 用到单机卡数，跨机用 PP，剩下的卡做 DP。

**要点**：
- Megatron 切 MLP：A 按列切，各卡独立做 GeLU；B 按行切，最后一次 all-reduce。注意力按头切。
- 每个 Transformer 层前向 2 次 all-reduce，反向 2 次。
- 气泡占比 `(p − 1)/m`；交错调度每卡 v 段，降到 `(1/v)(p − 1)/m`，通信增加 v 倍。
- 1F1B 限制在途 micro-batch 数，比 GPipe 式省激活显存。

**追问链**：
1. 为什么 A 不按行切？→ 按行切要先把部分和加起来才能过 GeLU，GeLU 非线性，会多一个同步点。
2. 想减小气泡，为什么不无限加大 m？→ batch 大小固定时 m 越大 micro-batch 越小，算子效率下降；GPipe 式还要存 m 份激活。
3. 显存够、只是想更快，先加哪种？→ 常见做法是先加 DP：通信只在每步梯度同步，最容易扩展；模型放不下时才加 TP 和 PP。

**易错点**：
- 说 TP 跨机效果好。跨机带宽低，TP 的每层通信会成为瓶颈。

## train-sys-004 FSDP 和 ZeRO-3 是什么关系，FSDP 是怎么工作的
- level: 2
- tags: [training]

**一句话答案**：FSDP 是 PyTorch 原生的分片数据并行，全分片即 ZeRO-3；参数按 unit 展平均分，计算前 all-gather 拼回，反向后 reduce-scatter 梯度。

**要点**：
- 全分片时切参数、梯度、优化器状态，和 ZeRO-3 相同。
- FSDP 论文说受 ZeRO 启发，设计和 PyTorch 的 autograd、缓存分配器对齐。
- 分片因子 F：F = 1 等于 DDP；F 等于总卡数是全分片；中间是混合分片，组内分片、组间复制。
- FlatParameter 补齐到能被 F 整除，集合通信输入均匀，不用额外拷贝。
- 反向预取：先发下一个 unit 的 all-gather，再发当前的 reduce-scatter。

**追问链**：
1. unit 怎么切？→ 通常按 Transformer 层包一层。切得细峰值显存低，但集合通信次数多，可能降吞吐。
2. 什么时候用混合分片？→ 中等模型：全复制放不下，全分片又浪费跨机带宽；组内分片能减少跨机通信。
3. 峰值显存由什么决定？→ 分片后的模型状态，加上最大的一个未分片 unit。

**易错点**：
- 以为 FSDP 是一种张量并行。它是数据并行的分片版本。

## train-sys-005 RL 框架为什么要在训练引擎和推理引擎之间切换，共卡和分离部署怎么选
- level: 3
- tags: [training, inference]

**一句话答案**：生成受显存带宽限制，适合小 TP 多副本的推理引擎；训练受算力限制，适合大规模并行的训练引擎；两边切分不同，每轮要重分片、同步权重。共卡省卡但阶段串行；分离部署能并行，但同步训练会互相等，要靠异步。

**要点**：
- HybridFlow：同一批卡上训练用 p-t-d 切分，生成用更小的 TP 加 micro DP；3D-HybridEngine 让训练分片落在生成分片里，摘要称零显存冗余。
- verl 是 HybridFlow 的开源实现：训练 FSDP/FSDP2/Megatron，推理 vLLM/SGLang/HF，支持不同的设备映射。
- OpenRLHF：Ray 分组分配角色，DeepSpeed 训练，vLLM 生成。
- slime：Megatron 训练，SGLang 生成，Data Buffer 连接两者。
- HybridFlow 实验：小集群上全部共卡最好；更大集群上 actor 和 critic 分开放更好。

**追问链**：
1. 共卡时显存怎么分？→ 生成阶段给 KV Cache 留空间；训练阶段把生成用权重卸载到 CPU，优化器状态也可卸载。HybridFlow 把生成权重放在单独缓冲，训练时卸到 CPU。
2. 训练端和推理端算出的 log 概率不一致怎么办？→ 用训练引擎重算旧策略 log 概率再算比率；监控两者差异和 clip 比例。
3. 分离部署要怎样才不浪费卡？→ 异步：采样端持续生成，训练端凑够就更新，并控制策略滞后，见 [04 章](../04-agentic-rl/README.md#5-异步-rollout-与长尾)。

**易错点**：
- 说 verl 只能共卡。README 写明支持把模型放到不同的 GPU 组。

## train-sys-006 混合精度为什么要 FP32 主权重，FP16 和 BF16 有什么区别
- level: 2
- tags: [training]

**一句话答案**：更新量太小，FP16 里会变 0，要用 FP32 主权重累加；FP16 指数位少，小梯度下溢，要 loss scaling；BF16 范围同 FP32，通常免 scaling，但尾数少，仍要主权重。

**要点**：
- FP16 里绝对值小于 2⁻²⁴ 的数变成 0；更新量是学习率乘梯度。
- Micikevicius 等三项技术：FP32 主权重、loss scaling、FP16 乘 FP32 累加。
- 论文观察到约 5% 的权重梯度指数小于 −24。
- loss scaling：loss 乘系数，更新前除回去，要在梯度裁剪之前；系数乘最大梯度不能超过 65504。
- 位数：FP32 (1, 8, 23)，FP16 (1, 5, 10)，BF16 (1, 8, 7)。

**追问链**：
1. 为什么除回 scale 要在梯度裁剪之前？→ 否则裁剪阈值、权重衰减等超参都要跟着 scale 调。
2. 动态 loss scaling 怎么做？→ 常见做法：出现 inf 或 NaN 就跳过这一步并减小系数，连续若干步正常就增大系数。
3. 主权重占多少显存？→ 4Ψ，是每参数 16 字节里的一项。

**易错点**：
- 说 BF16 精度比 FP16 高。BF16 范围大、精度低。

## train-sys-007 Transformer 一层的激活为什么是 sbh(34 + 5as/h) 字节，序列并行和重计算各去掉了哪一项
- level: 3
- tags: [training]

**一句话答案**：逐个算子数反向要存的张量：线性层、LayerNorm、dropout 共 34sbh，注意力矩阵 5as²b。FlashAttention 去掉 5as/h；序列并行切开 TP 剩下的 10sbh。

**要点**：
- 假设激活 2 字节、dropout mask 1 字节、MLP 宽 4h（Korthikanti 等，arXiv 2205.05198，第 4.1 节）。
- 注意力块 11sbh + 5as²b：QKV 共享输入 2、Q 和 K 4、V 2、输出投影输入 2、dropout mask 1；softmax 输出 2as²b、dropout mask as²b、dropout 输出 2as²b。
- MLP 19sbh：两个线性层输入 2 + 8，GeLU 输入 8，dropout mask 1。两个 LayerNorm 4sbh。
- TP 下块的输入、LayerNorm、dropout 共 10sbh 每卡一份，其余 24 除以 t：`sbh(10 + 24/t + 5as/(ht))`。
- TP + SP：`(sbh/t)(34 + 5as/h)`；全量重计算：`2sbh`。

**追问链**：
1. 序列并行为什么能切 LayerNorm 和 dropout？→ 它们沿序列维逐 token 独立计算，按 s 切开结果不变。
2. 序列并行为什么不增加通信？→ 把 TP 的 all-reduce 拆成 reduce-scatter 和 all-gather，中间夹 LayerNorm、dropout；ring all-reduce 本来就是这两步，带宽相同。
3. 1F1B 流水线下第一段存多少层激活？→ 在途 p 个 micro-batch，每个 L/p 层，合计 L 层，和 p 无关。

**易错点**：
- 以为激活和参数量成正比。它和 s·b 成正比，注意力项和 s² 成正比。
- 以为 ZeRO 能切激活。ZeRO-1/2/3 只切模型状态。

## train-sys-008 ring all-reduce 每张卡要发多少数据，为什么 ZeRO-2 通信是 2Ψ 而 ZeRO-3 是 3Ψ
- level: 2
- tags: [training]

**一句话答案**：每卡发 2(N−1)/N·D，与卡数几乎无关。ZeRO-2 拆成梯度归约散发加参数收集，共 2Ψ；ZeRO-3 前向、反向各收集一次参数，再归约散发梯度，共 3Ψ。

**要点**：
- ring：向量切 N 块，N−1 步 reduce-scatter 后每卡持有一块的全局和，再 N−1 步 all-gather 转发。
- ZeRO 论文把一次 reduce-scatter 或 all-gather 记作 Ψ，所以普通 DP 记 2Ψ（arXiv 1910.02054，第 7 节）。
- ZeRO-2 只是把 all-reduce 拆成两半，中间插进“只更新自己那份参数”，总量不变。
- ZeRO-3 前向用完参数就扔，反向要再 all-gather 一次，多出一个 Ψ，是基线的 1.5 倍。FSDP 论文也写全分片通信是 DDP 的 1.5 倍。

**追问链**：
1. ring 的代价在哪？→ 每卡带宽开销基本不随 N 增长，但要走 2(N−1) 步，卡数多时延迟变大。
2. ZeRO-3 的通信怎样才便宜？→ 和计算重叠：按层预取下一层的 all-gather；FSDP 还限流，最多两个在途 all-gather。
3. 张量并行每层通信多少？→ 前向、反向各 2 次 all-reduce，每次 s·b·h 个元素，每个 micro-batch 都做，所以只放机内。

**易错点**：
- 说 all-reduce 每卡要发 N·D。ring 实现下约 2D。

## train-sys-009 流水线并行的气泡比例 (p − 1)/m 是怎么推出来的，1F1B 和交错调度各改变了什么
- level: 2
- tags: [training]

**一句话答案**：首尾各空 p−1 个前向和反向，气泡 (p−1)(t_f+t_b) 除以理想时间 m(t_f+t_b)。1F1B 气泡不变，在途数从 m 降到 p；交错调度气泡变 1/v，通信多约 v 倍。

**要点**：
- Narayanan 等，arXiv 2104.04473，第 2.2 节；比例是相对理想计算时间，不是相对总时间。
- 占总时间的口径是 (p−1)/(m+p−1)，p=8、m=32 时 17.9%，前者 21.9%。
- 推导假设每段 t_f、t_b 相同；首末段多嵌入层和输出层时更慢，实际气泡更大。
- 交错调度块的时间变成 t_f/v、t_b/v；论文要求 m 是 p 的整数倍。

**追问链**：
1. 为什么不无限加大 m？→ batch 固定时 micro-batch 变小，算子效率下降；GPipe 式还要存 m 份激活。
2. 交错调度的通信为什么多？→ 相邻块放在不同设备，一个 micro-batch 跨设备 pv−1 次而不是 p−1 次。
3. d=1 时气泡和 TP 的关系？→ p = n/t，气泡 (n/t − 1)/m，t 越大气泡越小，但 TP 跨机后通信变贵。

**易错点**：
- 说 1F1B 减小了气泡。它省的是激活显存。

## train-sys-010 HybridFlow 的 3D-HybridEngine 怎样在共卡时做到训练到生成的重分片零冗余
- level: 3
- tags: [training, inference]

**一句话答案**：生成 TP 组隔着取 rank、micro DP 组取相邻 rank，每卡训练分片落在自己的生成分片里；只在 micro DP 组内 all-gather，峰值等于生成分片，不另存训练权重。

**要点**：
- 记号：训练 p-t-d，生成 p_g-t_g-d_g-d，d_g = pt/(p_g t_g)（arXiv 2409.19256，第 5 节）。
- 论文图 8：8 卡，训练 1-4-2，生成 1-2-2-2。朴素分组下 G2、G3、G6、G7 的训练分片和生成分片不重叠，要另存。
- 表 2：朴素分组通信 (tp−1)/(tp)·M、峰值 M、冗余 M/(tp)；HybridFlow 通信 (tp − t_g p_g)/(t_g p_g tp)·M、峰值 M/(t_g p_g)、冗余 0。
- 实现：训练和生成权重分缓冲，训练时生成权重卸到 CPU，生成后 KV Cache 卸到 CPU。

**追问链**：
1. 为什么生成要比训练用更小的 TP？→ 生成受显存带宽限制，小 TP 多副本更能用满 GPU；训练受算力限制，大模型并行更好。
2. 不共卡时权重怎么同步？→ NCCL 广播或点对点发给推理端；跨集群经共享存储，slime 的 delta 模式只传变化的字节。
3. 大集群上共卡为什么变差？→ batch 固定时 DP 规模变大，计算通信比下降；HybridFlow 实验里更大集群把 actor、critic 分开放更好。

**易错点**：
- 以为零冗余是不通信。它仍要组内 all-gather，只是不多存一份权重。
