# RL 训练框架、并行与显存估算

> 层级：垂类 · training · 预计 150 分钟 · 题目：[questions.md](questions.md)
> 前置：[03 GRPO 及其变体](../03-grpo-family/README.md)、[04 Agentic RL](../04-agentic-rl/README.md)（异步 rollout）、[inference/01 KV Cache 与显存](../../inference/01-kv-cache-memory/README.md)

## 1 这一章解决什么问题

算法岗也会被问系统题：一个 7B 模型全参训练要多少显存，ZeRO 三个阶段各切了什么，TP 和 PP 怎么放，RL 框架为什么要在训练引擎和推理引擎之间来回切。这些问题有固定的估算口径，答错一个数量级很容易被追问到底。

只背结论不够。面试官常换一个条件追问：序列长度翻倍激活怎么变，开了序列并行为什么不多通信，交错调度的代价在哪，BF16 为什么不用 loss scaling。能从定义推出来，换条件也答得上。

学完你应该能：

- 从“反向传播要存什么”推出每参数 16 字节和每层 `sbh(34 + 5as/h)` 字节，说出各种重计算省了哪一项；
- 推出 ring all-reduce 每卡发送 `2(N−1)/N` 倍数据，再推出 ZeRO 各级的 2Ψ 和 3Ψ；
- 说清 Megatron 的列切和行切为什么每层前向、反向各 2 次 all-reduce，序列并行为什么不增加通信；
- 推出流水线气泡 `(p−1)/m` 和交错调度的 `1/v`，知道 1F1B 省的是显存不是气泡；
- 说出 RL 框架里训练引擎和推理引擎怎么协同、权重怎么同步、共卡时显存怎么分；
- 解释 FP16 的下溢阈值、loss scaling 的规则，以及 BF16 为什么通常不用它；
- 用一组可运行的模拟核对上面每个公式。

异步 rollout 和长尾问题在 [04 章第 5 节](../04-agentic-rl/README.md#5-异步-rollout-与长尾)，训推 log 概率不一致在 [03 章第 8 节](../03-grpo-family/README.md#8-工程细节与常见坑)，这里不重复。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| Ψ | 参数量（ZeRO 论文的记号） |
| K | 优化器状态每参数的字节数，混合精度 Adam 为 12 |
| N_d | 数据并行度（ZeRO 的记号）；ring 算法里的 rank 数记 N |
| s | 序列长度 |
| b | micro-batch 大小 |
| h | 隐藏维度 |
| a | 注意力头数 |
| L | 层数 |
| V | 词表大小 |
| t | 张量并行度 |
| p | 流水线段数（流水线并行度） |
| m | 一个 batch 里的 micro-batch 数 |
| v | 交错调度时每张卡上的模型块数 |
| t_f、t_b | 一个 micro-batch 在一段上的前向、反向时间 |

单位：全章用十进制 GB（1e9 字节）。和 GiB 混用的坑见 [inference/01 第 6 节](../../inference/01-kv-cache-memory/README.md#6-工程细节与常见坑)。

## 2 显存估算：参数、梯度、优化器状态、激活

训练显存分两类。一类是**模型状态**：参数、梯度、优化器状态，和参数量成正比，和 batch、序列长度无关。另一类是**激活**：前向时存下来、反向要用的中间结果，和 s·b 成正比，注意力部分和 s² 成正比。ZeRO 论文把激活、临时缓冲和碎片合称“残余状态”（residual states）。

### 2.1 模型状态：每参数 16 字节

出处：Rajbhandari 等，ZeRO，arXiv 1910.02054，第 3.1 节。和 [inference/01](../../inference/01-kv-cache-memory/README.md) 用同一口径。

混合精度 Adam 一步更新要用到这些量：

```text
前向、反向：     用 FP16 参数 W16 算 loss，反向得到 FP16 梯度 g16
优化器：        m ← β1·m + (1−β1)·g          一阶矩（动量）
               v ← β2·v + (1−β2)·g²         二阶矩（方差）
               W32 ← W32 − lr · m̂ / (√v̂ + ε)  在 FP32 主权重上更新
               W16 ← fp16(W32)               转回 FP16 给下一步用
```

每一项都要常驻显存，一步都不能少：

| 项 | 精度 | 每参数字节 | 为什么不能省 |
|---|---|---|---|
| 参数 W16 | FP16 | 2 | 前向、反向要用 |
| 梯度 g16 | FP16 | 2 | 优化器要用 |
| 主权重 W32 | FP32 | 4 | 更新量太小，FP16 里加不上去（第 6.2 节） |
| 动量 m | FP32 | 4 | 跨步累积 |
| 方差 v | FP32 | 4 | 存的是 g²，g = 1e-4 时 g² = 1e-8，FP16 里直接变 0（第 6.5 节代码） |
| 合计 | | 16 | 其中优化器状态 K = 4 + 4 + 4 = 12 |

所以模型状态是 `(2 + 2 + K)Ψ = 16Ψ` 字节。论文的例子：1.5B 的 GPT-2 至少要 24 GB，而只存 FP16 参数只要 3 GB。BF16 混合精度的字节数和 FP16 相同。

**口径要先问清**。下面这些变化都会改数字：

| 配置 | 每参数字节 | 说明 |
|---|---|---|
| 混合精度 Adam（ZeRO 口径） | 16 | 本章默认 |
| 梯度用 FP32 累加（推导） | 18 | 梯度 4 字节。Megatron-LM 用 BF16 训练时默认这样做，见第 7 节坑 1 |
| 纯 FP32 训练（推导） | 16 | 参数 4、梯度 4、动量 4、方差 4，没有单独的主权重 |
| 优化器状态全用 BF16（推导） | 8 | 2 + 2 + 2 + 2，不留主权重；省一半，但有精度风险 |
| LoRA（推导） | 约 2 | 冻结的基座只存 2Ψ；可训练的适配器参数量很小，按 16 字节算 |
| 只做推理 | 2 | 只有参数 |

### 2.2 激活：从每个算子存什么推出 sbh(34 + 5as/h)

出处：Korthikanti 等，arXiv 2205.05198，第 4.1 节。假设激活 16 位（每元素 2 字节）、dropout mask 每元素 1 字节、MLP 宽度 4h（GPT 式结构）。

反向时，每个算子要用它前向的某些输入或输出算梯度。把一层里每个算子要存的东西逐个列出来，单位是字节，`sbh` 是一个 s×b×h 张量的元素数：

| 算子 | 要存的 | 字节 | 张量并行能切开吗 |
|---|---|---|---|
| Q、K、V 投影 | 三者共享的输入 | 2sbh | 否 |
| QKᵀ | Q 和 K | 4sbh | 是（按头切） |
| softmax | 输出 | 2as²b | 是 |
| softmax 后的 dropout | mask | as²b | 是 |
| 乘 V | dropout 输出和 V | 2as²b + 2sbh | 是 |
| 注意力输出投影 | 输入 | 2sbh | 是 |
| 注意力后的 dropout | mask | sbh | 否 |
| MLP 第一层（h → 4h） | 输入 | 2sbh | 否 |
| GeLU | 输入 | 8sbh | 是 |
| MLP 第二层（4h → h） | 输入 | 8sbh | 是 |
| MLP 后的 dropout | mask | sbh | 否 |
| 两个 LayerNorm | 各自的输入 | 4sbh | 否 |

加起来：注意力块 11sbh + 5as²b，MLP 块 19sbh，LayerNorm 4sbh。

```text
每层激活 = 34·sbh + 5·as²b = sbh · (34 + 5as/h)          论文式 1
```

LayerNorm 的均值和方差只有 sb 个元素，h 是几千，论文把它们忽略了。

**两项的来源不同**。34 那部分来自线性层、LayerNorm、dropout，和 s 成正比。5as/h 那部分来自 softmax、softmax dropout 和乘 V 时存的 a 个 s×s 注意力矩阵，和 s² 成正比。比一比两项：5as/h > 34 时注意力矩阵占一半以上。论文算过：GPT-3（a = 96，s = 2048，h = 12288）是 80，MT-NLG（a = 128，s = 2048，h = 20480）是 64，都大于 34。

### 2.3 并行和重计算怎么改变激活

| 配置（论文表 2） | 每层激活 | 变了什么 |
|---|---|---|
| 不并行 | `sbh(34 + 5as/h)` | 基线 |
| 张量并行 | `sbh(10 + 24/t + 5as/(ht))` | 块内部的激活按 t 切开；块的输入、LayerNorm、dropout 那 10sbh 在每张卡上都有一份 |
| 张量 + 序列并行 | `(sbh/t)(34 + 5as/h)` | 那 10sbh 按序列维切开，整层激活都除以 t |
| 张量并行 + 选择性重计算 | `sbh(10 + 24/t)` | 注意力矩阵不存，反向时重算 |
| 张量 + 序列并行 + 选择性重计算 | `34·sbh/t` | 两者叠加 |
| 全量重计算（每层只存输入） | `2sbh` | 反向时整层重跑一次前向 |

逐项看上一节的表，“否”那一列加起来正好是 2 + 1 + 2 + 1 + 4 = 10，“是”那一列的 sbh 项是 4 + 2 + 2 + 8 + 8 = 24。这就是张量并行公式里 10 和 24/t 的来历。第 2.4 节的代码把这两个数也核对了。

**三种重计算的取舍**：

- **全量重计算**：每层只存输入 2sbh，L 层共 2sbhL。代价是反向时多跑一次前向。论文报告，在他们的训练里这会带来 30% 到 40% 的时间开销。反向计算某一层时，要临时展开这一层的完整激活，所以峰值还要加一层的 `sbh(34 + 5as/h)`（或 FlashAttention 下的 34sbh）。
- **选择性重计算**：只重算注意力矩阵那部分。这部分占显存多，但每个元素的计算量很小。论文报告：GPT-3 和 MT-NLG 上省下 70% 和 65% 的激活显存，只多 2.7% 和 1.6% 的 FLOPs。
- **FlashAttention**（Dao 等，arXiv 2205.14135）：前向不存 s×s 的注意力矩阵，只存输出和 softmax 归一化统计量，反向分块重算。效果和选择性重计算一样，5as/h 这一项基本去掉，而且不另付一次完整前向。常见做法是训练时开着它，所以面试估算常用 `34·sbh` 作每层激活。

**流水线并行不按 p 均分激活**。用 1F1B 调度时，第一段要同时存 p 个 micro-batch 的激活（第 3.5 节），每段有 L/p 层，所以第一段存的是 `p × L/p = L` 层的量，和 p 无关。论文写的总激活是 `(sbhL/t)(34 + 5as/h)`，按第一段算。交错调度还要再乘 `1 + (p−1)/(pv)`（论文这处把块数记作 m，这里改记 v，避免和 micro-batch 数混淆）。

**公式没算进去的**：

- 输出层的 logits：交叉熵用 FP32 算，存 `4sbV/t` 字节（论文第 4.3 节）。词表大、序列长时这一项不小，14B 配置下就有 3.3 GB。论文说对 22B 模型，嵌入层、最后的 LayerNorm 和输出层这几项合计不到总激活的 0.01%，所以它的式 5 不计入。面试估算时 logits 单独列出更稳妥：它在 loss 处出现，峰值时和其他激活叠加（第 7 节坑 2）。
- 通信缓冲和碎片。ZeRO 论文举例：1.5B 模型一个扁平化的 FP32 梯度缓冲就要 6 GB。
- 公式按 GPT 式的 4h MLP 推出。SwiGLU 这类门控 MLP 要多存一个分支，中间维度也不是 4h，系数会变。面试时说明这是估算口径即可。

### 2.4 实现：显存估算函数

下面的函数实现 2.1 节的模型状态和 2.3 节的激活表。它核对了 34 = 10 + 24 的逐项拆分、ZeRO 论文的两个例子、正文 7B 和题目 14B 的全部数字，以及论文里选择性重计算省 70% 和 65% 的结论。

```python
GB = 1e9  # 全章用十进制 GB


def model_states(psi, nd=1, stage=0, K=12):
    """每卡模型状态字节数。psi 参数量；nd 数据并行度；stage 0 普通 DP，1/2/3 对应 ZeRO-1/2/3。"""
    p, g, o = 2 * psi, 2 * psi, K * psi          # FP16 参数、FP16 梯度、优化器状态
    if stage >= 1:
        o /= nd
    if stage >= 2:
        g /= nd
    if stage >= 3:
        p /= nd
    return p + g + o


def act_per_layer(s, b, h, a, t=1, sp=False, recompute="none"):
    """Korthikanti 等表 2：每层激活字节数。recompute: none / selective / full。"""
    sbh = s * b * h
    if recompute == "full":
        return 2 * sbh                            # 每层只存输入
    attn = 0 if recompute == "selective" else 5 * a * s / h
    if sp:
        return sbh * (34 + attn) / t
    return sbh * (10 + 24 / t + attn / t)


def logits_bytes(s, b, V):
    return 4 * s * b * V                          # FP32 logits


# 0) 逐项拆开 34：(系数, 是否随张量并行切开)，单位 sbh 字节；注意力矩阵三项单位 as²b
items = {
    "QKV 的共享输入": (2, False), "Q、K（给 QKᵀ）": (4, True), "V": (2, True),
    "输出投影的输入": (2, True), "注意力 dropout mask": (1, False),
    "MLP 第一层输入": (2, False), "MLP 第二层输入": (8, True), "GeLU 输入": (8, True),
    "MLP dropout mask": (1, False), "两个 LayerNorm 的输入": (4, False),
}
attn_matrix = {"softmax 输出": 2, "softmax dropout mask": 1, "dropout 输出（乘 V 用）": 2}
assert sum(c for c, _ in items.values()) == 34 and sum(attn_matrix.values()) == 5
assert sum(c for c, split in items.values() if not split) == 10      # 张量并行切不开的 10
assert sum(c for c, split in items.values() if split) == 24          # 切开的 24
# 5as²b = sbh · 5as/h：所以每层是 sbh(34 + 5as/h)
s_, b_, h_, a_ = 128, 2, 64, 4
assert 34 * s_ * b_ * h_ + 5 * a_ * s_ ** 2 * b_ == act_per_layer(s_, b_, h_, a_)

# 1) 每参数 16 字节；ZeRO 论文的两个例子
assert model_states(1) == 16
assert round(model_states(1.5e9) / GB) == 24 and 2 * 1.5e9 / GB == 3
paper = [model_states(7.5e9, 64, k) / GB for k in range(4)]
assert [round(x, 1) for x in paper] == [120.0, 31.4, 16.6, 1.9]      # 表 1 写 1.88

# 2) 7B、8 卡（正文 2.4 节）
psi = 7e9
z = [model_states(psi, 8, k) / GB for k in range(4)]
assert z == [112.0, 38.5, 26.25, 14.0]
L, h, a, s, b, V = 32, 4096, 32, 4096, 1, 32000
full = act_per_layer(s, b, h, a) / GB
flash = act_per_layer(s, b, h, a, recompute="selective") / GB
assert round(full, 2) == 3.25 and round(full * L) == 104
assert round(flash, 2) == 0.57 and round(flash * L, 1) == 18.3
assert round(act_per_layer(s, b, h, a, recompute="full") * L / GB, 2) == 1.07
assert round(logits_bytes(s, b, V) / GB, 2) == 0.52
assert round(z[2] + flash * L, 1) == 44.5                              # ZeRO-2 + 去掉注意力矩阵项

# 3) 14B、16 卡（train-sys-001）
psi, L, h, a, s, V = 14e9, 40, 5120, 40, 8192, 100000
assert [model_states(psi, 16, k) / GB for k in range(4)] == [224.0, 66.5, 40.25, 14.0]
assert round(act_per_layer(s, 1, h, a) / GB, 1) == 14.8
assert round(act_per_layer(s, 1, h, a, recompute="selective") * L / GB) == 57
assert round(logits_bytes(s, 1, V) / GB, 1) == 3.3

# 4) 张量并行 + 序列并行 = 不并行除以 t；TP=8 时 7B 一层的激活
s, h, a = 4096, 4096, 32
for t in (2, 4, 8):
    assert abs(act_per_layer(s, 1, h, a, t, sp=True) * t - act_per_layer(s, 1, h, a)) < 1
tp_only = act_per_layer(s, 1, h, a, t=8, recompute="selective")
tp_sp = act_per_layer(s, 1, h, a, t=8, sp=True, recompute="selective")
assert round(tp_only / tp_sp, 2) == round((10 + 24 / 8) / (34 / 8), 2) == 3.06

# 5) 论文的两个例子：选择性重计算省下的比例 = (5as/h) / (34 + 5as/h)
for a_, s_, h_, saved in [(96, 2048, 12288, 0.70), (128, 2048, 20480, 0.65)]:
    r = 5 * a_ * s_ / h_
    assert round(r / (34 + r), 2) == saved

# 6) RL：7B 策略、8 卡 ZeRO-3。GRPO 要 actor 训练状态 + reference 推理权重；PPO 再加 critic
actor, ref = model_states(7e9, 8, 3) / GB, 2 * 7e9 / 8 / GB
grpo, ppo = actor + ref, 2 * actor + ref
assert (round(grpo, 2), round(ppo, 2)) == (15.75, 29.75)
print("ok", z, round(full, 3), round(flash, 3))
```

### 2.5 数值例子：7B 模型能不能在 8 张 80 GB 卡上训

假设一个 7B 模型：L = 32、h = 4096、a = 32；训练 s = 4096、b = 1；8 卡数据并行。

```text
模型状态（不切分）： 16 × 7e9 = 112 GB          单卡 80 GB 放不下
ZeRO-1（切优化器）： 4Ψ + 12Ψ/8 = 28 + 10.5 = 38.5 GB
ZeRO-2（再切梯度）： 2Ψ + 14Ψ/8 = 14 + 12.25 = 26.25 GB
ZeRO-3（再切参数）： 16Ψ/8 = 14 GB

激活，每层不重计算： 4096 × 1 × 4096 × (34 + 5 × 32 × 4096 / 4096)
                  = 16.8M × 194 ≈ 3.25 GB；32 层 ≈ 104 GB
去掉注意力矩阵项（FlashAttention 或选择性重计算）：
                  16.8M × 34 ≈ 0.57 GB；32 层 ≈ 18.3 GB
全量重计算：        2 × 16.8M × 32 ≈ 1.07 GB
logits（V = 32000，FP32）：4096 × 32000 × 4 ≈ 0.52 GB
```

结论：这个配置下，ZeRO-2 加 FlashAttention，单卡约 26 + 18 ≈ 45 GB，加上 logits、缓冲和碎片，80 GB 卡可以放下；不切分则模型状态本身就放不下。

几个变化方向值得手算一遍：

| 改动 | 激活怎么变（FlashAttention 口径） | 原因 |
|---|---|---|
| s 从 4096 到 8192 | 翻倍，18.3 → 36.5 GB | 34·sbh 和 s 成正比；不开 FlashAttention 时注意力项和 s² 成正比，会变成约 4 倍 |
| b 从 1 到 4 | 4 倍 | 和 b 成正比；模型状态不变 |
| 卡数从 8 到 64 | 不变 | ZeRO 只切模型状态，激活每卡一份 |
| 加 TP = 8 和序列并行 | 除以 8，约 2.3 GB | 第 2.3 节表 |
| 只加 TP = 8、不加序列并行 | 约 7.0 GB | `sbh(10 + 24/8)`，是加序列并行的 3.06 倍 |

更大的 14B 配置（16 卡、s = 8192、词表 10 万）的完整估算在题目 train-sys-001 里，第 2.4 节的代码核对了它的数字。

## 3 数据并行、张量并行、流水线并行

| | 切什么 | 通信 | 代价 |
|---|---|---|---|
| 数据并行 DP | 切数据，每卡一份完整模型 | 每步对梯度做 all-reduce | 每卡都存完整模型状态 |
| 张量并行 TP | 切每层内部的矩阵 | 每层前向、反向都有 all-reduce | 通信频繁，要高带宽，适合节点内 |
| 流水线并行 PP | 按层切成多段，每段放一组卡 | 段之间点对点传激活和梯度 | 有流水线气泡；micro-batch 多了激活显存涨 |

### 3.1 集合通信的通信量：ring all-reduce 每卡发 2(N−1)/N 倍数据

后面所有通信量都建立在这一节上。三个集合通信原语，N 个 rank 各有一个长度 D 的向量：

| 原语 | 做完之后 |
|---|---|
| reduce-scatter | 向量切成 N 块，rank r 拿到某一块的全局和 |
| all-gather | 每个 rank 原来只有一块，做完每个 rank 都有全部 N 块 |
| all-reduce | 每个 rank 都拿到整个向量的全局和 = reduce-scatter + all-gather |

ring 算法把 N 个 rank 连成一个环，每个 rank 只发给下游。

```text
reduce-scatter：向量切成 N 块，每块 D/N
  第 k 步（k = 0..N−2）：rank r 把第 (r − k) mod N 块发给 r+1，r+1 加到自己那块上
  N−1 步后，rank r 手里第 (r+1) mod N 块累加了全部 N 份
  每个 rank 发送：(N − 1) · D/N

all-gather：每个 rank 把手里的完整块沿环转发，N−1 步后人人都有
  每个 rank 发送：(N − 1) · D/N

all-reduce = 两者相加：每个 rank 发送 2(N − 1)/N · D
```

`2(N−1)/N` 随 N 增大趋于 2，几乎和 N 无关。所以 ring all-reduce 每卡要发的数据量基本不随卡数增长，这是数据并行容易扩展的原因（推导）；代价是要走 2(N−1) 步，卡数多时延迟变大。ZeRO 论文把一次 reduce-scatter 或 all-gather 的量都记作 Ψ（忽略 (N−1)/N），所以普通 DP 的 all-reduce 记作 2Ψ。第 4.2 节的代码模拟了这个环，assert 每卡发送正好是 `2(N−1)/N · D`。

### 3.2 张量并行：Megatron 的列切和行切

出处：Shoeybi 等，arXiv 1909.08053，第 3 节。

MLP 是 `Z = GeLU(XA) · B`，X 是 (sb, h)，A 是 (h, 4h)，B 是 (4h, h)。要把 A、B 切到 t 张卡上，A 有两种切法：

- **A 按行切**：X 也要按列切成 `[X1, X2]`，于是 `XA = X1·A1 + X2·A2`。每张卡只有一个部分和，而 GeLU 非线性，`GeLU(X1A1 + X2A2) ≠ GeLU(X1A1) + GeLU(X2A2)`，所以过 GeLU 之前必须同步一次。
- **A 按列切** `A = [A1, A2]`：`XA = [XA1, XA2]`，每张卡算出的是完整结果的一部分列，`[Y1, Y2] = [GeLU(XA1), GeLU(XA2)]`，各卡独立做 GeLU，不用同步。

接下来 B 按行切 `B = [B1; B2]`，正好吃各卡的 Yi：

```text
Z = Y · B = [Y1, Y2] · [B1; B2] = Y1·B1 + Y2·B2
```

每张卡得到一个部分和 `Yi·Bi`，最后 all-reduce 一次加起来，再进 dropout。整个 MLP 前向只有这一次通信。

**反向**。对 X 求梯度：`∂L/∂X = Σ_i (∂L/∂Yi)·(∂Yi/∂X)`，每张卡只能算出自己那份的贡献，要 all-reduce 一次。权重梯度 `∂L/∂Ai`、`∂L/∂Bi` 只用本卡的数据，不用通信。论文把这两个通信写成一对共轭算子：

| 算子 | 前向 | 反向 | 位置 |
|---|---|---|---|
| f | 恒等 | all-reduce | 块的入口 |
| g | all-reduce | 恒等 | 块的出口 |

注意力块同理：Q、K、V 的投影按列切，每张卡分到若干个完整的头，注意力在卡内独立算完；输出投影按行切，出口一次 all-reduce。

**所以一个 Transformer 层前向 2 次 all-reduce（注意力、MLP 各一次的 g），反向 2 次（各一次的 f）**，每次的数据量是 s·b·h 个元素。和 DP 不同，这些通信在每个 micro-batch 的每一层都要做，还在关键路径上，所以 TP 必须放在 NVLink 这种高带宽链路上。

输出层也要切：嵌入矩阵按词表维切，每张卡算一部分 logits。论文把交叉熵和并行的 GEMM 融合，各卡先在本地算出自己那部分的统计量，只通信 b×s 个标量，不用 all-gather b×s×V 的 logits。

### 3.3 序列并行：把 all-reduce 拆开，通信量不变

出处：Korthikanti 等，arXiv 2205.05198，第 4.2.2 节。

张量并行没切 LayerNorm 和 dropout，这部分在 t 张卡上各算一份，就是第 2.3 节里切不开的那 10sbh。这些算子沿序列维是独立的，每个 token 自己算自己的，所以可以按 s 切。

做法是把 all-reduce 拆成它的两半，在中间插进 LayerNorm 和 dropout：

```text
TP：      … → g（all-reduce）→ dropout、残差、LayerNorm（每卡全量）→ f（恒等）→ 下一块 …
TP + SP： … → ḡ（reduce-scatter）→ dropout、残差、LayerNorm（每卡 s/t 个 token）→ g（all-gather）→ 下一块 …
```

论文里新算子的定义：g 前向 all-gather、反向 reduce-scatter；ḡ 前向 reduce-scatter、反向 all-gather。一层前向、反向合计：TP 是 4 次 all-reduce，TP + SP 是 4 次 all-gather 加 4 次 reduce-scatter。因为一次 ring all-reduce 本来就是一次 reduce-scatter 加一次 all-gather，两者的通信带宽相同，**序列并行不增加通信量**，激活从 `sbh(10 + 24/t + …)` 降到整层除以 t。

还有一个细节：第一个线性层需要完整的 Y（all-gather 之后的结果）。论文只存本卡那 s/t 份，反向时再 all-gather 一次，和计算 Y 的梯度重叠，隐藏延迟。

### 3.4 实现：用 numpy 模拟列切和行切

下面把一个 MLP 切到 t = 4 张卡上，核对四件事：列切加行切的结果和单卡一致；A 按行切会在 GeLU 处出错；反向时 X 的梯度要 all-reduce、权重梯度不用；MLP 前向和反向各 1 次 all-reduce；序列并行把 all-reduce 拆成 reduce-scatter 加 all-gather 后结果不变。

```python
import numpy as np

rng = np.random.default_rng(0)
s, h, t = 8, 8, 4                 # token 数（s·b 合并）、隐藏维、张量并行度
X = rng.normal(size=(s, h))       # (s, h)
A = rng.normal(size=(h, 4 * h))   # (h, 4h)  第一个线性层
B = rng.normal(size=(4 * h, h))   # (4h, h)  第二个线性层
gelu = lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
ALL_REDUCE = []                   # 记录每次 all-reduce 的元素数


def all_reduce(parts):
    ALL_REDUCE.append(parts[0].size)
    total = sum(parts)
    return [total.copy() for _ in parts]


Z_ref = gelu(X @ A) @ B           # 单卡参考结果

# 1) A 按列切、B 按行切：中间不通信，最后一次 all-reduce
A_cols = np.split(A, t, axis=1)   # 每卡 (h, 4h/t)
B_rows = np.split(B, t, axis=0)   # 每卡 (4h/t, h)
Y_parts = [gelu(X @ Ai) for Ai in A_cols]            # 每卡独立做 GeLU，(s, 4h/t)
Z_parts = [Yi @ Bi for Yi, Bi in zip(Y_parts, B_rows)]   # 部分和，(s, h)
Z_tp = all_reduce(Z_parts)
assert all(np.allclose(z, Z_ref) for z in Z_tp)
assert np.allclose(np.concatenate(Y_parts, 1), gelu(X @ A))   # 列切的输出拼起来就是完整的 Y

# 2) 反过来 A 按行切：部分和要先加起来才能过 GeLU，GeLU 非线性，各卡分别做是错的
A_rows, X_cols = np.split(A, t, axis=0), np.split(X, t, axis=1)
wrong = sum(gelu(Xi @ Ai) for Xi, Ai in zip(X_cols, A_rows))
assert not np.allclose(wrong, gelu(X @ A))

# 3) 反向：dL/dX = dZ·(∂Z/∂X)。每卡只能算出自己那份的贡献，要 all-reduce（f 算子）
dZ = rng.normal(size=(s, h))
pre = X @ A
dgelu = lambda x: (gelu(x + 1e-6) - gelu(x - 1e-6)) / 2e-6
dX_ref = ((dZ @ B.T) * dgelu(pre)) @ A.T
dX_parts = [((dZ @ Bi.T) * dgelu(X @ Ai)) @ Ai.T for Ai, Bi in zip(A_cols, B_rows)]
dX_tp = all_reduce(dX_parts)
assert all(np.allclose(d, dX_ref, atol=1e-5) for d in dX_tp)
# 权重梯度不用通信：每卡只要自己那块
dB_ref = gelu(pre).T @ dZ
assert np.allclose(np.concatenate([Yi.T @ dZ for Yi in Y_parts], 0), dB_ref)

# MLP 前向 1 次、反向 1 次；注意力块同理。每层合计前向 2 次、反向 2 次，每次 s·b·h 个元素
assert ALL_REDUCE == [s * h, s * h]

# 4) 序列并行：all-reduce = reduce-scatter + all-gather。把 all-reduce 拆开，
#    LayerNorm、dropout 只在自己那 s/t 段 token 上做，激活按 t 切开
def reduce_scatter(parts):
    return np.split(sum(parts), t, axis=0)           # 每卡拿到 s/t 行的完整和

def all_gather(shards):
    return np.concatenate(shards, 0)

ln = lambda x: (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
shards = reduce_scatter(Z_parts)                     # ḡ：前向 reduce-scatter
shards = [ln(z) for z in shards]                     # 只在 s/t 行上做 LayerNorm
assert shards[0].shape == (s // t, h)
assert np.allclose(all_gather(shards), ln(Z_ref))   # g：下一个块前 all-gather，结果不变
print("ok", ALL_REDUCE)
```

### 3.5 流水线并行：气泡 (p−1)/m 的推导

出处：Narayanan 等，arXiv 2104.04473，第 2.2 节。

L 层按顺序切成 p 段，段 i 放在设备 i 上。一个 batch 切成 m 个 micro-batch，前向从段 1 流到段 p，反向从段 p 流回段 1。为了保证同步的优化器语义，一个 batch 的所有 micro-batch 做完反向后统一更新一次（pipeline flush）。

**GPipe 式调度**：先做完全部 m 个前向，再做全部 m 个反向。

```text
时间 →（p = 4，m = 4，F 为前向，B 为反向）
设备 1  F1 F2 F3 F4 .  .  .  .  .  .  B4 B3 B2 B1
设备 2  .  F1 F2 F3 F4 .  .  .  .  B4 B3 B2 B1 .
设备 3  .  .  F1 F2 F3 F4 .  .  B4 B3 B2 B1 .  .
设备 4  .  .  .  F1 F2 F3 F4 B4 B3 B2 B1 .  .  .
（示意：画成 B 和 F 一样长；实际 t_b 约为 2 t_f）
```

设备 p 要等前面 p−1 段都做完第一个 micro-batch 的前向才能开始，开头空了 p−1 个前向时间。反向结束时对称，设备 1 末尾空了 p−1 个反向时间。

```text
气泡时间    t_pb = (p − 1)(t_f + t_b)
理想时间    t_id = m (t_f + t_b)                    没有空闲时一个 batch 要算的时间
气泡占比    t_pb / t_id = (p − 1) / m                论文式
```

两点要说清：

- 占比是**相对理想计算时间**，不是相对总时间。换成占总时间的比例是 `(p−1)/(m + p − 1)`。p = 8、m = 32 时前者 21.9%，后者 17.9%。
- 结果和 t_b/t_f 的比例无关，前提是每段的 t_f、t_b 相同。段切得不均时，最慢的那段决定节奏，实际气泡更大。

要气泡小就要 m ≫ p。但 batch 大小固定时，m 越大 micro-batch 越小，矩阵乘的效率下降；GPipe 式还要同时存 m 个 micro-batch 的激活。

**1F1B（PipeDream-Flush）**：第 d 段（从 0 起编号）先做 p−d−1 个前向热身，然后前向、反向交替，最后把剩下的反向做完。论文说它的气泡时间和 GPipe 式相同，但同时在途（前向做完、反向没做）的 micro-batch 数最多是 p，而不是 m。第 3.6 节的模拟里，第 d 段在途峰值是 p−d。所以 1F1B 省的是激活显存，不是气泡。这也是第 2.3 节“第一段存 L 层激活”的来历：第一段在途 p 个 micro-batch，每个 L/p 层。

**交错调度**：每张卡放 v 个不相邻的模型块，例如 4 张卡、16 层、v = 2 时，设备 1 放第 1、2、9、10 层，设备 2 放第 3、4、11、12 层。每个块的前向和反向时间变成 t_f/v、t_b/v，开头和末尾的空闲按块计：

```text
交错调度气泡  t_pb^int = (p − 1)(t_f + t_b) / v
气泡占比      (1/v) · (p − 1) / m
```

代价是通信：一个 micro-batch 要跨设备 pv − 1 次而不是 p − 1 次，论文记作通信量增加 v 倍。论文的交错调度还要求 m 是 p 的整数倍。

### 3.6 实现：流水线调度的模拟

下面的代码不套公式，而是按依赖关系一步步排出每个设备上每个前向、反向的开始和结束时间，量出整个 batch 的总时间，再和公式对比。它核对了：GPipe 式和 1F1B 的气泡都是 `(p−1)/m`，交错调度是 `(1/v)(p−1)/m`；结果和 t_b/t_f 无关；GPipe 式每段在途 m 个 micro-batch，1F1B 第 d 段在途 p−d 个；交错调度的跨设备传输次数；两种占比口径的换算。

```python
import numpy as np

def simulate(orders, deps, dur):
    """orders[d]: 设备 d 按顺序执行的操作列表；deps[op]: 依赖；dur[op]: 时长。返回每个操作的结束时间。"""
    end, ptr, free = {}, [0] * len(orders), [0.0] * len(orders)
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for d, order in enumerate(orders):
            if ptr[d] == len(order):
                continue
            op = order[ptr[d]]
            if all(x in end for x in deps[op]):
                start = max([free[d]] + [end[x] for x in deps[op]])
                end[op] = start + dur[op]
                free[d] = end[op]
                ptr[d] += 1
                remaining -= 1
                progressed = True
        assert progressed, "调度死锁"
    return end

def pipeline(p, m, v=1, sched="gpipe", tf=1.0, tb=2.0):
    S = p * v                       # 虚拟段数；段 k 放在设备 k % p
    deps, dur = {}, {}
    for j in range(m):
        for k in range(S):
            deps[("F", j, k)] = [("F", j, k - 1)] if k else []
            deps[("B", j, k)] = [("B", j, k + 1)] if k < S - 1 else [("F", j, S - 1)]
            dur[("F", j, k)], dur[("B", j, k)] = tf / v, tb / v
    orders = []
    for d in range(p):
        chunks = [d + c * p for c in range(v)]
        if sched == "gpipe":
            fw = [("F", j, k) for g in range(0, m, p) for k in chunks for j in range(g, min(g + p, m))]
            bw = [("B", j, k) for g in range(0, m, p) for k in reversed(chunks) for j in range(g, min(g + p, m))]
            orders.append(fw + bw)
        else:  # 1F1B，v = 1
            warm = min(p - d - 1, m)
            o = [("F", j, d) for j in range(warm)]
            for j in range(m - warm):
                o += [("F", warm + j, d), ("B", j, d)]
            o += [("B", j, d) for j in range(m - warm, m)]
            orders.append(o)
    end = simulate(orders, deps, dur)
    total = max(end.values())
    t_id = m * (tf + tb)
    return (total - t_id) / t_id, orders, end, dur

def peak_in_flight(order):
    """一个设备上同时存着激活（前向做完、反向没做）的 micro-batch 数的峰值。"""
    live, peak = set(), 0
    for kind, j, k in order:
        if kind == "F":
            live.add((j, k))
        else:
            live.discard((j, k))
        peak = max(peak, len(live))
    return peak


# 1) GPipe 和 1F1B 的气泡比例都是 (p−1)/m；交错调度降到 (1/v)(p−1)/m
for p, m in [(4, 8), (4, 16), (8, 32), (8, 64)]:
    for v in (1, 2, 4):
        bubble, *_ = pipeline(p, m, v)
        assert abs(bubble - (p - 1) / m / v) < 1e-9
    bubble, *_ = pipeline(p, m, 1, "1f1b")
    assert abs(bubble - (p - 1) / m) < 1e-9

# 2) 公式与 tf、tb 的比例无关：气泡 = (p−1)(tf+tb)，理想时间 = m(tf+tb)
for tf, tb in [(1, 1), (1, 3), (2, 5)]:
    bubble, *_ = pipeline(4, 12, 1, "1f1b", tf, tb)
    assert abs(bubble - 3 / 12) < 1e-9

# 3) 显存：GPipe 第一段要存 m 个 micro-batch 的激活，1F1B 最多 p 个
p, m = 4, 16
_, g_orders, *_ = pipeline(p, m, 1, "gpipe")
_, f_orders, *_ = pipeline(p, m, 1, "1f1b")
assert [peak_in_flight(o) for o in g_orders] == [m] * p
assert [peak_in_flight(o) for o in f_orders] == [p - d for d in range(p)]

# 4) 交错调度的代价：段 k 放在设备 k % p，相邻虚拟段总在不同设备上，
#    一个 micro-batch 前向跨设备传激活的次数从 p−1 变成 pv−1，约 v 倍
def cross_device_sends(p, v):
    S = p * v
    return sum(1 for k in range(1, S) if k % p != (k - 1) % p)

assert [cross_device_sends(4, v) for v in (1, 2, 4)] == [3, 7, 15]
assert cross_device_sends(4, 4) / cross_device_sends(4, 1) == 5.0     # (pv−1)/(p−1)，p 大时趋于 v

# 5) 占比写成相对“总时间”是另一个口径：(p−1)/(m+p−1)
bubble, *_ = pipeline(8, 32)
assert abs(bubble / (1 + bubble) - 7 / 39) < 1e-9
print("ok")
```

### 3.7 怎么组合：PTD-P

Megatron 把三种并行组合起来，论文叫 PTD-P。总卡数 n = p·t·d（d 是数据并行度）。论文的经验结论：

- TP 一般用到单机 GPU 数为止，跨机再用 PP 扩大模型。TP 每层都 all-reduce，跨机带宽低时会成为瓶颈；PP 段间只做点对点传输，便宜得多。
- 模型并行总规模 t·p 只要足够放下模型，剩下的卡用 DP 扩展。DP 每个 batch 才同步一次梯度，TP 每个 micro-batch 都要通信。
- d = 1 时 p = n/t，气泡是 `(n/t − 1)/m`。t 越大气泡越小，但 TP 跨机后通信变贵。所以要在两者之间权衡。
- 最优 micro-batch 大小取决于模型、p、DP 规模和 batch 大小。

## 4 ZeRO 与 FSDP

### 4.1 ZeRO 三级：每卡显存和通信量的推导

**问题**：DP 的每张卡都存完整的 16Ψ 模型状态，N_d 张卡冗余了 N_d 倍。但每张卡在一步里并不需要全部状态：优化器只在更新时用，每张卡完全可以只更新 1/N_d 的参数。

出处：Rajbhandari 等，arXiv 1910.02054，第 5 节和第 7 节。

**ZeRO-1（P_os，切优化器状态）**：每张卡只存 1/N_d 的 12Ψ 优化器状态，只负责更新这部分参数。

```text
每卡 = 2Ψ（FP16 参数）+ 2Ψ（FP16 梯度）+ 12Ψ/N_d = 4Ψ + 12Ψ/N_d
N_d 很大时 ≈ 4Ψ，论文称 4 倍缩减
```

**ZeRO-2（P_os+g，再切梯度）**：既然每张卡只更新 1/N_d 的参数，它也只需要这部分的梯度。梯度不做 all-reduce，改做 reduce-scatter，每张卡只收到自己那份的全局和。

```text
每卡 = 2Ψ + (2Ψ + 12Ψ)/N_d = 2Ψ + 14Ψ/N_d        N_d 大时 ≈ 2Ψ，8 倍缩减
```

**ZeRO-3（P_os+g+p，再切参数）**：FP16 参数也只存 1/N_d。前向算到某一层前，把这一层的参数 all-gather 回来，用完就扔。

```text
每卡 = 16Ψ / N_d                                  缩减倍数等于 N_d
```

**通信量**，按 ZeRO 论文的口径（一次 reduce-scatter 或 all-gather 记 Ψ，见第 3.1 节）：

| 阶段 | 每步做什么 | 通信量 |
|---|---|---|
| 普通 DP | 梯度 all-reduce = reduce-scatter + all-gather | 2Ψ |
| ZeRO-1 | 和普通 DP 一样 all-reduce 梯度，各卡只用自己那份更新 | 2Ψ |
| ZeRO-2 | 梯度 reduce-scatter（Ψ），更新后 all-gather 参数（Ψ） | 2Ψ |
| ZeRO-3 | 前向 all-gather 参数（Ψ），反向再 all-gather 一次（Ψ），梯度 reduce-scatter（Ψ） | 3Ψ，基线的 1.5 倍 |

ZeRO-2 不多通信的关键是：普通 DP 的 all-reduce 本来就是 reduce-scatter 加 all-gather。ZeRO-2 只是把这两半拆开，中间插进“只更新自己那份”的优化器步。all-gather 的对象从梯度换成了更新后的参数，量一样。

ZeRO-3 多出来的 Ψ 是反向那次 all-gather：前向用完的参数已经扔掉，反向要重新收集一次。论文的 7.5B、N_d = 64 例子：四行分别是 120 GB、31.4 GB、16.6 GB、1.9 GB（表 1 写作 1.88）。

还要注意 ZeRO 的三级都不切激活。激活在 ZeRO 论文里归 ZeRO-R：模型并行时把激活检查点切开存（P_a），设定临时缓冲大小，主动整理碎片。

### 4.2 实现：ring all-reduce 与 ZeRO 的通信量

下面的代码模拟一个 N 个 rank 的环，每个 rank 只能发给下游，并记下每个 rank 发出的元素数。它核对了：all-reduce 结果等于逐元素求和；每卡发送正好是 `2(N−1)/N · D`；按 ZeRO 各级的通信步骤走一遍，普通 DP 和 ZeRO-2 一样是 2Ψ，ZeRO-3 是 3Ψ。

```python
import numpy as np

rng = np.random.default_rng(0)


class Ring:
    """N 个 rank 连成环，rank r 只能发给 (r+1) % N。sent[r] 记 rank r 发出的元素数。"""

    def __init__(self, n):
        self.n, self.sent = n, [0] * n

    def reduce_scatter(self, data):
        """data[r]: rank r 的完整向量，切成 n 块。N−1 步后 rank r 持有第 (r+1) % n 块的全局和。"""
        n = self.n
        chunks = [np.array_split(d.astype(float), n) for d in data]
        for step in range(n - 1):
            msgs = []
            for r in range(n):
                c = (r - step) % n                       # 这一步 rank r 发出第 c 块
                msgs.append((r, c, chunks[r][c].copy()))
                self.sent[r] += chunks[r][c].size
            for r, c, m in msgs:
                chunks[(r + 1) % n][c] += m              # 下游累加
        return {r: ((r + 1) % n, chunks[r][(r + 1) % n]) for r in range(n)}

    def all_gather(self, owned, size):
        """owned[r] = (块号, 块内容)。N−1 步后每个 rank 拿到全部块。"""
        n = self.n
        have = [{c: v.copy()} for c, v in (owned[r] for r in range(n))]
        cur = [owned[r][0] for r in range(n)]           # 每个 rank 这一步要转发的块号
        for _ in range(n - 1):
            msgs = []
            for r in range(n):
                v = have[r][cur[r]]
                msgs.append((r, cur[r], v))
                self.sent[r] += v.size
            for r, c, v in msgs:
                have[(r + 1) % n][c] = v.copy()
                cur[(r + 1) % n] = c
        return [np.concatenate([have[r][c] for c in range(n)])[:size] for r in range(n)]

    def all_reduce(self, data):
        return self.all_gather(self.reduce_scatter(data), data[0].size)


# 1) ring all-reduce 的结果等于逐元素求和；每个 rank 发出 2(N−1)/N · D 个元素
for n, D in [(4, 1200), (8, 4096), (16, 1 << 14)]:
    ring = Ring(n)
    data = [rng.normal(size=D) for _ in range(n)]
    out = ring.all_reduce(data)
    assert all(np.allclose(o, sum(data)) for o in out)
    assert all(x == 2 * (n - 1) * D // n for x in ring.sent)   # D 能被 n 整除时精确相等
    print(n, D, ring.sent[0], round(ring.sent[0] / D, 4), round(2 * (n - 1) / n, 4))

# 每个 rank 的发送量几乎和 N 无关：N → ∞ 时趋于 2D，这是 ring 适合大规模 DP 的原因
assert 2 * (1024 - 1) / 1024 > 1.99

# 2) ZeRO 的通信量：按 ZeRO 论文的口径（每个 rank 收发约 Ψ 个元素记为 Ψ）
psi, n = 1 << 12, 8
grads = [rng.normal(size=psi) for _ in range(n)]

dp = Ring(n); dp.all_reduce(grads)                         # 普通 DP / ZeRO-1：all-reduce 梯度
z2 = Ring(n)                                               # ZeRO-2：reduce-scatter 梯度，更新后 all-gather 参数
owned = z2.reduce_scatter(grads)
params = z2.all_gather({r: (c, np.zeros_like(v)) for r, (c, v) in owned.items()}, psi)
z3 = Ring(n)                                               # ZeRO-3：前向、反向各 all-gather 一次参数，再 reduce-scatter 梯度
shard = {r: ((r + 1) % n, np.array_split(np.zeros(psi), n)[(r + 1) % n]) for r in range(n)}
z3.all_gather(shard, psi); z3.all_gather(shard, psi); z3.reduce_scatter(grads)

unit = (n - 1) / n * psi                                   # 一次 reduce-scatter 或 all-gather 的发送量
assert dp.sent[0] == z2.sent[0] == 2 * unit                # 2Ψ
assert z3.sent[0] == 3 * unit                              # 3Ψ，是基线的 1.5 倍
assert z3.sent[0] / dp.sent[0] == 1.5
print("ok", dp.sent[0], z2.sent[0], z3.sent[0])
```

### 4.3 FSDP 和 ZeRO-3 的关系

出处：Zhao 等，arXiv 2304.11277。

FSDP 是 PyTorch 原生的分片数据并行。论文说它受 ZeRO 启发，设计上和 PyTorch 的 autograd、缓存分配器等组件对齐。全分片时切的是参数、梯度、优化器状态这三样，和 ZeRO-3 相同。

FSDP 的工作方式：

- **FSDP unit**：模型按 unit 切分，通常一个 Transformer 层包成一个 unit。
- **FlatParameter**：每个 unit 的参数展平拼接成一个一维张量，右侧补齐到能被分片数 F 整除（补齐最多 F − 1 个元素），再均分给各卡。NCCL 的 all-gather 和 reduce-scatter 要求各 rank 输入大小相同，均分后集合通信不需要额外拷贝。
- **一步的通信**：前向前对当前 unit 做 all-gather 拿到完整参数，算完释放；反向前再 all-gather 一次；反向后对梯度做 reduce-scatter，每卡只留自己那份。和 ZeRO-3 的三次通信一一对应。论文写明：全分片用带宽最优的 ring 算法时，通信量是 DDP 的 1.5 倍。
- **分片因子 F**：F = 1 退化成 DDP（梯度 all-reduce）；F = W（总卡数）是全分片；1 < F < W 是混合分片，组内分片、组间复制，跨机通信更少。
- **预取**：反向时先发下一个 unit 的 all-gather，再发当前 unit 的 reduce-scatter，让通信和计算重叠。前向也可以按上一步记下的执行顺序预取。
- **限流**：CPU 跑得太快时，会提前发出很多 all-gather，缓存分配器来不及复用显存块。FSDP 的 rate limiter 最多允许两个在途的 all-gather，这是还能做到通信计算重叠的最小数。

**峰值显存**约等于分片后的模型状态，加上同时展开的一到两个 unit 的完整参数和梯度，再加激活。unit 切得越细峰值越低，但集合通信次数越多，可能降吞吐。

**FSDP2**（`fully_shard`）是新一代接口。PyTorch 源码的 docstring 写明：分片参数用在第 0 维切开的 DTensor 表示，按参数逐个切，不再拼成 FlatParameter；`reshard_after_forward=True` 时前向后释放完整参数、反向再 all-gather，`False` 时保留完整参数省掉反向那次 all-gather，代价是显存。设成整数时，前向后只在更小的组里重新分片（例如单机内），反向的 all-gather 走机内带宽。verl 的 README 推荐用 FSDP2。

| | ZeRO-3（DeepSpeed） | FSDP（FSDP1） | FSDP2 |
|---|---|---|---|
| 切什么 | 参数、梯度、优化器状态 | 同左（全分片时） | 同左 |
| 分片单位 | 参数分区 | 每个 unit 一个 FlatParameter | 每个参数一个 DTensor，第 0 维切 |
| 通信量 | 3Ψ | 全分片时 DDP 的 1.5 倍 | 同左 |
| 混合分片 | 本章不展开 | 分片因子 1 < F < W | 二维 mesh，`reshard_after_forward` 可设整数 |

## 5 RL 框架里训练与推理引擎的协同

### 5.1 为什么要两套引擎

RL 每轮要先生成（推理）再更新（训练）。生成适合用 vLLM、SGLang 这类推理引擎：PagedAttention、连续批处理、CUDA Graph 都是为自回归解码设计的。训练要用 FSDP 或 Megatron 这类训练引擎。两者对同一个 actor 的最优切分不同，权重每轮都要从训练端同步到推理端。

HybridFlow（Sheng 等，arXiv 2409.19256）给出了切分不同的原因：训练受算力限制，模型并行度大更好；生成受显存带宽限制（原因见 [inference/01 第 2.3 节](../../inference/01-kv-cache-memory/README.md#23-每步读写多少字节为什么-decode-受带宽限制)），同样的切分会让 GPU 吃不饱，更小的 TP 和 PP、更多副本更好。代价是两种切分之间要重分片。

### 5.2 3D-HybridEngine：共卡时的重分片

HybridFlow 的做法是训练和生成用同一批卡、同一份权重。记号：训练用 p-t-d 切分（p 段流水线、t 路张量并行、d 个副本）；生成用 p_g-t_g-d_g-d，其中 d_g = pt/(p_g t_g) 叫 micro DP 组大小，训练里的每个副本在生成时变成 d_g 个副本。

**朴素分组的问题**（论文称 HybridFlow-V）。论文图 8 的例子：8 张卡，训练 1-4-2，TP 组是 [G1, G2, G3, G4] 和 [G5, G6, G7, G8]。生成改成 TP = 2，如果生成 TP 组还取相邻的 rank，[G1, G2]、[G3, G4]……，那么切换时每张卡要在整个训练 TP 组里 all-gather 出完整参数，再只留自己那份。G2、G3、G6、G7 留下的生成分片和自己的训练分片不重叠，训练分片还得另存一份。

**HybridFlow 的分组**：生成 TP 组隔着取 rank，[G1, G3]、[G2, G4]、[G5, G7]、[G6, G8]；micro DP 组取相邻 rank，[G1, G2]、[G3, G4]……。这样每张卡的训练分片正好落在它的生成分片里，只需在 micro DP 组内各做一次 all-gather，几个组并发进行。

论文表 2 的对比（M 是模型参数大小）：

| | DS-Chat | HybridFlow-V | HybridFlow |
|---|---|---|---|
| 通信量 | (tpd−1)/(tpd) · M | (tp−1)/(tp) · M | (tp − t_g p_g)/(t_g p_g tp) · M |
| 参数峰值显存 | M | M | M/(t_g p_g) |
| 冗余 | M/(tpd) | M/(tp) | 0 |

下面的代码在论文图 8 的配置上按两种分组算一遍，核对表 2 的三行。

```python
import numpy as np

# HybridFlow 图 8 的例子：8 张卡，训练 p-t-d = 1-4-2，生成 p_g-t_g-d_g-d = 1-2-2-2
N, tp, tg = 8, 4, 2
dg = tp // tg                                 # micro DP 组大小 = 训练 TP / 生成 TP
M = 16                                        # 模型参数单位数（每单位 1 份显存）
W = np.arange(M)
train_chunk = lambda g: g % tp                # 训练时第 g 张卡持有第 g % 4 块（每块 M/4）
chunks = np.split(W, tp)


def gen_shard(i):
    """生成时第 i 个 TP 分片 = 连续 tp/tg 个训练块。"""
    return set(range(i * dg, (i + 1) * dg))


def reshard(gen_rank, gather_group):
    """返回每张卡要从别处收的单位数、峰值参数显存、为训练额外保留的冗余。"""
    recv, peak, redundant = [], [], []
    for g in range(N):
        need = gen_shard(gen_rank(g))
        have = {train_chunk(g)}
        group = gather_group(g)               # all-gather 的参与者
        gathered = {train_chunk(x) for x in group}
        assert need <= gathered
        recv.append(sum(len(chunks[c]) for c in gathered - have))
        peak.append(sum(len(chunks[c]) for c in gathered | have))
        # 自己的训练块不在生成分片里：训练权重得另存一份
        redundant.append(0 if have <= need else len(chunks[train_chunk(g)]))
    return recv, peak, redundant


# 朴素分组（HybridFlow-V）：生成 TP 组取相邻 rank，[G1,G2],[G3,G4]...；在整个训练 TP 组里 all-gather
naive = reshard(gen_rank=lambda g: g % tg, gather_group=lambda g: [g // tp * tp + i for i in range(tp)])
# HybridFlow 分组：生成 TP 组隔 dg 取 rank，[G1,G3],[G2,G4]...；micro DP 组取相邻 rank，只在组内 all-gather
hybrid = reshard(gen_rank=lambda g: (g % tp) // dg, gather_group=lambda g: [g // dg * dg + i for i in range(dg)])

# 论文：朴素分组下 G2、G3、G6、G7（0 起编号 1、2、5、6）的训练块和生成块不重叠
assert [g for g in range(N) if naive[2][g]] == [1, 2, 5, 6]
# 和论文表 2 对上（单位：M）
assert max(naive[0]) / M == (tp - 1) / tp                        # 通信 (tp−1)/tp · M
assert max(naive[1]) == M                                        # 峰值 M（先拼出完整模型）
assert max(naive[2]) / M == 1 / tp                               # 冗余 M/(tp)
assert max(hybrid[0]) / M == (tp - tg) / (tg * tp) == (dg - 1) / tp   # 通信 M/4
assert max(hybrid[1]) / M == 1 / tg                              # 峰值 = 生成分片大小 M/2
assert max(hybrid[2]) == 0                                       # 零冗余
# 生成 TP 组 [G1,G3] 拿到的两个分片正好拼成完整模型
assert gen_shard((0 % tp) // dg) | gen_shard((2 % tp) // dg) == set(range(tp))
print("ok", naive[0], hybrid[0])
```

实现上，论文说 3D-HybridEngine 在 Megatron-LM 和 vLLM 之上写了约 2.4k 行。训练和生成的权重放在不同的缓冲：训练时把生成用的权重卸载到 CPU，切换时再加载回来；用 NCCL 原语在 micro DP 组内收集参数；生成结束后把 KV Cache 卸到 CPU，下一轮再加载。

### 5.3 三个框架的做法

| 框架 | 训练后端 | 推理后端 | 核对过的要点 |
|---|---|---|---|
| verl（HybridFlow 的开源实现） | FSDP、FSDP2、Megatron-LM | vLLM、SGLang、HF Transformers | README 推荐 FSDP2，并说 FSDP2 的 CPU 卸载兼容梯度累积；支持把模型放到不同的 GPU 组上（flexible device mapping）；用 3D-HybridEngine 做 actor 重分片 |
| OpenRLHF（arXiv 2405.11143） | DeepSpeed（README 写 ZeRO-3、AutoTP、RingAttention） | vLLM | 基于 Ray 把 GPU 分给不同角色；论文称可在 hybrid engine 和异步训练等模式间切换；README 里 `--train.colocate_all` 让全部模型和 vLLM 共卡，`--vllm.sync_backend nccl` 指定权重同步走 NCCL，另有 `--vllm.enable_sleep`、`--ds.enable_sleep` 让一方空闲时释放显存 |
| slime（THUDM，GitHub） | Megatron | SGLang（带 router） | 训练、rollout、Data Buffer 三个模块；训练完成后把参数同步给 rollout；README 提到全异步 rollout；非共卡时可用 Delta Weight Sync，只通过共享存储传两次同步之间变化的字节 |

OpenRLHF 的 README 还有一个和异步相关的开关：`--train.partial_rollout_enable` 用 vLLM 的暂停和恢复做权重同步，生成可以和训练重叠，代价是一条在途样本里可能同时有新旧两版权重生成的 token。

### 5.4 权重同步：数据量和几种方式

每轮训练后，推理端要拿到新权重。按 BF16 算要搬 2Ψ 字节，7B 模型是 14 GB。

| 方式 | 适用 | 做法 | 代价 |
|---|---|---|---|
| 共卡、原地重分片 | 训练和生成在同一批卡 | 3D-HybridEngine 式的组内 all-gather | 只搬缺的那部分；切换要等训练做完 |
| NCCL 广播或点对点 | 训练和推理在不同的卡，同一个集群 | 训练端收集完整参数，按推理端的切分发过去 | 每轮搬 2Ψ；推理端要暂停或做部分 rollout |
| 经共享存储 | 跨集群、跨机房 | 写检查点，推理端从磁盘加载；slime 的 delta 模式只写变化的字节 | 慢，但不需要两边网络直连 |

一个推导：训练端用 ZeRO-3 时每卡只有 1/N 的参数，推理端 TP = 1 要完整的 2Ψ，每卡要从别的卡收 `(N−1)/N · 2Ψ`。8 卡、7B 时是 12.25 GB。权重同步时推理端要临时放一份新权重或一个缓冲，这是 [inference/01 第 6 节](../../inference/01-kv-cache-memory/README.md#6-工程细节与常见坑) 里“同步权重时 OOM”的来源。

### 5.5 共卡与分离部署

| | 共卡（colocate） | 分离部署 |
|---|---|---|
| 做法 | 训练和生成轮流用同一批卡 | 训练和生成各占一批卡 |
| 好处 | 不用为推理单独留卡 | 两边可以并行 |
| 代价 | 每轮有重分片、显存释放和重新分配；同一时刻只有一个阶段在跑 | 同步训练时一边干活另一边等；要异步才真正并行，代价是策略滞后（见 [04 章](../04-agentic-rl/README.md#5-异步-rollout-与长尾)） |

HybridFlow 的实验观察：16 到 64 卡上全部模型共卡最好；卡数更多时，把 actor 和 critic 放到不同的卡上并行执行吞吐更高。论文解释：batch 固定时，大集群上共卡的 DP 规模变大，计算通信比下降，吞吐不再线性增长。

### 5.6 数值例子：GRPO 共卡时一张卡的显存怎么分

下面是一个假设配置的推导，不是任何框架的实测。7B 稠密模型（L = 32、h = 4096、8 个 KV 头、head_dim 128），8 张 80 GB 卡，GRPO，训练和生成共卡；每条样本 prompt 1024 token、回答 4096 token；运行时预留 6 GB（假设值）。

- **训练阶段**：actor 用 ZeRO-3 是 14 GB，reference 的 BF16 权重分片 1.75 GB，全量重计算下的激活约 2.1 GB，FP32 logits 约 0.66 GB，加预留约 24.5 GB。
- **生成阶段**：TP = 1，每卡一份完整 BF16 权重 14 GB，每 token 的 KV 是 128 KiB。训练状态全部卸到 CPU，每卡能同时放 89 条 5120 token 的序列；留在卡上只能放 65 条，少了约 27%。
- **权重同步**：每卡要收 12.25 GB。

PPO 还要再加一个 critic 的 16Ψ 训练状态，GRPO 去掉了它，见 [03 章第 2.6 节](../03-grpo-family/README.md#26-和-ppo-比省了什么代价是什么)。

```python
GB, KiB = 1e9, 1024

# 假设：7B 稠密模型（L=32、n_kv=8、head_dim=128），8 张 80 GB 卡，GRPO，训练与生成共卡
psi, N, L, n_kv, d_h = 7e9, 8, 32, 8, 128
prompt, resp = 1024, 4096
reserve = 6 * GB                     # CUDA 上下文、通信缓冲、碎片的预留（假设值）

# 训练阶段（每卡）：actor ZeRO-3 + reference 推理权重分片 + 全量重计算的激活 + logits
actor = 16 * psi / N
ref = 2 * psi / N
sbh = (prompt + resp) * 1 * 4096
act = 2 * sbh * L + 34 * sbh         # 每层只存输入，反向时再展开一层（FlashAttention 口径）
logits = 4 * (prompt + resp) * 32000   # 词表 32000，FP32
train_peak = actor + ref + act + logits + reserve
assert round(train_peak / GB, 1) == 24.5

# 生成阶段（每卡）：TP=1，每卡一份完整 BF16 权重，剩下的给 KV Cache
kv_per_token = 2 * L * n_kv * d_h * 2                 # K、V 各一份，BF16
assert kv_per_token == 128 * KiB
gen_w = 2 * psi


def max_seqs(kept_on_gpu):
    free = 80 * GB - gen_w - reserve - kept_on_gpu
    return int(free // (kv_per_token * (prompt + resp)))


offload_all = max_seqs(0)                             # 训练状态全部卸载到 CPU
keep_states = max_seqs(actor + ref)                   # 训练状态留在卡上
assert (offload_all, keep_states) == (89, 65)
# 留在卡上，生成阶段的并发少了约四分之一；卸载则每轮多两次 PCIe 搬运
assert round(1 - keep_states / offload_all, 2) == 0.27

# 权重同步：ZeRO-3 每卡只有 1/N 的参数，生成要完整的 2Ψ。每卡从别的卡收 (N−1)/N · 2Ψ
recv = (N - 1) / N * 2 * psi
assert recv / GB == 12.25
print("ok", round(train_peak / GB, 2), offload_all, keep_states)
```

## 6 混合精度

出处：Micikevicius 等，arXiv 1710.03740；Kalamkar 等，arXiv 1905.12322。

### 6.1 三种格式的位数和边界

一个浮点数是 `(−1)^符号 × 2^(指数 − 偏置) × 1.尾数`。指数位数决定范围，尾数位数决定精度。

| 格式 | 符号、指数、尾数位 | 最大正规数 | 最小正规数 | 最小次正规数 | 1 附近的间隔 |
|---|---|---|---|---|---|
| FP32 | 1, 8, 23 | 3.40e38 | 1.17e−38 | 1.4e−45 | 2⁻²³ |
| FP16 | 1, 5, 10 | 65504 | 2⁻¹⁴ ≈ 6.10e−5 | 2⁻²⁴ ≈ 5.96e−8 | 2⁻¹⁰ |
| BF16 | 1, 8, 7 | 3.39e38 | 1.17e−38 | 9.2e−41 | 2⁻⁷ |

FP16 的最小正规数是 2⁻¹⁴。比它小的数用次正规数表示，尾数里的有效位越来越少，到 2⁻²⁴ 只剩最后一位。论文说“绝对值小于 2⁻²⁴ 的数在 FP16 里变成 0”。精确地说，就近舍入时小于 2⁻²⁵ 的数舍成 0，2⁻²⁵ 到 2⁻²⁴ 之间的舍成 2⁻²⁴，第 6.5 节的代码核对了这一点。另一头，超过 65504 加半个间隔（65520）就上溢成 inf。

### 6.2 FP32 主权重：两个原因

论文第 3.1 节给了两个原因，都在优化器那一步：

1. **更新量本身下溢**。更新量是学习率乘梯度，可能小于 2⁻²⁴。论文观察到，Mandarin 语音模型的权重梯度里约 5% 的指数小于 −24，乘上学习率后在 FP16 里会变成 0。
2. **更新量加不上去**。即使更新量能表示，权重比更新量大太多时，加法要先把更新量右移对齐小数点，移出尾数就丢了。FP16 尾数 10 位，权重是更新量的 2048 倍以上时可能丢失。例如 `1 + 2⁻¹²` 在 FP16 里还是 1。

修法是在 FP32 副本上累加更新，再转回 FP16 给下一步前向用。第 6.5 节的代码在 FP16 里加 1000 次 2⁻¹²，权重一直是 1；在 FP32 里累加，转回 FP16 后是 1.244。

BF16 尾数只有 7 位，原因 2 更严重。按同样的推理（推导，不是论文结论），权重是更新量的 2⁸ = 256 倍以上就可能丢。所以 BF16 训练仍要 FP32 主权重。

### 6.3 loss scaling：把激活梯度搬进 FP16 的范围

激活梯度常常比权重梯度还小。论文图 3 统计了 Multibox SSD 的激活梯度：67% 是 0，很多非零值落在 FP16 表示不了的范围里；对这个模型，绝对值低于 2⁻²⁷ 的梯度不影响训练，但 [2⁻²⁷, 2⁻²⁴) 区间的必须保住。

做法：前向算完 loss 后乘一个系数 S，按链式法则所有梯度都乘了 S，整体右移进 FP16 的范围，反向本身不用改。更新前在 FP32 里除回 S。要点：

- **除回要在梯度裁剪之前**。先除回 S，裁剪阈值、权重衰减等超参都不用跟着 S 调。
- **S 怎么选**。论文试过的常数系数在 8 到 32K 之间，很多网络不需要 scaling；也可以按梯度统计选，让 S 乘以最大梯度不超过 65504。只要不上溢，S 大一点没有坏处。
- **上溢的处理**。上溢会让梯度出现 inf 和 NaN，一旦进入更新就永久污染权重。论文建议检测到上溢就跳过这一步。

**动态 loss scaling** 是 PyTorch `GradScaler` 的默认做法。默认参数 `init_scale=65536.0`、`growth_factor=2.0`、`backoff_factor=0.5`、`growth_interval=2000`。源码的说明：发现 inf 或 NaN 时跳过 `optimizer.step()`，scale 乘 backoff_factor；连续 growth_interval 步没有跳过，scale 乘 growth_factor。所以训练开头会连跳几步，等 scale 降到合适的值；之后每 2000 步试着翻倍一次，超了再减半，偶尔跳一步是正常的。

### 6.4 BF16 为什么通常不用 loss scaling

BF16 的指数位和 FP32 一样是 8 位，动态范围相同，小梯度不会下溢。Kalamkar 等的实验里，BF16 训练不改超参数；FP16 则要额外调 loss scaling。代价是 BF16 尾数只有 7 位，精度低，所以 FP32 主权重仍然要留。

反过来，PyTorch 文档提醒：大多数 BF16 预训练的模型放不进 FP16 最大 65504 的范围，换成 FP16 微调会上溢而不是下溢，这时 scale 甚至会降到 1 以下。BF16 训练的模型改用 FP16 时要先检查这一点。

### 6.5 实现：FP16 下溢与 loss scaling 的演示

下面用 numpy 的 float16 核对：FP16 的边界；BF16 的范围和精度（用截断 FP32 低 16 位模拟）；一批落在 [2⁻³⁰, 2⁻²⁰] 的梯度约一半下溢成 0，乘 65536 后全部保住，相对误差不超过半个间隔；scale 只乘 1024 时一部分落进次正规区间，精度变差；scale 太大时上溢；GradScaler 规则的模拟；FP32 主权重保住了小更新。

```python
import numpy as np

np.seterr(over="ignore")                                   # 上溢是要演示的现象，不打印警告

f16, f32 = np.float16, np.float32

# 1) FP16 的几个边界：最大值、最小正规数、最小次正规数
fi = np.finfo(f16)
assert float(fi.max) == 65504.0
assert float(fi.tiny) == 2.0 ** -14                       # 最小正规数 ≈ 6.10e-5
assert float(f16(2.0 ** -24)) == 2.0 ** -24               # 最小次正规数 ≈ 5.96e-8，还能表示
assert float(f16(2.0 ** -25)) == 0.0                       # 再小一半：就近舍入（平局取偶）到 0
assert float(f16(2.0 ** -24 * 0.6)) == 2.0 ** -24          # 介于 2^-25 和 2^-24 之间：舍入到 2^-24
assert np.isinf(f16(70000.0))                              # 超过 65504（加半个 ulp）上溢成 inf

# Adam 的二阶矩存 g²：g = 1e-4 时 g² = 1e-8，在 FP16 里直接变 0，所以优化器状态用 FP32
assert float(f16(f32(1e-4) ** 2)) == 0.0 and float(f32(1e-4) ** 2) > 0

# 2) BF16：截掉 FP32 的低 16 位（这里用向零截断，真实硬件一般就近舍入）
def bf16(x):
    u = np.asarray(x, f32).view(np.uint32) & np.uint32(0xFFFF0000)
    return u.view(f32)

assert float(bf16(2.0 ** -100)) == 2.0 ** -100            # 指数 8 位：小梯度不下溢
assert float(bf16(1 + 2.0 ** -7)) == 1 + 2.0 ** -7        # 尾数 7 位：1 后面第 7 位还在
assert float(bf16(1 + 2.0 ** -8)) == 1.0                   # 第 8 位丢了，精度比 FP16（10 位）低

# 3) 激活梯度下溢与 loss scaling：一批落在 [2^-30, 2^-20] 的梯度（对数均匀分布）
rng = np.random.default_rng(0)
g = (2.0 ** rng.uniform(-30, -20, 100_000)).astype(f32)
lost = np.mean(g.astype(f16) == 0)
assert 0.45 < lost < 0.55                                  # 约一半（< 2^-25 的那部分）变成 0

S = f32(2.0 ** 16)                                         # loss 乘 65536：链式法则让梯度同乘 65536
g_scaled = (g * S).astype(f16)
assert np.mean(g_scaled == 0) == 0.0
back = g_scaled.astype(f32) / S                            # 更新前在 FP32 里除回去
assert g_scaled.min() >= 2.0 ** -14                         # 放大后全部落进 FP16 的正规数区间
assert np.max(np.abs(back - g) / g) <= 2.0 ** -11          # 相对误差不超过半个 ulp
# scale 只乘 1024 不够：[2^-20, 2^-10] 里低于 2^-14 的那段落进次正规区间，有效位变少
small = (g * f32(2.0 ** 10)).astype(f16).astype(f32) / f32(2.0 ** 10)
assert np.max(np.abs(small - g) / g) > 2.0 ** -11

# scale 太大：最大梯度乘 S 超过 65504 就上溢成 inf，这一步必须跳过
big = np.array([1e-3, 100.0], f32)
assert np.isinf((big * S).astype(f16)).any()

# 4) 动态 loss scaling（GradScaler 的规则）：出现 inf 跳过本步、scale × 0.5；
#    连续 growth_interval 步正常就 scale × 2
def dynamic_scaling(max_grads, init=65536.0, growth=2.0, backoff=0.5, interval=2000):
    scale, good, skipped, hist = init, 0, 0, []
    for gmax in max_grads:
        if not np.isfinite(f16(gmax * scale)):
            scale *= backoff; good = 0; skipped += 1
        else:
            good += 1
            if good == interval:
                scale *= growth; good = 0
        hist.append(scale)
    return scale, skipped, hist

steps = np.full(10_000, 4.0)                                # 每步最大梯度约 4
scale, skipped, hist = dynamic_scaling(steps)
# 65536 和 32768 乘 4 都上溢；16384 × 4 = 65536 也超过 FP16 的上溢阈值 65520（65504 加半个 ulp），
# 所以前三步都跳过，scale 降到 8192 才稳
assert hist[:3] == [32768.0, 16384.0, 8192.0] and float(f16(65519.0)) == 65504.0
# 稳定后每 2000 步试着翻倍，超了又减半：scale 在可用的最大值附近来回
assert set(hist[3:]) <= {8192.0, 16384.0}
assert skipped == 3 + 4                                     # 前 3 步，加上之后每 2000 步试探失败一次（共 4 次）

# 5) 为什么要 FP32 主权重：权重是更新量的 2048 倍以上时，FP16 加法把更新吞掉
w, upd = f16(1.0), f16(2.0 ** -12)                         # 比值 4096
assert w + upd == w                                         # FP16：1 + 2^-12 == 1，更新丢失
w32 = f32(1.0)
for _ in range(1000):
    w32 += f32(2.0 ** -12)                                  # FP32 主权重里累加
w16 = f16(1.0)
for _ in range(1000):
    w16 = f16(w16 + f16(2.0 ** -12))
assert w16 == 1.0 and abs(float(w32) - (1 + 1000 * 2.0 ** -12)) < 1e-6
assert float(f16(w32)) > 1.2                                # 转回 FP16 后，累积的 0.244 保留下来
print("ok", round(float(lost), 3), float(f16(w32)))
```

### 6.6 RL 里的精度问题

推理引擎和训练引擎的算子实现、精度、并行切分不同，同一个 token 的对数概率会有差异。常见做法是用训练引擎重算旧策略的对数概率，再算重要性比率。现象、排查和处理见 [03 章第 8 节坑 6](../03-grpo-family/README.md#8-工程细节与常见坑)，这里不重复。

## 7 工程细节与常见坑

每条写现象、原因、排查和处理。估算类的坑可以用第 2.4 节的函数复核。

**1. 估算和实测差了 10% 到 20%：每参数不止 16 字节**

- 现象：按 16Ψ 算能放下，实际启动就 OOM；或者 `torch.cuda.max_memory_allocated()` 比估算多出约 2Ψ。
- 原因：16 字节是 ZeRO 论文的口径，FP16 梯度。很多框架用 FP32 累加梯度：Megatron-LM 的参数解析代码里，BF16 训练且主梯度是 FP32（默认）时，会自动打开 `accumulate_allreduce_grads_in_fp32`，梯度按 4 字节存，每参数变成 18 字节（推导）。还有一类是框架额外存一份 BF16 参数给通信或推理端用。
- 排查：打印每个参数和 `.grad`（或框架的 main_grad 缓冲）的 dtype；用 `memory_allocated()` 分别在模型构建后、优化器第一步后读一次，两者之差就是优化器状态加梯度缓冲。
- 处理：按框架的实际 dtype 重算；显存紧时可以考虑 BF16 梯度归约或更高的 ZeRO 级别，前者要验证收敛。

**2. 激活的数量级估错：忘了 s²，或者忘了 logits**

- 现象：s 从 4K 加到 16K，显存不是涨 4 倍而是十几倍，直接 OOM；或者显存峰值出现在最后的 loss 计算，而不是中间层。
- 原因：不开 FlashAttention 时，注意力矩阵那项和 s² 成正比（第 2.2 节）。FP32 logits 是 `4sbV` 字节，词表 15 万、s = 32K、b = 1 时约 19.7 GB（推导：4 × 32768 × 150000），算交叉熵时可能还要再复制一份。
- 排查：用 `torch.cuda.memory._record_memory_history()` 抓快照，看峰值落在哪个算子；对比开关 FlashAttention 的峰值。
- 处理：确认 FlashAttention 或 SDPA 的高效实现确实生效；logits 用分块交叉熵（按 token 或按词表切块算），不一次物化整个 s×V 矩阵。

**3. 显存“够”，但还是 OOM：碎片和缓存分配器**

- 现象：`memory_allocated` 明显低于卡的容量，`memory_reserved` 却接近上限，报错里写着“reserved but unallocated”很大。
- 原因：张量的生命周期各不相同，空闲块大小对不上新的申请。ZeRO 论文观察到训练很大的模型时碎片很严重，极端情况下还有 30% 以上显存空闲也会 OOM。PyTorch 的缓存分配器默认不把释放的块还给系统（除非调用 `empty_cache()`），`nvidia-smi` 里看到的是 reserved，不是实际占用。RL 里每轮在训练和生成之间切换，申请模式大变，更容易碎。
- 排查：对比 `memory_allocated()` 和 `memory_reserved()`；看 `memory_summary()` 里的碎片统计。
- 处理：PyTorch 文档里的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 针对申请大小经常变化的场景（文档标为实验性）；阶段切换后调用 `empty_cache()`；FSDP 用限流避免同时展开太多 unit。

**4. ZeRO-3 或 FSDP 吞吐低：通信没有藏在计算后面**

- 现象：GPU 利用率低，profiler 里计算 kernel 之间有空隙，NCCL 的 all-gather 和计算串行。
- 原因：ZeRO-3 的通信是 3Ψ，只有和计算重叠才便宜。unit 切得太细，每次通信量小、次数多，启动开销占主导；没开预取，下一层的 all-gather 要等这一层算完才发；跨机带宽低时，全分片每一步都跨机。
- 排查：nsys 或 PyTorch profiler 看通信流和计算流是否重叠；统计每步 all-gather 次数和单次大小。
- 处理：开前向和反向预取；按 Transformer 层包 unit；中等模型用混合分片（组内分片、组间复制），或 FSDP2 的 `reshard_after_forward` 设成单机卡数；根节点设 `reshard_after_forward=False`。

**5. TP 跨机后吞吐断崖下降**

- 现象：TP 从 8 加到 16，每卡吞吐降了一半以上，扩展效率远低于 DP。
- 原因：TP 每层前向、反向各 2 次 all-reduce，每次 s·b·h 个元素，且在关键路径上（第 3.2 节）。机间网络带宽比机内 NVLink 低得多，跨机后每层都卡在通信上。Megatron 论文的结论是 TP 超过单机卡数后，跨机 all-reduce 会拖慢整个计算。
- 排查：profiler 看 TP 通信占每层时间的比例；对比 TP 8 加 PP 2 与 TP 16 的吞吐。
- 处理：TP 不超过单机卡数，跨机用 PP 或 DP（第 3.7 节）。

**6. 流水线气泡比公式大**

- 现象：按 `(p−1)/m` 算气泡应该 10%，实测 GPU 空闲 25%。
- 原因：公式假设每段的 t_f、t_b 相同。第一段有嵌入层、最后一段有输出层和 loss，词表大时最后一段明显更慢，节奏由最慢的段决定。还有一种是把 m 设得太小，或者交错调度时 m 不是 p 的整数倍。
- 排查：分别量每段的前向和反向时间；看 timeline 上哪一段一直在忙、其他段在等。
- 处理：给首末两段少分几层；增大 m（代价是 micro-batch 变小或 batch 变大）；用交错调度，接受 v 倍的点对点通信。

**7. FP16 训练 loss 变成 NaN，或者 loss scale 一路降到很小**

- 现象：日志里 GradScaler 的 scale 不断减半，跳过的步数越来越多；或者 loss 突然 NaN。
- 原因：scale 太大导致梯度上溢；或者模型本身的激活就超出 65504（BF16 预训练的模型换 FP16 微调时常见，第 6.4 节），scale 降到 1 以下也救不回来。另一类是在除回 scale 之前做了梯度裁剪，阈值被放大了 S 倍，等于没裁剪。
- 排查：记录每步的 scale 和是否跳过；检查前向激活的最大绝对值；确认 `scaler.unscale_(optimizer)` 在 `clip_grad_norm_` 之前。
- 处理：硬件支持时换 BF16；FP16 时按 6.3 节的顺序除回再裁剪。

**8. 共卡 RL 生成阶段并发远低于预期，或权重同步时 OOM**

- 现象：推理引擎日志里的 KV token 数只有估算的一半；切换到生成后第一次同步权重就 OOM。
- 原因：训练状态（优化器状态、梯度、FP32 主权重）没卸载，占着显存（第 5.6 节：89 条变 65 条）；推理引擎按启动时的空闲显存定 KV 大小，之后训练进程又申请了显存；权重同步要临时放一份新权重或收发缓冲。
- 排查：在“训练结束、生成开始”的时刻读 `memory_allocated`；对比推理引擎启动日志里的 KV 容量和你的估算。
- 处理：确认优化器状态和梯度在生成阶段已卸载或释放（verl 的 offload 选项、OpenRLHF 的 sleep 选项）；推理引擎的显存占比要扣掉同步缓冲；按 tensor 分桶同步，不一次收集完整模型。

## 8 对比与取舍

### 8.1 几种显存手段各省什么

| 手段 | 省什么 | 每卡显存 | 额外代价 |
|---|---|---|---|
| ZeRO-1 | 优化器状态 | 4Ψ + 12Ψ/N_d | 几乎没有，通信仍是 2Ψ |
| ZeRO-2 | + 梯度 | 2Ψ + 14Ψ/N_d | 几乎没有，通信仍是 2Ψ |
| ZeRO-3 / FSDP 全分片 | + 参数 | 16Ψ/N_d | 通信 3Ψ，要靠预取和计算重叠 |
| 张量并行 | 模型状态和块内激活 | 模型状态除以 t | 每层 4 次 all-reduce，要机内带宽 |
| 序列并行（配合 TP） | 剩下的 10sbh 激活 | 整层激活除以 t | 不增加通信量 |
| 流水线并行 | 模型状态按段切 | 模型状态除以 p；激活不除 | 气泡 (p−1)/m |
| FlashAttention / 选择性重计算 | 注意力矩阵 5as/h | 每层 34sbh | 少量重算 |
| 全量重计算 | 几乎全部激活 | 每层 2sbh | 多一次前向，论文报告 30% 到 40% 时间 |
| CPU 卸载 | 优化器状态等 | 视卸载内容 | PCIe 带宽，常常拖慢一步 |

### 8.2 什么情况先用哪个

- **模型状态放得下，只想更快**：先加 DP。通信每个 batch 一次，最容易扩展。
- **模型状态放不下，单机内**：ZeRO-2 加 FlashAttention 通常先试；还不够再上 ZeRO-3 或 FSDP 全分片。ZeRO-1、ZeRO-2 几乎免费，ZeRO-3 的代价看卡间带宽。
- **激活放不下**：先开 FlashAttention，再选择性重计算，最后全量重计算。用了 TP 就一定加序列并行，它不多通信。
- **模型大到要跨机切**：TP 用到单机卡数，跨机加 PP，剩下的卡做 DP（PTD-P）。
- **RL 小集群（几十卡以内）**：共卡加 3D-HybridEngine 式重分片，训练阶段卸载生成权重和 KV，生成阶段卸载训练状态。
- **RL 大集群或生成长尾严重**：训练和生成分开放，配合异步 rollout，接受策略滞后，见 [04 章](../04-agentic-rl/README.md#5-异步-rollout-与长尾)。
- **FP16 还是 BF16**：硬件支持时用 BF16，不用 loss scaling；只有 FP16 时用动态 loss scaling。两者都留 FP32 主权重。

### 8.3 面试时怎么串起来讲

1. 先算账：16Ψ 的模型状态加 `sbh(34 + 5as/h)·L` 的激活，再加 logits 和缓冲。
2. 模型状态大，用 ZeRO 切：说出三级各切什么，通信 2Ψ、2Ψ、3Ψ，背后是 all-reduce = reduce-scatter + all-gather。
3. 激活大，用重计算和序列并行：说出它们各去掉公式里的哪一项。
4. 单卡放不下一层，用 TP：列切、行切，每层 4 次 all-reduce，所以只在机内。
5. 跨机，用 PP：气泡 (p−1)/m，1F1B 省显存不省气泡，交错调度用 v 倍通信换 1/v 气泡。
6. RL 特有：训练和生成的最优切分不同，要重分片和同步权重；共卡和分离部署各有代价。

## 9 自测清单

- [ ] 写出每参数 16 字节的构成，说出 K = 12 是哪三项，说出 FP32 梯度累加时变成多少
- [ ] 逐项列出一层要存的激活，推出 34 和 5as/h，再说出 10 和 24/t 怎么来
- [ ] 用 `sbh(34 + 5as/h)` 估一个 7B 配置的激活，说出 FlashAttention、选择性重计算、全量重计算各省了什么
- [ ] 推出 ring all-reduce 每卡发送 2(N−1)/N · D
- [ ] 写出 ZeRO-1/2/3 的每卡显存公式，推出 2Ψ、2Ψ、3Ψ，算出 7.5B、64 卡的四个数
- [ ] 说出 TP 切 MLP 时 A 按列、B 按行的原因，写出 f、g 两个算子，说出每层 all-reduce 次数
- [ ] 说出序列并行为什么不增加通信
- [ ] 推出 PP 的气泡占比，说出 1F1B 和交错调度各改了什么
- [ ] 说出 FSDP 的 FlatParameter、三次通信、分片因子，以及 FSDP2 的变化
- [ ] 讲清 3D-HybridEngine 的分组为什么能零冗余，比较共卡和分离部署
- [ ] 写出 FP16 的最大值、最小正规数、下溢阈值，说出 loss scaling 的顺序和动态规则
- [ ] 说出至少五个训练显存或吞吐的坑，以及每个的现象和排查方法

## 资料

- Rajbhandari 等，[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，arXiv 1910.02054，2019-10（3.1 节模型状态、5 节 ZeRO-DP、7 节通信量）
- Shoeybi 等，[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)，arXiv 1909.08053，2019-09（3 节列切和行切、f 和 g 算子）
- Narayanan 等，[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)，arXiv 2104.04473，2021-04（2.2 节气泡与交错调度，3 节 PTD-P 的取舍）
- Korthikanti 等，[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)，arXiv 2205.05198，2022-05（4 节激活推导、序列并行，5 节选择性重计算，表 2）
- Zhao 等，[PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)，arXiv 2304.11277，2023-04
- Dao 等，[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，arXiv 2205.14135，2022-05
- Micikevicius 等，[Mixed Precision Training](https://arxiv.org/abs/1710.03740)，arXiv 1710.03740，2017-10（3.1 节主权重，3.2 节 loss scaling）
- Kalamkar 等，[A Study of BFLOAT16 for Deep Learning Training](https://arxiv.org/abs/1905.12322)，arXiv 1905.12322，2019-05
- Sheng 等，[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)，arXiv 2409.19256，2024-09（5 节 3D-HybridEngine、表 2，7 节实验）
- Hu 等，[OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)，arXiv 2405.11143，2024-05
- PyTorch 文档，[Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)（GradScaler 默认参数、BF16 模型换 FP16 的提醒）和 [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)（缓存分配器、`expandable_segments`），2026-11 查阅
- PyTorch 源码，[`torch/amp/grad_scaler.py`](https://github.com/pytorch/pytorch/blob/main/torch/amp/grad_scaler.py) 和 [`torch/distributed/fsdp/_fully_shard/_fully_shard.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fully_shard.py)（FSDP2 的 DTensor 分片与 `reshard_after_forward`），main 分支，2026-11 查阅
- Megatron-LM 源码，[`megatron/training/arguments.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/arguments.py)（BF16 下梯度 FP32 累加），main 分支，2026-11 查阅
- verl，[GitHub README](https://github.com/verl-project/verl)，2026-11 查阅
- OpenRLHF，[GitHub README](https://github.com/OpenRLHF/OpenRLHF)，2026-11 查阅
- slime，[GitHub README](https://github.com/THUDM/slime) 和 [Delta Weight Sync 文档](https://github.com/THUDM/slime/blob/main/docs/en/advanced/delta-weight-sync.md)，2026-11 查阅
