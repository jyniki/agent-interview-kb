# GRPO 及其变体：Dr. GRPO、DAPO、GSPO

> 层级：垂类 · training · 预计 210 分钟 · 题目：[questions.md](questions.md)
> 前置：[core/12 强化学习入门](../../../core/12-rl-basics/README.md)（策略梯度与基线、PPO 的 clip 目标、KL 估计量）
> 相关：[04 Agentic RL](../04-agentic-rl/README.md)（多轮 rollout、工具返回的 loss mask）、[05 奖励与 Verifier](../05-reward-verifier/README.md)（奖励从哪来）、[07 RL 训练框架](../07-training-systems/README.md)（训练和推理引擎怎么协同）

## 1 这一章解决什么问题

PPO 用价值模型（critic）给每个 token 估一个基线。LLM 场景里这有两个麻烦。第一，critic 通常和策略同规模，要多存一份权重、梯度和优化器状态。第二，奖励往往只在回答结尾给一次，训一个在每个 token 上都准的价值函数很难。

GRPO（Group Relative Policy Optimization）换了一个思路：同一个问题采 G 个回答，用这组回答的平均奖励当基线，不再训练 critic。它简单、省显存，2025 年以来成了推理模型和 Agentic RL 的常用起点。

原始 GRPO 的目标里有几处看起来无害的归一化和近似。规模一大，它们就表现为具体的问题：错误回答越写越长，熵快速下降，没有梯度的组越来越多，MoE 模型训练崩溃。Dr. GRPO、DAPO、GSPO 各修了其中一部分。

学完你应该能：

- 从带基线的策略梯度推出 GRPO 的目标，说清组内均值作为基线在什么条件下无偏；
- 逐项解释 DeepSeekMath 式 3，推导 k3 KL 估计量为什么无偏且非负；
- 用式子和数字说明长度偏差和难度偏差，讲清 Dr. GRPO、DAPO、GSPO 每处改动在数学上改了什么；
- 只用 numpy 写出 GRPO 的优势和 loss（含 clip、KL、response mask），并用它验证上面的结论；
- 知道训练时会看到哪些现象，出了问题从哪里查。

下图是一步 GRPO 训练的数据流，虚线标出各变体改动的位置。

```mermaid
flowchart LR
  A[问题 q] --> B[旧策略采 G 个回答]
  B --> C[Verifier 给每个回答打分]
  C --> D[组内优势]
  D --> E[token 级比率与 clip]
  E --> F[聚合成一个 loss]
  F --> H[反向传播更新策略]
  B -.->|DAPO 动态采样，过滤全对和全错的组| C
  D -.->|Dr. GRPO 不除 std| E
  E -.->|DAPO 调大上界，GSPO 改成序列级比率| F
  F -.->|DAPO 除以总 token 数，Dr. GRPO 除以常数| H
```

## 2 原始 GRPO：从策略梯度到组内基线

出处：[Shao 等，DeepSeekMath，arXiv 2402.03300](https://arxiv.org/abs/2402.03300)，2024-02，第 4.1 节和附录 A.1.6。精读见 [papers/02](../../../papers/02-deepseekmath-grpo.md)。

### 2.1 起点：带基线的策略梯度

记号：q 是问题；o 是回答，o_t 是它的第 t 个 token，o_<t 是它前面的 token；|o| 是回答的 token 数；π_θ 是参数为 θ 的策略；R(q, o) 是整个回答的奖励，例如答案对为 1、错为 0。

目标是最大化期望奖励 J(θ) = E_{o~π_θ(·|q)} [R(q, o)]。用对数导数技巧（推导见 [core/12 第 3 节](../../../core/12-rl-basics/README.md#3-策略梯度与基线)）得到：

```text
∇_θ J = E_{o~π_θ} [ Σ_t ∇_θ log π_θ(o_t | q, o_<t) · ( R(q, o) − b ) ]
```

b 是基线。只要 b 不依赖被求导的那个动作，就有 E[∇log π · b] = b · ∇Σπ = b · ∇1 = 0。减去它不改变梯度的期望，只降低方差。结果奖励只在结尾给一次，所以每个 t 的“之后的回报”都等于 R(q, o)。PPO 用 critic 估计的 V(s_t) 做 b。

PPO 在此基础上做两件事（[core/12 第 5 节](../../../core/12-rl-basics/README.md#5-ppo限制每次更新的幅度)）。一是用重要性比率 ρ_t = π_θ(o_t | …) / π_θold(o_t | …) 复用旧策略 π_θold 采的样本。二是用 clip 限制每次更新的幅度，ε 是 clip 范围：

```text
L_clip,t = min( ρ_t A_t , clip(ρ_t, 1−ε, 1+ε) A_t )
```

GRPO 保留这个 clip 目标，只换掉优势 A_t 的来源。

### 2.2 组内均值为什么是合理的基线

对同一个 q，从 π_θold 独立采 G 个回答 o_1 … o_G，奖励记为 r_1 … r_G。GRPO 的结果监督优势（论文 4.1.2 节）是：

```text
Â_i = ( r_i − mean(r) ) / std(r)          回答 i 的所有 token 共享这个值
```

先只看分子 r_i − mean(r)。mean(r) 里含有 r_i 本身，而 r_i 依赖 o_i，所以它不是严格意义上“和动作无关”的基线。把它拆开：

```text
r_i − mean(r) = r_i − (1/G) r_i − (1/G) Σ_{j≠i} r_j
              = ((G−1)/G) · ( r_i − (1/(G−1)) Σ_{j≠i} r_j )
              = ((G−1)/G) · ( r_i − b_{−i} )
```

b_{−i} 是去掉自己之后其他 G−1 个回答的平均奖励（留一均值，leave-one-out）。其他回答和 o_i 独立采样，所以 b_{−i} 不依赖 o_i，是合法的基线。括号里正是 RLOO（REINFORCE Leave-One-Out）的优势。Dr. GRPO 论文附录 A 写出了同一个等式：G/(G−1) · Ã_i = Â_i^RLOO。

结论：用组内均值做基线，梯度的期望等于真实策略梯度乘常数 (G−1)/G。方向不变，常数可以并进学习率。第 6.1 节的代码穷举所有分组，验证了这一点。

再看分母 std(r)。它同样由这组奖励算出，而且随 r_i 变化。除以它不再是乘一个常数，每组被乘上不同的系数，期望梯度的方向会偏。这就是 Dr. GRPO 说的难度偏差，第 3 节展开。

无偏性成立要满足三个条件：

1. 同组的 G 个回答独立采样。如果同一个问题的回答之间共享前缀，或者被去重过，留一均值就和 o_i 相关了。
2. 基线只用奖励的均值，不除以依赖本组奖励的量。
3. 每个回答的权重和它自己的长度无关。原始 GRPO 的 1/|o_i| 违反这一条，见第 3.1 节。

### 2.3 完整目标：DeepSeekMath 式 3

```text
J_GRPO(θ) = E[ q ~ P(Q), {o_i}_{i=1..G} ~ π_θold(O|q) ]
    (1/G) Σ_{i=1..G} (1/|o_i|) Σ_{t=1..|o_i|} {
        min( ρ_{i,t} Â_{i,t} , clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t} ) − β · D_KL[π_θ ‖ π_ref]
    }
ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_θold(o_{i,t} | q, o_{i,<t})
```

逐项看：

| 项 | 含义 | 说明 |
|---|---|---|
| E[q, {o_i}] | 先抽一个问题，再用旧策略采一组 G 个回答 | 期望是对“问题 × 一整组回答”取的，不是对单个回答 |
| 1/G · Σ_i | 组内 G 个回答取平均 | 每个回答的总权重相同 |
| 1/\|o_i\| · Σ_t | 回答内对 token 取平均 | 长度偏差的来源，见 3.1 |
| ρ_{i,t} | 新旧策略在这个 token 上的概率比 | 一批数据只更新一次时恒为 1 |
| Â_{i,t} | 优势 | 结果监督下 Â_{i,t} = Â_i，同一回答所有 token 相同 |
| min(…, clip(…)) | PPO 的截断目标 | 比率走出 [1−ε, 1+ε] 且继续朝有利方向走时，梯度为 0 |
| β · D_KL | 对参考模型 π_ref 的 KL 惩罚 | π_ref 通常是 RL 开始时的 SFT 模型；论文 β = 0.04 |

和 PPO 相比改了两处。第一，优势由组内相对奖励算出，不需要 critic。论文的理由之一是奖励模型本来就用同一问题下不同回答的比较数据训练，组内相对的算法和它一致。第二，KL 直接加在 loss 里，不加进每个 token 的奖励，这样 Â 的计算不受 KL 影响。

过程监督（4.1.3 节）的版本把每个推理步骤的奖励一起标准化，token 的优势是它之后所有步骤标准化奖励之和。本章只讨论结果监督。奖励怎么设计见 [05 章](../05-reward-verifier/README.md)。

### 2.4 k3 KL 估计量：为什么无偏且非负

GRPO 不算整个词表上的 KL，只在采到的 token 上算一个估计值（论文式 4，引用 [Schulman 2020](http://joschu.net/blog/kl-approx.html)）：

```text
D_KL ≈ π_ref(o_{i,t}|…) / π_θ(o_{i,t}|…) − log( π_ref(o_{i,t}|…) / π_θ(o_{i,t}|…) ) − 1
```

记 x = π_ref / π_θ，估计量就是 k3 = x − 1 − log x。设 token 从 π_θ 采样：

**无偏**。E[x] = Σ_o π_θ(o) · π_ref(o)/π_θ(o) = Σ_o π_ref(o) = 1，所以 E[x − 1] = 0。剩下 E[−log x] = Σ π_θ log(π_θ/π_ref) = KL(π_θ ‖ π_ref)。两项相加：

```text
E_{o~π_θ}[ x − 1 − log x ] = 0 + KL(π_θ ‖ π_ref)
```

x − 1 是一个期望为 0 的控制变量。Schulman 的文章里写成 −log r + λ(r − 1)，k3 取 λ = 1。

**非负**。log 是凹函数，在 x = 1 处的切线是 x − 1，凹函数在切线下方，所以 log x ≤ x − 1，即 k3 ≥ 0，在 x = 1 时取等号。k3 就是 log x 和它切线之间的竖直距离。

对比 k1 = −log x：同样无偏，但单个样本可以为负，方差大。k2 = (log x)²/2 恒非负、方差小，但有偏。[core/12 第 6 节](../../../core/12-rl-basics/README.md#6-kl-约束三种别走太远)有三者的表。第 6.6 节的代码用采样验证了这些性质。

**作为 loss 求导时要小心**。k3 的值是 KL(π_θ ‖ π_ref) 的无偏估计，但它的梯度不是这个 KL 的梯度。把采到的 token 当作固定数据，对单个样本求导：

```text
∇_θ k3 = ( 1 − π_ref/π_θ ) · ∇_θ log π_θ(o)
E_{o~π_θ}[ ∇_θ k3 ] = Σ_o ( π_θ − π_ref ) ∇_θ log π_θ = −Σ_o π_ref ∇_θ log π_θ = ∇_θ KL(π_ref ‖ π_θ)
```

中间一步用了 Σ_o π_θ ∇log π_θ = ∇Σπ_θ = 0。所以把 k3 直接放进 loss，期望梯度对应的是反向 KL(π_ref ‖ π_θ)。它同样会把 π_θ 拉向 π_ref，实践中可用，但不要说成“对 KL(π_θ ‖ π_ref) 做梯度下降”。另外，token 实际是从 π_θold 采的；一批数据更新多次时，这个估计还带有一点 off-policy 偏差。第 6.6 节的代码验证了这个梯度等式。

### 2.5 梯度长什么样：一批数据只更新一次时

DeepSeekMath 的设置是每次采样后只更新一次（4.2 节）。更新前 π_θ = π_θold，ρ = 1，clip 不触发。附录 A.1.6 在这个假设下写出梯度（式 20、21）：

```text
∇_θ J_GRPO = E[ (1/G) Σ_i (1/|o_i|) Σ_t ( Â_{i,t} + β (π_ref/π_θ − 1) ) ∇_θ log π_θ(o_{i,t} | q, o_{i,<t}) ]
```

括号里是每个 token 的梯度系数。它由两部分组成：优势，加上一个 KL 修正。π_θ 比 π_ref 更偏好这个 token 时，π_ref/π_θ < 1，修正为负，往回拉；反之往前推。

clip 只在同一批数据更新多次（论文算法 1 里的 μ > 1）时才起作用。DAPO 每个 rollout 步做 16 次梯度更新，GSPO 把一批 rollout 分成 4 个 mini-batch，这两个设置里 clip 是真的在截断。

### 2.6 和 PPO 比，省了什么、代价是什么

| | PPO | GRPO |
|---|---|---|
| 基线 | critic 逐 token 估计 V(s_t)，用 GAE 算优势 | 同一问题 G 个回答的平均奖励 |
| KL 约束 | 通常作为惩罚加进每个 token 的奖励 | 直接加在 loss 里 |
| 要加载的模型 | 策略、参考、奖励、critic | 策略、参考、奖励（规则奖励时不需要奖励模型） |
| 每题采样数 | 通常 1 个 | G 个，DeepSeekMath 用 64 |

省掉的是一个和策略同规模的 critic：它的权重、梯度和优化器状态。显存按每参数多少字节估算见 [07 章第 2 节](../07-training-systems/README.md#2-显存估算参数梯度优化器状态激活)。

代价有三个。第一，每个问题要采 G 个回答，rollout 开销按 G 倍增长。第二，同组奖励全相同时（全对或全错），优势全为 0，这组只剩 KL 项的梯度。第三，结果监督下同一回答所有 token 的优势相同，没有 token 级的信用分配。多轮场景里的信用分配见 [04 章第 4 节](../04-agentic-rl/README.md#4-多轮信用分配)。

## 3 Dr. GRPO：去掉两处归一化

出处：[Liu 等，Understanding R1-Zero-Like Training: A Critical Perspective，arXiv 2503.20783](https://arxiv.org/abs/2503.20783)，2025-03，第 3.1、3.2 节和附录 A。精读见 [papers/05](../../../papers/05-dr-grpo.md)。

论文图 4 把 GRPO 里实际起作用的优势写成“无偏优势被重新加权”：

```text
a_{i,t} = Ã_{i,t} / ( std(R) · |o_i| )        Ã_{i,t} = R(q, o_i) − mean(R)
```

两个分母各带来一种偏差。下面分别用式子说明。

### 3.1 长度偏差：1/|o_i| 让 token 的权重依赖回答长度

取 μ = 1、β = 0，π_θ = π_θold，式 3 对回答 i 第 t 个 token 的 log 概率求导，得到它的梯度系数：

```text
c_{i,t} = Â_i / ( G · |o_i| )
```

同一回答的 |o_i| 个 token 系数相同，加起来是 Â_i / G，和长度无关。也就是说，GRPO 给每个回答分配同样的“总量”，再平摊到它的 token 上。回答越长，每个 token 分到的越少。

分正确和错误两种情况看：

- **正确回答（Â_i > 0）**。两条都对的回答，短的那条每个 token 被推得更高。策略偏向简短的正确回答。
- **错误回答（Â_i < 0）**。两条都错的回答，长的那条每个 token 被压得更少。策略在答错时偏向写长。

换个角度看期望。把同一回答里的 token 梯度加起来，Σ_t ∇log π_θ(o_{i,t} | …) = ∇log π_θ(o_i | q)。于是 GRPO 的梯度估计变成：

```text
(1/G) Σ_i ( Â_i / |o_i| ) · ∇_θ log π_θ(o_i | q)
```

把基线当作常数时，它是 E[(R − b)/|o|] 的梯度。GRPO 优化的不是期望奖励，而是“按长度折算后的”奖励：答对时越短越好，答错时越长越不亏。第 6.3 节的第一段代码在一个只有四种回答的玩具问题上精确算出了这个期望梯度。

训练里的表现：奖励上升放缓后，GRPO 的回答长度仍在涨；论文图 5 把正确和错误回答分开统计，GRPO 下错误回答明显更长。

论文还指出，这个偏差在 GRPO 之前就存在。trl、OpenRLHF、verl 等开源 PPO 实现都按回答长度归一化 loss（表 2、代码清单 1）：OpenRLHF 式的写法先按每条回答的 token 数求平均（`masked_mean(loss, mask, dim=-1).mean()`），trl、verl 式的写法按整批 token 数求平均。作者推测它沿用了预训练的习惯：预训练把 token 打包进固定长度的上下文，按长度求平均没有问题；RL 里回答长度不是常数，同样的写法就引入了偏差。

### 3.2 难度偏差：std 归一化放大难度极端的题

设奖励只取 0 和 1，一组 G 个回答里答对的比例是 p（组内通过率）。那么 mean(r) = p，总体标准差 std(r) = √(p(1−p))。

| | 正确回答的优势 | 错误回答的优势 | 一组优势绝对值之和 |
|---|---|---|---|
| 不除 std：r − mean | 1 − p | −p | 2G · p(1−p) |
| 除以 std | √((1−p)/p) | −√(p/(1−p)) | 2G · √(p(1−p)) |

用“一组优势绝对值之和”衡量这道题在更新里占多大分量。除以 std 后，每道题的分量被乘上 1/√(p(1−p))。p = 0.5 时这个系数是 2，p 越接近 0 或 1 系数越大。几乎全对和几乎全错的题，在更新里的相对权重被放大了。

具体数字：G = 16，15 对 1 错。不除 std 时那条错误回答的优势是 −15/16 ≈ −0.94；除以 std 后是 −√15 ≈ −3.87。相对“一半对”的题，这道题的权重被放大了约 2.07 倍（两组 std 之比 0.5 / 0.242）。第 6.4 节的代码验证了这些数。

这为什么是偏差？2.2 节说过，除以一个随本组奖励变化的量，不再是乘常数。不同组被乘上不同的系数，期望梯度的方向就变了。第 6.1 节的代码在一个三选一的玩具问题上穷举所有分组，显示除以 std 后期望梯度各分量相对真梯度的倍数不再相同。

另一个副作用：除以 std 让优势对奖励的尺度不敏感。一组奖励 [1, 1, 1, 0.999] 和 [1, 1, 1, 0] 得到几乎一样的标准化优势。奖励里有微小噪声时（例如评分模型的打分抖动），噪声会被放大成和真实差异一样大的信号。

论文的说法是：优势归一化是 RL 的常用技巧，但通常在整个 batch 上做。batch 级归一化对所有样本乘同一个数，不改变相对权重；按问题归一化才会让不同问题的权重不同。

### 3.3 修法：去掉 std，分母换常数

两步：

1. 优势改为 Ã_i = r_i − mean(r)，不除 std。
2. 聚合时，每条回答对 token 求和后除以一个全局常数 MAX_TOKENS（例如生成长度上限），再对回答求平均。

代码清单 1 里的改动只有一行（摘自论文，不可单独运行）：

```python no-run
def masked_mean(tensor, mask, dim):
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)      # 有偏
    # 改为：return (tensor * mask).sum(axis=-1) / MAX_TOKENS       # Dr. GRPO

loss_variant1 = masked_mean(ppo_loss, response_mask, dim=-1).mean()   # 按每条回答长度归一化（如 OpenRLHF）
loss_variant2 = masked_mean(ppo_loss, response_mask, dim=None).mean() # 按整批 token 数归一化（如 trl、verl）
```

改完后每个 token 的梯度系数是 Ã_i / (G · MAX_TOKENS)，只差一个常数。附录 A 推出的梯度（式 6）是：

```text
∇_θ J = E[ (1/G) Σ_i Σ_t ∇_θ log π_θ(o_{i,t} | q, o_{i,<t}) · Ã_{i,t} ]
```

这就是 2.1 节的 REINFORCE 加基线，2.2 节已经说明它等于 RLOO 乘 (G−1)/G。常数取多少只影响梯度范数，相当于调学习率。论文用的设置里 β = 0，不带 KL：规则 Verifier 不存在奖励模型的分布偏移，去掉 KL 还省掉参考模型。

注意 loss_variant2。它按整个 batch 的 token 总数归一化，同一 batch 里每个 token 的权重相同，看起来已经没有长度偏差。论文仍把它列为有偏写法，原因见下一节 DAPO 的对比。

## 4 DAPO：四个工程改进

出处：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale，arXiv 2503.14476](https://arxiv.org/abs/2503.14476)，2025-03，第 2.3 节和第 3 节。精读见 [papers/04](../../../papers/04-dapo.md)。

DAPO 是在长 CoT 推理训练里总结出的一组改动。目标（论文式 8）：

```text
J_DAPO(θ) = E[ (q,a) ~ D, {o_i} ~ π_θold(·|q) ]
    ( 1 / Σ_{i=1..G} |o_i| ) Σ_{i=1..G} Σ_{t=1..|o_i|}
        min( r_{i,t}(θ) Â_{i,t} , clip(r_{i,t}(θ), 1−ε_low, 1+ε_high) Â_{i,t} )
s.t.  0 < |{ o_i | is_equivalent(a, o_i) }| < G
```

a 是标准答案，r_{i,t}(θ) 就是前文的 ρ_{i,t}，Â 仍是除以 std 的组内优势。和式 3 相比改了四处：clip 上下界解耦、加了采样约束、分母改成整组 token 总数、去掉 KL。去掉 KL 的理由（2.3 节）是长 CoT 训练中模型本来就要大幅偏离初始模型，这个约束没有必要。奖励是规则奖励，答对 1、答错 −1。

### 4.1 Clip-Higher：为什么只调大上界

优势为正时，clip 的上界把新概率限制在 π_θ ≤ (1+ε) π_θold 以内。这个上限按比例算，对不同概率的 token 宽严差别很大。论文的例子（3.1 节），ε = 0.2：

| 旧概率 | 上界 (1+ε)·π_old | 实际意义 |
|---|---|---|
| 0.9 | 1.08 | 超过 1，等于不受限 |
| 0.01 | 0.012 | 最多再涨 0.002 |

高概率的“利用”token 可以继续变大，低概率的“探索”token 很难被提上来。结果是策略的熵快速下降，同一组的回答越来越像。论文观察到被上界截断的 token 平均概率低于 0.2（图 3a）。

做法是把上下界分开，只调大上界：ε_low = 0.2，ε_high = 0.28。下界不动，是因为优势为负时下界限制概率的下降 π_θ ≥ (1−ε) π_θold；调大 ε_low 会把这些 token 压到接近 0，采样空间塌缩。

### 4.2 Dynamic Sampling：零优势组从哪来

组内通过率为 p 的题，G 个回答全对或全错的概率是 p^G + (1−p)^G。p = 0.9、G = 8 时约 0.43；G = 16 时约 0.19。训练越往后，模型做对的题越多，零优势组越多（论文图 3b）。

零优势组的代价不只是“这组没有梯度”。在 token 级聚合下，这些组的 token 仍然算在分母里，把其他组的梯度一起稀释了，相当于有效学习率随零优势组比例下降。

做法：过采样，丢掉通过率恰为 0 或 1 的题，直到 batch 填满有效的题。论文认为同步 RL 系统里生成时间由长尾样本决定，多采一些题对总时间影响不大。

### 4.3 Token 级聚合：和 Dr. GRPO 的常数分母差在哪

三种聚合方式下，一个 token 的权重是：

| 聚合 | loss 写法 | 回答 i 中一个 token 的权重 |
|---|---|---|
| GRPO 原版（样本级） | (1/G) Σ_i (1/\|o_i\|) Σ_t ℓ_{i,t} | 1 / (G · \|o_i\|) |
| DAPO（token 级） | Σ_i Σ_t ℓ_{i,t} / Σ_j \|o_j\| | 1 / Σ_j \|o_j\| |
| Dr. GRPO（常数） | (1/G) Σ_i Σ_t ℓ_{i,t} / MAX_TOKENS | 1 / (G · MAX_TOKENS) |

ℓ_{i,t} 是每个 token 的 clip 目标。

**一个 batch 内**，DAPO 和 Dr. GRPO 都让所有 token 的权重相同，不看它在哪条回答里。长回答的总权重正比于它的长度。论文 3.3 节的说法是：某个生成模式只要会提高或降低奖励，不论出现在多长的回答里，都被同等地推高或压低。长回答里的乱码和重复也就罚得动了。

**跨 batch**，两者不同。DAPO 的分母 Σ_j |o_j| 是随机的，取决于这一组（或这一批）所有回答的长度。同一个 token，放在回答普遍较长的组里，权重就更小。期望梯度是 E[分子 / 分母]，不等于 E[分子] 除以一个常数，所以它和真实策略梯度之间仍差一个依赖长度的加权。Dr. GRPO 的分母是常数，期望梯度严格是策略梯度乘常数。这就是 Dr. GRPO 把 loss_variant2 列为有偏的原因。

还有一个容易忽略的差别：分母按什么范围算。DAPO 式 8 的分母写在每个问题内部，是这一组 G 个回答的总长度，因此每道题的总权重相同。实现里常见的是对整个 mini-batch 的 token 求平均（loss_variant2），这时题和题之间也按 token 数加权了。两种写法组内的 token 权重都相同，组间的权重不同。

实际取舍：DAPO 的写法让 loss 的量级稳定，不用为不同长度上限调学习率；Dr. GRPO 的写法在理论上无偏，但梯度尺度随实际长度和 MAX_TOKENS 之比变化，回答普遍很短时梯度会偏小。

### 4.4 Overlong Reward Shaping：截断样本怎么处理

长度上限会截断一部分回答。截断的回答没有最终答案，规则 Verifier 判它错。可它可能推理正确、只是太长。给它负奖励，模型分不清是推理错了还是写长了，这是奖励噪声。

两步处理（3.4 节）：

1. **Overlong Filtering**：截断样本的 loss 直接 mask 掉。论文发现这一步就明显稳定了训练。
2. **Soft Overlong Punishment**（式 13）：在上限前留一段缓冲区，缓冲区内惩罚从 0 线性降到 −1，加到正确性奖励上。

```text
R_length(y) = 0                                   若 |y| ≤ L_max − L_cache
            = ((L_max − L_cache) − |y|) / L_cache  若 L_max − L_cache < |y| ≤ L_max
            = −1                                  若 |y| > L_max
```

L_max 是允许的最大长度，L_cache 是缓冲区长度。论文取期望最大长度 16,384，缓冲区 4,096，生成上限 20,480。第 6.5 节有它的实现。

工具超时造成的截断是环境问题，不应该按长度惩罚，见 [04 章第 2 节](../04-agentic-rl/README.md#2-多轮-rollout-与环境交互)。

## 5 GSPO：把重要性比率放到序列级

出处：[Group Sequence Policy Optimization，arXiv 2507.18071](https://arxiv.org/abs/2507.18071)，2025-07，第 3、4、5 节。精读见 [papers/06](../../../papers/06-gspo.md)。

### 5.1 token 级比率的问题

重要性采样的原理是 E_{z~π_tar}[f(z)] = E_{z~π_beh}[(π_tar(z)/π_beh(z)) · f(z)]（论文式 4）。它要靠从行为分布 π_beh 取大量样本求平均，权重才能把分布校正到目标分布 π_tar。

GRPO 在每个 token 位置算一个比率 ρ_{i,t}。可每个位置的下一个 token 分布只有一个样本。单个样本的比率起不到校正分布的作用，只是一个随机的乘性噪声。噪声随序列长度累积，再被 clip 放大，论文认为这是大模型训练崩溃的原因之一。

论文提炼出一条原则：优化目标的单位要和奖励的单位一致。奖励给的是整个序列，比率也应该按序列算。

### 5.2 序列级比率：定义和为什么开 1/|y| 次方

记号换成 GSPO 论文的：x 是问题，y_i 是回答，π_θ(y_i | x) = Π_t π_θ(y_{i,t} | x, y_{i,<t}) 是整个回答的似然。

```text
J_GSPO(θ) = E[ x ~ D, {y_i} ~ π_θold(·|x) ] (1/G) Σ_i min( s_i(θ) Â_i , clip(s_i(θ), 1−ε, 1+ε) Â_i )     （式 5）

s_i(θ) = ( π_θ(y_i|x) / π_θold(y_i|x) )^(1/|y_i|)
       = exp( (1/|y_i|) Σ_t log( π_θ(y_{i,t}|x, y_{i,<t}) / π_θold(y_{i,t}|x, y_{i,<t}) ) )               （式 7）
```

第二行说明 s_i 是各 token 比率的几何平均。Â_i 仍是除以 std 的组内优势（式 6）。

为什么开 1/|y_i| 次方？不开的话，整句似然比是所有 token 比率的乘积，log 是所有 token log 比率的和。每个 token 的 log 比率有一点噪声，和就随长度线性累积，比率本身指数级放大。几个 token 的变化就能让它剧烈波动，而且长短回答的比率尺度完全不同，需要不同的 clip 范围。开方后取的是平均，数值范围统一。

### 5.3 梯度上的差别

省略 clip，两者的梯度（式 10 和式 12）：

```text
∇J_GSPO = E[ (1/G) Σ_i  s_i(θ) · Â_i · (1/|y_i|) Σ_t ∇_θ log π_θ(y_{i,t} | x, y_{i,<t}) ]
∇J_GRPO = E[ (1/G) Σ_i  Â_i · (1/|y_i|) Σ_t  w_{i,t}(θ) · ∇_θ log π_θ(y_{i,t} | x, y_{i,<t}) ]
w_{i,t}(θ) = π_θ(y_{i,t}|…) / π_θold(y_{i,t}|…)
```

GSPO 的式子来自 ∇s_i = s_i · ∇log s_i = s_i · (1/|y_i|) Σ_t ∇log π_θ(y_{i,t}|…)，其中 π_θold 不含 θ。

区别在 token 梯度的加权：

- GRPO 里每个 token 乘自己的 w_{i,t}。论文指出这些权重在优势为正时落在 (0, 1+ε]，为负时落在 [1−ε, +∞)，彼此不相等，影响会随训练累积。
- GSPO 里同一回答所有 token 乘同一个 s_i。

clip 的粒度也变了。GRPO 逐 token clip，一条回答里有的 token 被截断、有的不被截断。GSPO 按整条回答 clip，比率超出范围就整条回答都不贡献梯度。论文观察到 GSPO 被 clip 掉的 token 比例比 GRPO 高两个数量级（图 2），训练效率反而更高；论文据此认为 GRPO 的 token 级梯度估计本身噪声大。

注意 GSPO 的梯度里仍有 1/|y_i|，即回答内先对 token 求平均。它没有采用 DAPO 的 token 级聚合，也保留了 std 归一化。这几个改动方向互相正交。

### 5.4 clip 范围为什么小几个数量级

log s_i 是 |y_i| 个 token log 比率的平均。粗略估计：如果每个 token 的 log 比率在 0 附近有标准差 σ 的波动，且彼此近似独立，平均后的波动约为 σ/√|y_i|。回答有几千个 token 时，s_i 离 1 的距离比单个 token 比率小一到两个数量级。用 GRPO 的 ε = 0.2 去 clip 它，几乎永远不会触发。

论文实验里 GSPO 的左右 clip 范围是 3e-4 和 4e-4；作为对比，GRPO 调参后用 0.2 和 0.27。第 6.7 节的代码演示了这个量级差别。

### 5.5 为什么对 MoE 更稳

论文在 48 层的 MoE 模型上观察到：每次梯度更新后，同一个 rollout 样本在新策略下约有 10% 的激活专家和旧策略不同（5.3 节）。一个 token 换了专家，它的概率可能剧烈变化，token 级比率随之远离 1，被 clip 或者产生很大的梯度权重。

GRPO 的应对是 Routing Replay：缓存旧策略激活的专家，算比率时在新策略里重放同样的路由。论文说没有它 GRPO 在 MoE 上不能正常收敛，代价是额外的显存和通信，也限制了模型的实际容量。

GSPO 只关心序列似然。单个 token 的剧变被 1/|y_i| 平均掉了：一个 token 的概率从 0.5 掉到 0.05，log 比率变化 log 10 ≈ 2.3，在 400 个 token 的回答里只让 log s_i 变化约 0.006。所以 GSPO 不需要 Routing Replay。

论文还提到一个工程上的好处（5.4 节）：训练引擎和推理引擎的数值精度不一致，通常要用训练引擎重算旧策略的似然。序列级似然对精度差异更宽容，GSPO 有可能直接用推理引擎返回的似然。

### 5.6 GSPO-token：需要 token 级优势时

多轮场景可能希望同一回答里不同 token 有不同优势。GSPO-token（式 13、14）把比率写成：

```text
s_{i,t}(θ) = sg[ s_i(θ) ] · π_θ(y_{i,t}|…) / sg[ π_θ(y_{i,t}|…) ]
```

sg 是 stop-gradient（PyTorch 的 detach）。第二个因子数值上恒为 1，所以 s_{i,t} 的值等于 s_i，梯度却流过每个 token 自己的 log 概率。所有 token 优势相同时，它和 GSPO 完全等价。多轮 Agent 训练里工具返回的 token 仍要 mask 掉，见 [04 章第 3 节](../04-agentic-rl/README.md#3-工具返回的-loss-mask)。

## 6 实现：只用 numpy 验证上面的结论

下面每段代码都能单独运行，结尾用 assert 检查结论。张量形状约定：N = B·G 是一个 batch 里的回答数（B 个问题，每题 G 个回答，同题的回答相邻存放），T 是补齐后的最大长度；logp 类张量是 (N, T)，每个 token 一个 log 概率；mask 是 (N, T)，模型生成的 token 为 1，prompt、padding、工具返回为 0。

这里用“对 log π 的解析梯度”代替自动求导：真实训练里 logp 由模型算出，梯度会继续反传到参数，但每个 token 的梯度系数就是 loss 对这个 token logp 的导数。验证系数就能验证 loss 写得对不对。

### 6.1 组内均值基线：期望有多偏

一个三选一的玩具问题，策略是 softmax(z)。穷举 G = 3 时所有 27 种分组，精确算出三种优势的期望梯度，和真梯度 ∇_z E[R] 比较。

```python
import itertools
import numpy as np

# 一个只有 3 个候选回答的“问题”：策略 π 是 softmax(z)，回答 a 的奖励是 R[a]
z = np.array([0.3, -0.2, 0.1])
R = np.array([1.0, 0.0, 0.4])
pi = np.exp(z) / np.exp(z).sum()

def score(a):
    """∇_z log π(a)，softmax 的对数导数：onehot(a) − π，形状 (3,)"""
    return np.eye(3)[a] - pi

true_grad = sum(pi[a] * R[a] * score(a) for a in range(3))   # ∇_z E[R]

G = 3
g_mean, g_loo, g_std = np.zeros(3), np.zeros(3), np.zeros(3)
for group in itertools.product(range(3), repeat=G):          # 穷举所有 3^G 种组，求精确期望
    p = np.prod(pi[list(group)])
    r = R[list(group)]
    for i, a in enumerate(group):
        loo = (r.sum() - r[i]) / (G - 1)                     # 留一均值：不含自己
        g_mean += p * (r[i] - r.mean()) * score(a) / G       # GRPO 的组内均值基线
        g_loo += p * (r[i] - loo) * score(a) / G             # RLOO 基线
        if r.std() > 0:
            g_std += p * (r[i] - r.mean()) / r.std() * score(a) / G

# 留一基线：精确无偏
assert np.allclose(g_loo, true_grad)
# 组内均值基线：期望是真梯度乘常数 (G−1)/G，方向不变
assert np.allclose(g_mean, (G - 1) / G * true_grad)
# 再除以 std：各分量的缩放不再相同，不是常数倍，方向变了
scale = g_std / true_grad
assert scale.max() - scale.min() > 0.1
print("std 版本各分量相对真梯度的倍数：", scale.round(3))
```

结果：留一基线精确无偏；组内均值基线差一个常数 (G−1)/G；再除以 std 后，三个分量的倍数分别约 1.88、1.73、2.30，方向偏了。

### 6.2 GRPO 的优势和 loss

完整的 GRPO loss：组内优势、token 比率、clip、k3 KL、response mask，以及三种聚合方式。手造一个 B = 2、G = 4 的 batch，其中一组全对。

```python
import numpy as np

def group_advantages(rewards, G, scale_std=True, eps=1e-6):
    """rewards: (N,)，N = B*G，同一问题的 G 个回答相邻存放。返回 (N,)。"""
    r = rewards.reshape(-1, G)                          # (B, G)
    adv = r - r.mean(axis=1, keepdims=True)
    if scale_std:
        adv = adv / (r.std(axis=1, keepdims=True) + eps)  # 总体标准差，除以 G
    return adv.reshape(-1)

def token_weights(mask, agg, max_tokens=None):
    """每个 token 在总 loss 里的权重，形状 (N, T)。"""
    N = mask.shape[0]
    if agg == "grpo":      # (1/N) Σ_i (1/|o_i|) Σ_t
        return mask / (N * mask.sum(axis=1, keepdims=True))
    if agg == "dapo":      # Σ_i Σ_t / Σ_i |o_i|
        return mask / mask.sum()
    if agg == "drgrpo":    # (1/N) Σ_i Σ_t / MAX_TOKENS
        return mask / (N * max_tokens)
    raise ValueError(agg)

def grpo_loss(logp, logp_old, logp_ref, adv, mask, eps_low=0.2, eps_high=0.2,
              beta=0.04, agg="grpo", max_tokens=None):
    """logp, logp_old, logp_ref, mask: (N, T)；adv: (N,)。
    返回 (要最小化的 loss, d loss / d logp，形状 (N, T))。"""
    # padding 位置的 log 概率可能是垃圾值甚至 −inf，先置 0，否则 inf × 0 = NaN
    logp, logp_old, logp_ref = (np.where(mask > 0, x, 0.0) for x in (logp, logp_old, logp_ref))
    A = adv[:, None]                                    # (N, 1)，广播到每个 token
    ratio = np.exp(logp - logp_old)                     # ρ_{i,t}，(N, T)
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high)
    surr = np.minimum(ratio * A, clipped * A)           # PPO clip 目标，(N, T)
    d_surr = np.where(ratio * A <= clipped * A, ratio * A, 0.0)  # 取到截断项时梯度为 0
    log_x = logp_ref - logp                             # log(π_ref / π_θ)
    kl = np.exp(log_x) - log_x - 1                      # k3 估计，(N, T)
    d_kl = 1 - np.exp(log_x)
    per_tok = -(surr - beta * kl)                       # 目标取负号变成 loss
    d_per_tok = -(d_surr - beta * d_kl)
    w = token_weights(mask, agg, max_tokens)
    return (w * per_tok).sum(), w * d_per_tok

# 手造 batch：B=2 个问题，每题 G=4 个回答，补齐长度 T=6
rng = np.random.default_rng(0)
G, T = 4, 6
rewards = np.array([1., 0., 0., 1.,     # 问题 1：两对两错
                    1., 1., 1., 1.])    # 问题 2：全对，std = 0
lengths = np.array([3, 6, 4, 5, 2, 6, 3, 4])
mask = (np.arange(T)[None, :] < lengths[:, None]).astype(float)   # (8, 6)
logp_old = np.log(rng.uniform(0.05, 0.9, size=(8, T)))
logp_ref = logp_old + rng.normal(0, 0.1, size=(8, T))
logp = logp_old + rng.normal(0, 0.3, size=(8, T))   # 模拟更新过几步后的新策略

adv = group_advantages(rewards, G)
assert np.all(np.isfinite(adv))                     # std = 0 的组没有 NaN
assert np.allclose(adv[4:], 0)                      # 全对的组优势全为 0
assert np.allclose(adv[:4], [1, -1, -1, 1])         # 两对两错：mean 0.5，std 0.5

loss, grad = grpo_loss(logp, logp_old, logp_ref, adv, mask)

# 1) 解析梯度和数值梯度一致（只在有效 token 上检查）
h = 1e-6
for i, t in [(0, 0), (1, 5), (2, 3), (3, 4), (5, 1)]:
    lp_p, lp_m = logp.copy(), logp.copy()
    lp_p[i, t] += h
    lp_m[i, t] -= h
    num = (grpo_loss(lp_p, logp_old, logp_ref, adv, mask)[0]
           - grpo_loss(lp_m, logp_old, logp_ref, adv, mask)[0]) / (2 * h)
    assert abs(num - grad[i, t]) < 1e-6, (i, t, num, grad[i, t])

# 2) padding 位置的 logp 怎么变都不影响 loss，梯度为 0
lp_pad = logp.copy()
lp_pad[mask == 0] += 5.0
assert np.isclose(grpo_loss(lp_pad, logp_old, logp_ref, adv, mask)[0], loss)
assert np.all(grad[mask == 0] == 0)

lp_old_bad = logp_old.copy()
lp_old_bad[mask == 0] = -np.inf                     # 例如推理引擎对 padding 返回 log 0
assert np.isfinite(grpo_loss(logp, lp_old_bad, logp_ref, adv, mask)[0])

# 3) 第一次更新前 π_θ = π_θold = π_ref：比率为 1，KL 为 0，
#    每个 token 的梯度系数就是 −(token 权重 × 优势)
loss0, grad0 = grpo_loss(logp_old, logp_old, logp_old, adv, mask)
w = token_weights(mask, "grpo")
assert np.allclose(grad0, -w * adv[:, None])
assert np.isclose(loss0, -(w * adv[:, None]).sum())
# GRPO 聚合下每条回答总权重都是 1/N，组内优势之和为 0，所以第一步的 loss 恰好是 0。
# loss 曲线贴着 0 是正常的，它不反映训练进展
assert abs(loss0) < 1e-12

# 4) 正优势且比率超过 1+ε 的 token 被 clip，梯度为 0
lp_big = logp_old.copy()
lp_big[0, 0] += np.log(1.5)                         # 回答 0 优势为 +1，ρ = 1.5 > 1.2
_, g = grpo_loss(lp_big, logp_old, logp_old, adv, mask, beta=0.0)
assert g[0, 0] == 0 and g[0, 1] != 0
print("loss =", round(loss, 4))
```

几个检查点：解析梯度和数值梯度一致；padding 位置怎么变都不影响 loss；第一次更新前 loss 恰为 0，梯度系数是 −权重 × 优势；比率超出上界的正优势 token 梯度为 0。

### 6.3 三种聚合方式：token 权重是否依赖长度

第一段在玩具问题上精确算出 1/|o| 带来的期望梯度。四种完整回答：对-短、对-长、错-短、错-长。

```python
import numpy as np

# 一个问题只有 4 种完整回答：(是否正确, token 数)
R = np.array([1., 1., 0., 0.])
L = np.array([100., 400., 100., 400.])        # 对-短、对-长、错-短、错-长
z = np.zeros(4)                               # 策略 logits，初始均匀
pi = np.exp(z) / np.exp(z).sum()
b = pi @ R                                    # 基线取期望奖励 0.5，看作常数

# 估计量 (R−b)·f(o)·∇log π(o) 的精确期望。softmax 下 ∇_z log π(o) = onehot(o) − π
def expected_grad(weight):
    return sum(pi[o] * (R[o] - b) * weight[o] * (np.eye(4)[o] - pi) for o in range(4))

g_grpo = expected_grad(1 / L)                 # GRPO：每条回答除以自身长度
g_const = expected_grad(np.full(4, 1 / 400))  # Dr. GRPO：除以常数

# 它等于 ∇_z E[(R−b)/|o|]，即 GRPO 实际在最大化 E[(R − b)/|o|]
f = (R - b) / L
assert np.allclose(g_grpo, pi * (f - pi @ f))

# 正确回答之间：短的被推得更高
assert g_grpo[0] > g_grpo[1]
# 错误回答之间：长的被压得更少，它在错误回答里所占的比例会上升
assert g_grpo[3] > g_grpo[2]
# 常数分母：同为正确或同为错误的回答，梯度相同，不看长度
assert np.isclose(g_const[0], g_const[1]) and np.isclose(g_const[2], g_const[3])
print("GRPO:", g_grpo.round(5), " 常数分母:", g_const.round(5))
```

第二段构造一长一短两条回答，比较三种聚合下每个 token 的权重，并演示 DAPO 分母跨 batch 变化、梯度累积时的写法。

```python
import numpy as np

def token_weights(mask, agg, max_tokens=None):
    """mask: (N, T)。返回每个 token 在总 loss 里的权重，(N, T)。"""
    N = mask.shape[0]
    if agg == "grpo":
        return mask / (N * mask.sum(axis=1, keepdims=True))
    if agg == "dapo":
        return mask / mask.sum()
    if agg == "drgrpo":
        return mask / (N * max_tokens)
    raise ValueError(agg)

T = 12
lengths = np.array([3, 12])                       # 一短一长两条回答
mask = (np.arange(T)[None, :] < lengths[:, None]).astype(float)

w = {a: token_weights(mask, a, max_tokens=16) for a in ("grpo", "dapo", "drgrpo")}
tok = lambda a, i: w[a][i, 0]                     # 第 i 条回答里任意一个有效 token 的权重

# GRPO 原版：token 权重 = 1/(N·|o_i|)，短回答的 token 是长回答的 12/3 = 4 倍
assert np.isclose(tok("grpo", 0) / tok("grpo", 1), 12 / 3)
# 但每条回答的总权重相同（都是 1/N），和长度无关
assert np.allclose(w["grpo"].sum(axis=1), 1 / 2)
# DAPO 和 Dr. GRPO：同一 batch 内每个 token 权重相同，不看它在哪条回答里
for a in ("dapo", "drgrpo"):
    assert np.isclose(tok(a, 0), tok(a, 1))
# 回答的总权重正比于长度
assert np.isclose(w["dapo"][1].sum() / w["dapo"][0].sum(), 12 / 3)

# 跨 batch：另一个 batch 里回答整体更长
mask2 = (np.arange(T)[None, :] < np.array([12, 12])[:, None]).astype(float)
w2_dapo = token_weights(mask2, "dapo")
w2_dr = token_weights(mask2, "drgrpo", max_tokens=16)
# DAPO 的分母 Σ|o_i| 随 batch 变：同样一个 token，在长 batch 里权重更小
assert w2_dapo[0, 0] < tok("dapo", 0)
# Dr. GRPO 的分母是常数：同样一个 token，在哪个 batch 里权重都一样
assert np.isclose(w2_dr[0, 0], tok("drgrpo", 0))

# 梯度累积的坑：把 batch 切成两个 micro-batch，各自做 token 平均再求平均，
# 结果等于“按 micro-batch 做 GRPO 式平均”，不是整批 token 平均
per_tok = np.random.default_rng(0).normal(size=(2, T)) * mask
global_mean = per_tok.sum() / mask.sum()
micro = np.mean([per_tok[k].sum() / mask[k].sum() for k in range(2)])
assert not np.isclose(global_mean, micro)
# 正确做法：每个 micro-batch 除以“全局” token 数，再相加
fixed = sum(per_tok[k].sum() / mask.sum() for k in range(2))
assert np.isclose(global_mean, fixed)
print({a: (round(tok(a, 0), 4), round(tok(a, 1), 4)) for a in w})
```

### 6.4 难度偏差：除以 std 放大了哪类题

```python
import numpy as np

def adv(r, scale_std, eps=1e-6):
    a = r - r.mean()
    return a / (r.std() + eps) if scale_std else a

G = 16
easy = np.array([1.] * 15 + [0.])        # 几乎全对：通过率 p = 15/16
half = np.array([1.] * 8 + [0.] * 8)     # 一半对：p = 1/2

# 用“组内优势绝对值之和”衡量一道题在更新里的总分量
mass = lambda r, s: np.abs(adv(r, s)).sum()

ratio_raw = mass(easy, False) / mass(half, False)   # 不除 std（Dr. GRPO）
ratio_std = mass(easy, True) / mass(half, True)     # 除以 std（GRPO）
assert ratio_std > ratio_raw                        # 除以 std 后，几乎全对的题相对权重变大
assert np.isclose(ratio_std / ratio_raw, half.std() / easy.std(), rtol=1e-4)  # 放大倍数 = std 之比

# 0/1 奖励、通过率 p 时的闭式：不除 std，总分量 = 2G·p(1−p)；除以 std，= 2G·sqrt(p(1−p))
for r in (easy, half):
    p = r.mean()
    assert np.isclose(mass(r, False), 2 * G * p * (1 - p))
    assert np.isclose(mass(r, True), 2 * G * np.sqrt(p * (1 - p)), rtol=1e-4)

# 那一条唯一的错误回答：除以 std 后优势从 −0.94 变成 −3.87
assert np.isclose(adv(easy, False)[-1], -15 / 16)
assert np.isclose(adv(easy, True)[-1], -np.sqrt(15), rtol=1e-4)

# 零优势组的比例：通过率 p 的题，G 个回答全对或全错的概率是 p^G + (1−p)^G
zero_frac = lambda p, G: p ** G + (1 - p) ** G
assert np.isclose(zero_frac(0.9, 8), 0.9 ** 8 + 0.1 ** 8)
assert zero_frac(0.9, 8) > 0.4 and zero_frac(0.9, 16) < 0.2   # G 越大，零优势组越少
# 除以 std 让优势对奖励尺度不敏感：0.001 的差异和 1 的差异得到同样的优势
tiny = np.array([1., 1., 1., 0.999])
assert np.allclose(adv(tiny, True), adv(np.array([1., 1., 1., 0.]), True), atol=1e-2)
assert np.abs(adv(tiny, False)).max() < 1e-3
print("easy/half 相对权重：不除 std %.3f，除 std %.3f" % (ratio_raw, ratio_std))
```

### 6.5 DAPO 的软超长惩罚

```python
import numpy as np

def soft_overlong(length, L_max, L_cache):
    """DAPO 式 13：长度惩罚，加到正确性奖励上。"""
    if length <= L_max - L_cache:
        return 0.0
    if length <= L_max:
        return ((L_max - L_cache) - length) / L_cache
    return -1.0

L_max, L_cache = 20480, 4096          # DAPO 4.1 节：生成上限 20,480，缓冲区 4,096
assert soft_overlong(16384, L_max, L_cache) == 0.0     # 期望最大长度以内不罚
assert soft_overlong(16384 + 2048, L_max, L_cache) == -0.5
assert soft_overlong(L_max, L_max, L_cache) == -1.0
assert soft_overlong(L_max + 1, L_max, L_cache) == -1.0
xs = np.arange(16000, 20600, 37)
ys = np.array([soft_overlong(x, L_max, L_cache) for x in xs])
assert np.all(np.diff(ys) <= 0)                        # 越长罚得越多，单调不增
```

### 6.6 k3 估计量：无偏、非负，梯度对应反向 KL

在一个 50 个 token 的词表上，用 20 万次采样估计 KL(π_θ ‖ π_ref)，并检查 k3 的期望梯度。

```python
import numpy as np

rng = np.random.default_rng(0)
V = 50                                                  # 词表大小
logits_theta = rng.normal(size=V)
logits_ref = logits_theta + rng.normal(0, 0.5, size=V)  # 参考模型和当前策略有一定差距
p_theta = np.exp(logits_theta) / np.exp(logits_theta).sum()
p_ref = np.exp(logits_ref) / np.exp(logits_ref).sum()

true_kl = np.sum(p_theta * np.log(p_theta / p_ref))     # KL(π_θ ‖ π_ref)，精确值

x = rng.choice(V, size=200_000, p=p_theta)              # 从 π_θ 采样 token
r = p_ref[x] / p_theta[x]                               # r = π_ref / π_θ
k1 = -np.log(r)
k2 = 0.5 * np.log(r) ** 2
k3 = r - 1 - np.log(r)

assert np.all(k3 >= 0)                                  # 每个样本都非负
assert np.any(k1 < 0)                                   # k1 单个样本可以为负
assert abs(k3.mean() - true_kl) < 0.01 * true_kl + 1e-3 # 期望 ≈ 真实 KL
assert abs(k1.mean() - true_kl) < 0.02 * true_kl + 1e-3
assert k3.std() < k1.std()                              # 这个例子里 k3 方差更小

# 无偏的关键一步：E_{π_θ}[r] = Σ π_θ · π_ref/π_θ = Σ π_ref = 1，所以 E[r − 1] = 0
assert np.isclose(np.sum(p_theta * (p_ref / p_theta)), 1.0)

# k3 对 log π_θ 的导数是 1 − π_ref/π_θ。目标里是 −β·k3，
# 所以梯度系数是 β(π_ref/π_θ − 1)，和 DeepSeekMath 附录式 21 的 KL 部分一致
lp, lr, h = np.log(0.3), np.log(0.2), 1e-6
k3_of = lambda lp: np.exp(lr - lp) - (lr - lp) - 1
num = (k3_of(lp + h) - k3_of(lp - h)) / (2 * h)
assert np.isclose(num, 1 - np.exp(lr - lp))
# 作为 loss 求导时：把采样到的 token 当成固定数据，对 π_θ 的 logits z 求 ∇k3，
# 期望等于“反向” KL(π_ref ‖ π_θ) 的梯度 p_θ − p_ref，而不是 KL(π_θ ‖ π_ref) 的梯度
onehot_minus_p = np.eye(V) - p_theta                    # 第 x 行是 ∇_z log π_θ(x)，(V, V)
exp_grad_k3 = (p_theta * (1 - p_ref / p_theta)) @ onehot_minus_p
assert np.allclose(exp_grad_k3, p_theta - p_ref)
def fwd_kl(z):
    p = np.exp(z) / np.exp(z).sum()
    return np.sum(p * np.log(p / p_ref))
num_fwd = np.array([(fwd_kl(logits_theta + h * e) - fwd_kl(logits_theta - h * e)) / (2 * h)
                    for e in np.eye(V)])
assert not np.allclose(num_fwd, exp_grad_k3, atol=1e-3)
def rev_kl(z):
    p = np.exp(z) / np.exp(z).sum()
    return np.sum(p_ref * np.log(p_ref / p))
num_rev = np.array([(rev_kl(logits_theta + h * e) - rev_kl(logits_theta - h * e)) / (2 * h)
                    for e in np.eye(V)])
assert np.allclose(num_rev, exp_grad_k3, atol=1e-6)
print("真实 KL %.4f，k1 %.4f±%.3f，k2 %.4f，k3 %.4f±%.3f"
      % (true_kl, k1.mean(), k1.std(), k2.mean(), k3.mean(), k3.std()))
```

这个例子里真实 KL 约 0.146。k1 和 k3 的样本均值都约 0.145；k1 的样本标准差约 0.54，k3 约 0.14；k2 的均值约 0.156，有偏。

### 6.7 GSPO 的序列级比率

一条 400 个 token 的回答，每个 token 的 log 比率有标准差 0.02 的噪声；然后让其中一个 token 的概率从 0.5 掉到 0.05，模拟 MoE 换了专家。

```python
import numpy as np

rng = np.random.default_rng(1)
L = 400                                                  # 回答长度
logp_old = np.log(rng.uniform(0.2, 0.95, size=L))        # 旧策略下每个 token 的 log 概率
logp_new = logp_old + rng.normal(0, 0.02, size=L)        # 更新一步后，每个 token 小幅变化

def seq_ratio(logp_new, logp_old):
    """GSPO 的 s_i = exp(mean_t(log π_θ − log π_θold))，标量。"""
    return np.exp(np.mean(logp_new - logp_old))

tok_ratio = np.exp(logp_new - logp_old)                  # GRPO 的 ρ_{i,t}，(L,)
s = seq_ratio(logp_new, logp_old)

# 1) 序列级比率 = token 比率的几何平均 = 整句似然比开 1/|y| 次方
assert np.isclose(s, np.prod(tok_ratio) ** (1 / L))
assert np.isclose(s, np.exp(np.log(tok_ratio).mean()))

# 2) 不开方的整句似然比随长度指数放大：同样的 token 噪声，整句比率可以远离 1
full = np.exp(np.sum(logp_new - logp_old))
assert abs(np.log(full)) > 10 * abs(np.log(s))

# 3) 一个 token 的概率剧变（例如 MoE 换了专家）：旧概率 0.5，新概率 0.05
spike_new = logp_new.copy()
spike_old = logp_old.copy()
spike_old[100], spike_new[100] = np.log(0.5), np.log(0.05)
tok_spike = np.exp(spike_new - spike_old)
s_spike = seq_ratio(spike_new, spike_old)
assert tok_spike[100] < 0.11                             # token 级比率掉到 0.1，远超 clip 范围 [0.8, 1.2]
assert abs(s_spike - s) < 0.01                           # 序列级比率只变化约 log(10)/400 ≈ 0.006

# 4) 序列比率离 1 的典型距离：远小于 token 比率，所以 clip 范围要小几个数量级
dev_tok = np.median(np.abs(tok_ratio - 1))
dev_seq = abs(s - 1)
assert dev_seq < dev_tok / 10
# 用 GRPO 的 ε=0.2 去 clip 序列比率，几乎永远不会触发
assert 0.8 < s_spike < 1.2

# 5) 梯度权重：GSPO 里同一回答每个 token 的权重都是 s_i；GRPO 里是各自的 ρ_{i,t}
w_gspo = np.full(L, s_spike)
w_grpo = tok_spike
assert np.ptp(w_gspo) == 0 and np.ptp(w_grpo) > 0.8
print("token 比率中位偏离 %.4f，序列比率偏离 %.5f，突变后 token %.3f / 序列 %.5f"
      % (dev_tok, dev_seq, tok_spike[100], s_spike))
```

这个例子里 token 比率离 1 的中位距离约 0.015，序列比率离 1 约 0.001，差一个数量级多。真实回答有几千个 token，差距更大，这就是 clip 范围要用 1e-4 量级的原因。

## 7 数值例子：一组 G = 4 的完整手算

设 G = 4，奖励 r = [1, 1, 1, 0]（三对一错），四个回答长度 |o| = [100, 200, 300, 400]，错误回答最长。Dr. GRPO 的常数取 MAX_TOKENS = 2000。取 μ = 1、β = 0，所以 ρ = 1，clip 不起作用。

**第一步：优势**。

- mean(r) = 0.75。
- 总体标准差 std(r) = √((3 × 0.25² + 0.75²) / 4) = √0.1875 ≈ 0.4330。
- 除以 std 的优势（GRPO、DAPO）：正确回答 0.25 / 0.4330 = 1/√3 ≈ 0.5774；错误回答 −0.75 / 0.4330 = −√3 ≈ −1.7321。
- 不除 std 的优势（Dr. GRPO）：正确 0.25，错误 −0.75。

**第二步：每个 token 的梯度系数**（目标对这个 token 的 log π 求导，单位 10⁻³）。

| 回答 | 长度 | GRPO：Â/(G·\|o\|) | DAPO：Â/Σ\|o\| | Dr. GRPO：Ã/(G·MAX_TOKENS) |
|---|---|---|---|---|
| 1 对 | 100 | 0.5774/400 = 1.4434 | 0.5774/1000 = 0.5774 | 0.25/8000 = 0.0313 |
| 2 对 | 200 | 0.5774/800 = 0.7217 | 0.5774 | 0.0313 |
| 3 对 | 300 | 0.5774/1200 = 0.4811 | 0.5774 | 0.0313 |
| 4 错 | 400 | −1.7321/1600 = −1.0825 | −1.7321/1000 = −1.7321 | −0.75/8000 = −0.0938 |

**第三步：读表**。

- GRPO 里三条正确回答的 token 系数差了 3 倍，只因为长度不同。
- 错误回答每个 token 的惩罚和最短正确回答每个 token 的奖励之比：GRPO 是 1.0825 / 1.4434 = 0.75；DAPO 和 Dr. GRPO 都是 3，只由优势之比决定。长度把错误回答的惩罚压低了 4 倍。
- 如果错误回答再写长一倍（400 → 800）：GRPO 里它每个 token 的惩罚减半，变成 −0.5413；DAPO 的分母变成 1400，连正确回答的 token 系数也降到 0.4124；Dr. GRPO 里所有系数都不变。

下面的代码把这张表算了一遍。

```python
import numpy as np

r = np.array([1., 1., 1., 0.])              # G = 4：三对一错
L = np.array([100, 200, 300, 400])          # 各回答 token 数，错误回答最长
G, L_MAX = 4, 2000

mean, std = r.mean(), r.std()               # 0.75，sqrt(0.1875) ≈ 0.4330（总体标准差）
A_std = (r - mean) / std                    # GRPO / DAPO 的优势
A_raw = r - mean                            # Dr. GRPO 的优势
assert np.allclose(A_std, [1/np.sqrt(3)] * 3 + [-np.sqrt(3)])
assert np.allclose(A_raw, [0.25] * 3 + [-0.75])

# μ = 1、β = 0 时每个 token 的 ∇log π 系数（目标对 log π 的导数）
w_grpo = A_std / (G * L)
w_dapo = A_std / L.sum()
w_dr = A_raw / (G * L_MAX)
assert np.allclose(w_grpo, [1.4434e-3, 7.217e-4, 4.811e-4, -1.0825e-3], rtol=1e-3)
assert np.allclose(w_dapo, [5.774e-4] * 3 + [-1.7321e-3], rtol=1e-3)
assert np.allclose(w_dr, [3.125e-5] * 3 + [-9.375e-5])

# 错误回答每个 token 的惩罚 / 最短正确回答每个 token 的奖励
assert np.isclose(w_grpo[3] / w_grpo[0], -0.75)      # GRPO：长度把惩罚压到 0.75 倍
assert np.isclose(w_dapo[3] / w_dapo[0], -3.0)       # 只剩优势之比
assert np.isclose(w_dr[3] / w_dr[0], -3.0)

# 错误回答再写长一倍（400 → 800），其他不变
L2 = np.array([100, 200, 300, 800])
assert np.isclose((A_std / (G * L2))[3], w_grpo[3] / 2)     # GRPO：它的每个 token 罚得更轻
assert np.isclose((A_std / L2.sum())[0], A_std[0] / 1400)   # DAPO：连正确回答的权重也变小
assert np.isclose((A_raw / (G * L_MAX))[3], w_dr[3])        # Dr. GRPO：不变
print(np.round(w_grpo * 1e3, 4), np.round(w_dapo * 1e3, 4), np.round(w_dr * 1e3, 4))
```

## 8 工程细节与常见坑

每条按“现象 → 原因 → 排查和处理”写。监控指标建议至少有：训练奖励、验证集准确率、正确和错误回答分开的平均长度、截断率、策略熵、零优势组比例、clip 比例（上界和下界分开）、log 比率的分位数、KL。

**1. 回答长度持续上涨，奖励却不涨了**

- 现象：训练奖励进入平台期后，平均长度仍在稳定上涨；分开看，主要是错误回答在变长。
- 原因：3.1 节的长度偏差。答错时写长，每个 token 罚得更轻。
- 排查：把长度曲线按正确和错误拆开；找到 loss 聚合代码，看是不是 `masked_mean(..., dim=-1).mean()` 这种按每条回答求平均的写法；同时看截断率，长度涨到上限附近时截断样本会干扰判断（见第 4 条）。
- 处理：换成 DAPO 的 token 级聚合或 Dr. GRPO 的常数分母。长度增长本身不一定是坏事，要对照验证集准确率判断（DAPO 4.3 节也提醒长度和准确率要一起看）。

**2. 熵快速下降，同组回答几乎一样**

- 现象：策略熵在训练早期就快速下降；同一题的 G 个回答越来越像；零优势组比例随之上升。
- 原因：4.1 节。上界 clip 按比例限制概率上涨，低概率 token 很难被提上来，高概率 token 不受限。
- 排查：分别统计被上界和被下界截断的 token 比例，以及被上界截断的 token 的平均概率。DAPO 观察到后者低于 0.2。
- 处理：Clip-Higher，只调大 ε_high（DAPO 用 0.28）。反方向也要防：熵暴涨通常伴随乱码和重复，这时检查 ε_high 是否过大、学习率是否过高。

**3. 零优势组比例越来越高，有效 batch 变小**

- 现象：一个 batch 里组内 std 为 0 的题占比持续上升，训练奖励上升变慢，梯度范数变小。
- 原因：模型做对的题越来越多，通过率接近 1 的题全对概率 p^G 很大（4.2 节）。token 级聚合下，这些组的 token 还占着分母，稀释了其他组的梯度。
- 排查：直接记录零优势组比例和“有梯度的 token 数”。
- 处理：Dynamic Sampling 过滤全对和全错的题；或者按通过率筛题、做课程；增大 G 也能降低零优势组比例，代价是 rollout 成本。过滤后也要把这些组从 token 计数的分母里去掉。

**4. 截断样本带来奖励噪声**

- 现象：截断率上升后训练变得不稳，奖励曲线抖动；或者模型开始故意写短，牺牲正确率。
- 原因：截断回答没有最终答案，被判错，但推理可能是对的。
- 排查：统计截断率，以及截断样本里推理过程看起来正确的比例（抽样人工看即可）。
- 处理：先做 Overlong Filtering，把截断样本的 loss mask 掉；再加 4.4 节的软超长惩罚。在 Agent 场景里，工具超时导致的截断应该丢弃或重跑，不按长度惩罚，见 [04 章](../04-agentic-rl/README.md#2-多轮-rollout-与环境交互)。

**5. loss 突然变成 NaN**

- 现象：某一步之后 loss 和梯度全是 NaN，参数被污染。
- 原因有三类。一是 std 为 0 的组没加 eps，0/0。二是 padding 位置的 log 概率是 −inf（例如推理引擎对 padding 返回 log 0），先算 exp 或差值再乘 mask，inf × 0 = NaN。三是 eps 太小、奖励是连续值时，std 极小的组被除出极大的优势，梯度爆炸。
- 排查：在 loss 前对优势、log 比率做 `isfinite` 检查，定位到具体样本。
- 处理：std 加 eps，或者直接把 std 为 0 的组的优势置 0；先用 mask 把无效位置的 log 概率置 0 再计算（6.2 节的 `np.where` 就是这么做的）；连续奖励时考虑 Dr. GRPO 的不除 std，或在 batch 级归一化。

**6. 训练引擎和推理引擎的 log 概率对不上**

- 现象：每个 rollout 步的第一个 mini-batch 理论上是 on-policy 的，比率应该是 1，实际却明显偏离 1，clip 比例在第一步就不为 0。
- 原因：rollout 用推理引擎（如 vLLM、SGLang），训练用训练引擎（如 Megatron、FSDP），两边的 kernel、精度、并行切分不同，同一个 token 算出的 log 概率不同。GSPO 论文 5.4 节说，实践中通常用训练引擎重算旧策略的似然。
- 排查：取一批 rollout，在训练引擎里用同一份参数重算 log 概率，和推理引擎返回的值逐 token 比较，看 log 比率的分布和最大值。
- 处理：π_θold 的 log 概率一律用训练引擎重算，不直接用推理引擎返回的值。序列级比率对这种误差更宽容（5.5 节）。训练和推理引擎的协同见 [07 章第 5 节](../07-training-systems/README.md#5-rl-框架里训练与推理引擎的协同)。

**7. MoE 模型上 token 比率剧烈波动，训练崩溃**

- 现象：log 比率的尾部分位数（如 99.9 分位）很大，clip 比例高，训练一段时间后奖励断崖式下降，回退检查点也救不回来。
- 原因：5.5 节。每次更新后约 10% 的激活专家会变，单个 token 的概率随之剧变。
- 排查：按层统计新旧策略的专家选择差异；看 log 比率的极值是否集中在少数 token。
- 处理：Routing Replay，或改用 GSPO 的序列级比率。

**8. KL 项算在了不该算的 token 上**

- 现象：KL 数值随 padding 长度或工具返回长度变化；或者 π_θ 和 π_ref 相同时 KL 不为 0。
- 原因：KL 项漏乘 response mask，prompt、padding 或工具返回的 token 也被算了进去。另一类错误是策略项和 KL 项用了不同的聚合方式，两者的相对权重随长度变化。
- 排查：令 π_θ = π_ref，KL 必须恰为 0；把 padding 位置的 log 概率随便改，loss 必须不变（6.2 节检查了这两点）。
- 处理：KL 和策略项用同一个 mask、同一种聚合。另外记得 2.4 节的结论：k3 作为 loss 时期望梯度对应 KL(π_ref ‖ π_θ)，作为监控指标时它的值估计的是 KL(π_θ ‖ π_ref)。

**9. 梯度累积改变了聚合方式**

- 现象：换一个 micro-batch 大小，训练曲线就不一样。
- 原因：每个 micro-batch 各自除以自己的 token 数再求平均，结果是“按 micro-batch 做样本级平均”，不是整个 batch 的 token 级平均。micro-batch 越小，越接近 GRPO 原版的长度偏差。
- 排查：固定数据，比较一次性算整个 batch 的 loss 和梯度累积后的结果。
- 处理：先算出整个 batch 的有效 token 数，每个 micro-batch 的 loss 都除以这个全局数再累加（6.3 节第二段代码验证了这两种写法的差别）。Dr. GRPO 的常数分母天然没有这个问题。

**10. 拿 loss 曲线判断训练进展**

- 现象：loss 一直在 0 附近抖动，看不出趋势。
- 原因：组内优势之和为 0。μ = 1 时 ρ = 1，GRPO 聚合下 loss 恰好等于 0（6.2 节验证了）；其他聚合下也只是很小的数。loss 的值不反映策略变好没有。
- 处理：看训练奖励、验证集准确率、熵、长度和 KL，不看 loss。

**11. 把 GSPO 的 clip 范围和 GRPO 的混用**

- 现象：GSPO 用 ε = 0.2 时 clip 比例几乎为 0，等于没有信任域；GRPO 误用 3e-4 时几乎所有 token 都被截断，不学习。
- 原因：两种比率的尺度差几个数量级（5.4 节）。
- 处理：换算法时重新调 clip 范围，并以 clip 比例作为调参依据。

## 9 对比与取舍

### 9.1 五种方法的差别

| | PPO | GRPO | Dr. GRPO | DAPO | GSPO |
|---|---|---|---|---|---|
| 基线 | critic 估计 V(s_t)，GAE | 组内均值 | 组内均值 | 组内均值 | 组内均值 |
| 优势是否除以组内 std | 不涉及（实现里常做 batch 级归一化） | 是 | 否 | 是 | 是 |
| 比率粒度 | token | token | token | token | 序列（几何平均） |
| 聚合 | 论文式子按 token；很多开源实现按回答长度平均 | 回答内平均，再组内平均 | 回答内求和，除以常数 | 组内所有 token 求和，除以总 token 数 | 回答内平均（式 10 里的 1/\|y\|） |
| clip | 对称，常用 0.2 | 对称 ε | 对称，论文用 0.2 | 解耦：0.2 / 0.28 | 序列级：3e-4 / 4e-4 |
| KL | 加进每个 token 的奖励 | 加进 loss，k3，β = 0.04 | 去掉（β = 0） | 去掉 | 论文公式省略 KL |
| 采样 | 每题通常 1 个 | 每题 G 个 | 每题 G 个 | G 个，过采样并过滤零优势组 | 每题 G 个 |
| 要加载的模型 | 策略、参考、奖励、critic | 策略、参考、奖励 | 策略、奖励（规则奖励时只有策略） | 同 Dr. GRPO | 视是否加 KL |
| 解决的问题 | 稳定的策略更新 | 去掉 critic | 长度偏差、难度偏差 | 熵塌缩、零优势组、长回答 token 权重小、截断噪声 | token 比率噪声、MoE 不稳 |

### 9.2 什么场景选哪个

- **奖励稠密，或需要逐 token 的信用分配**，且显存够放 critic：PPO。critic 能给每个位置一个基线，G 个样本的组内基线做不到。
- **结果奖励、可验证任务、显存紧**：从 GRPO 起步，但聚合方式要先改掉。按回答长度求平均的写法几乎没有理由保留。
- **在意回答长度，或要分析长度变化的原因**：Dr. GRPO。它的期望梯度就是策略梯度乘常数，长度变化可以归因到奖励本身。代价是梯度尺度依赖奖励尺度和 MAX_TOKENS，换奖励设计或长度上限要重调学习率。
- **长 CoT、生成长度上万 token、规则奖励**：DAPO 的整套改动。四项可以单独拆用：Clip-Higher 和 Overlong Filtering 改动小，几乎总值得加；Dynamic Sampling 会增加 rollout 量，在生成很贵的异步系统里要重新评估代价。
- **MoE 模型、回答很长、或训练推理引擎分离**：GSPO。它不需要 Routing Replay，对精度差异更宽容。需要 token 级优势时用 GSPO-token。

这些改动大多互相正交。Dr. GRPO 改的是优势和聚合的分母，DAPO 改的是 clip、采样和聚合，GSPO 改的是比率的单位。实际系统里常见的做法是组合使用，例如序列级比率配合不除 std 的优势，或者 token 级聚合配合 Clip-Higher。组合后要重新检查第 8 节的那些指标，因为这些组合没有在原论文里一起验证过。

### 9.3 面试时怎么串起来讲

按“问题 → 修法”讲，比按论文罗列更清楚：

1. GRPO 用组内均值替代 critic。它是合法基线，因为等价于留一均值乘常数 (G−1)/G。代价是采样成本和零优势组。
2. 归一化有偏。1/|o_i| 让 token 权重依赖长度，std 让难度极端的题权重变大。Dr. GRPO 两处都去掉；DAPO 改成 token 级聚合但保留 std。
3. 采样效率：DAPO 的 Dynamic Sampling 过滤零优势组。
4. 探索：DAPO 的 Clip-Higher 防止熵塌缩。
5. 稳定性：GSPO 把比率和 clip 放到序列级，和序列级奖励的单位对齐，对 MoE 尤其重要。

最后接一句 Agent 场景：回答里包含工具返回的内容，这部分不是模型生成的，计算 loss 时要屏蔽，见 [04 Agentic RL](../04-agentic-rl/README.md)。

## 10 自测清单

- [ ] 从 ∇E[R] 出发，写出带基线的策略梯度，说明基线为什么不改变期望
- [ ] 证明 r_i − mean(r) = ((G−1)/G)(r_i − b_{−i})，说出组内均值基线无偏的三个条件
- [ ] 写出 DeepSeekMath 式 3，解释 G、|o_i|、ρ、Â、ε、β、π_ref 各是什么
- [ ] 推导 k3 = x − 1 − log x 的无偏性和非负性，说出它作为 loss 时梯度对应哪个方向的 KL
- [ ] 写出 GRPO 下一个 token 的梯度系数 Â_i/(G·|o_i|)，分正确和错误回答说明长度偏差的方向
- [ ] 对 0/1 奖励写出除以 std 后一道题的总权重 2G·√(p(1−p))，说明难度偏差
- [ ] 说出 DAPO 和 Dr. GRPO 的分母在一个 batch 内和跨 batch 的区别
- [ ] 写出 GSPO 的序列级比率，解释为什么开 1/|y| 次方、为什么 clip 范围是 1e-4 量级
- [ ] 不看代码，手算第 7 节的表
- [ ] 手写 GRPO 的组内优势、clip、k3 KL 和 token 级聚合（见 [coding/questions.md](../../../coding/questions.md) 的 code-model-002）
- [ ] 说出至少 6 个训练中的现象，以及各自怎么排查

## 资料

- 论文精读：[DeepSeekMath / GRPO](../../../papers/02-deepseekmath-grpo.md)、[DeepSeek-R1](../../../papers/03-deepseek-r1.md)、[DAPO](../../../papers/04-dapo.md)、[Dr. GRPO](../../../papers/05-dr-grpo.md)、[GSPO](../../../papers/06-gspo.md)
- Shao 等，[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)，arXiv 2402.03300，2024-02。GRPO 目标见 4.1 节式 3、式 4，梯度系数见附录 A.1.6 式 20、21，超参数见 4.2 节
- Liu 等，[Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)，arXiv 2503.20783，2025-03。两个偏差见 3.1 节和图 4，代码清单 1，基线无偏性和与 RLOO 的等价见附录 A
- Yu 等，[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)，arXiv 2503.14476，2025-03。式 8、式 12、式 13，超参数见 4.1 节
- Zheng 等，[Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)，arXiv 2507.18071，2025-07。式 5 到式 14，clip 范围见 5.1 节，MoE 专家变化见 5.3 节，训推精度见 5.4 节
- Ahmadian 等，[Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)，arXiv 2402.14740，2024-02。RLOO
- Schulman，[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)，2020-03。k1、k2、k3 估计量
- Schulman 等，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，arXiv 1707.06347，2017-07
