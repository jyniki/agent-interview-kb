# 强化学习入门：从策略梯度到 PPO

> 层级：通用层 · 预计 200 分钟 · 题目：[questions.md](questions.md)
> 前置：[11 后训练入门](../11-post-training-basics/README.md)（RLHF 流程、KL 约束的奖励最大化）
> 后续：[training/03 GRPO 及其变体](../../tracks/training/03-grpo-family/README.md)（组内基线、长度偏差、序列级比率）

## 1 这一章解决什么问题

GRPO、DAPO、Agentic RL 这些高频题，底层都是策略梯度和 PPO。面试官从 GRPO 往下追，常落到这几个问题：策略梯度怎么推出来，为什么能减基线，优势函数是什么，GAE 的 λ 在权衡什么，PPO 的 clip 到底限制了什么，KL 和 clip 有什么不同，LLM 里的“状态”和“动作”是什么。

监督学习有现成的标签，loss 对参数可导。RL 只有奖励：模型先采样一个动作，环境再给一个分数。采样是离散的，分数也不对参数可导。这一章从定义出发，一步步推到 PPO，并说明每一步解决了上一步的什么缺陷：

1. REINFORCE 用对数导数技巧绕开不可导，但方差大；
2. 基线和优势函数降方差，不改变期望；
3. Actor-Critic 和 GAE 用价值函数进一步降方差，代价是引入偏差；
4. PPO 用重要性采样复用数据，用 clip 防止一次更新走太远；
5. LLM 场景的特殊性让 critic 难训，引出 GRPO 系方法。第 5 步的细节留给 [training/03](../../tracks/training/03-grpo-family/README.md)。

学完你应该能：

- 把 LLM 生成写成一个 MDP，说清状态、动作、奖励，以及 token 级和序列级两种建模；
- 从 ∇J 出发推出策略梯度定理，说明环境转移为什么不用可导；
- 证明减去只依赖状态的基线不改变期望，推出方差最优的常数基线；
- 推出 GAE 是 k 步优势估计的指数加权平均，说出 λ = 0 和 λ = 1 两个极限；正确处理终止和截断；
- 写出 PPO 的 clip 目标，证明它是未截断目标的下界，按优势的正负说清哪些梯度被截断；
- 区分 clip、对旧策略的 KL 惩罚、对参考模型的 KL；
- 用 numpy 在一个小 MDP 上跑通 PPO，并知道训练时看哪些指标。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| s_t, a_t, r_t | 第 t 步的状态、动作、奖励 |
| τ | 一条轨迹 (s_0, a_0, r_0, s_1, a_1, r_1, …) |
| π_θ(a\|s) | 参数为 θ 的策略，给出状态 s 下选动作 a 的概率 |
| γ | 折扣因子，0 ≤ γ ≤ 1 |
| J(θ) | 目标：期望（折扣）回报 |
| G_t | 从 t 开始的回报 Σ_{k≥t} γ^{k−t} r_k（reward-to-go） |
| V^π(s), Q^π(s, a) | 状态价值、动作价值：从 s（或 s 和 a）出发、之后按 π 行动的期望回报 |
| A^π(s, a) | 优势 Q^π(s, a) − V^π(s) |
| b(s) | 基线，只依赖状态 |
| δ_t | TD 残差 r_t + γV(s_{t+1}) − V(s_t) |
| λ | GAE 参数，0 ≤ λ ≤ 1 |
| ρ_t | 重要性比率 π_θ(a_t\|s_t) / π_θold(a_t\|s_t)；PPO 论文记作 r_t(θ)，本章改用 ρ 以免和奖励 r_t 混淆 |
| ε | PPO 的 clip 范围 |
| π_ref | 参考模型，RLHF 里通常是 SFT 模型 |

## 2 MDP：把生成写成决策过程

**MDP**（马尔可夫决策过程）由状态 s、动作 a、转移概率 P(s'|s, a)、奖励 r 和折扣 γ 组成。“马尔可夫”指下一状态只依赖当前状态和动作，不依赖更早的历史。策略 π_θ(a|s) 给出每个状态下选动作的概率。目标是最大化期望回报：

```text
J(θ) = E_{τ~π_θ} [ Σ_t γ^t r_t ]
τ 的概率：P(τ; θ) = p(s_0) · Π_t π_θ(a_t | s_t) · P(s_{t+1} | s_t, a_t)
```

P(τ; θ) 里只有 π_θ 那一项依赖 θ。初始分布 p(s_0) 和转移 P 属于环境，我们不知道它们的形式，也不需要对它们求导。这一点是第 3 节推导的关键。

LLM 生成对应到 MDP 上：

| MDP 概念 | 单轮生成 | Agent 多轮 |
|---|---|---|
| 状态 s_t | prompt + 已生成的 token | 对话历史 + 已有的工具结果 + 已生成 token |
| 动作 a_t | 下一个 token | 下一个 token（工具调用也是 token 序列） |
| 转移 | 确定的：把 token 拼到末尾 | 生成部分确定；工具结果由环境决定，有随机性 |
| 奖励 | 通常只在序列末尾给一次 | 可能在末尾给，也可能每个工具步给中间奖励 |
| 折扣 γ | 常取 1 | 常取 1 或接近 1 |
| 回合结束 | 生成 EOS，或达到长度上限 | 任务完成、步数上限、工具出错 |

状态包含了全部前缀，所以 LLM 生成天然满足马尔可夫性。代价是状态空间巨大，任何两个状态几乎都不相同，只能靠神经网络泛化。

**token 级还是序列级？** 也可以把整个回答看成一个动作，问题退化成上下文赌博机（contextual bandit）：看到 prompt x，选一个回答 y，拿到奖励 R(x, y)。[Ahmadian 等](https://arxiv.org/abs/2402.14740)（2024-02）认为奖励只在结尾出现时，把整段生成当成一个动作更合适。

两种建模在 γ = 1、奖励只在末尾时给出同一个策略梯度（推导）。因为 log π(y|x) = Σ_t log π(y_t | x, y_<t)，序列级的 ∇log π(y|x)·R 拆开就是每个 token 的 ∇log π(y_t|…)·R 之和，而 token 级里每个位置的 reward-to-go 都等于 R。两者的区别在基线：token 级可以给每个前缀一个不同的基线 V(s_t)，需要价值模型；序列级只需要一个 prompt 级的基线，可以用同一 prompt 多次采样的平均奖励代替。第 7 节再谈这个选择。

## 3 策略梯度与基线

### 3.1 对数导数技巧：梯度怎么穿过采样

**问题**：J(θ) 是对 π_θ 采样后再取期望，采样过程离散，不能直接反向传播。

**做法**：把期望写成求和，求导后乘一个 π/π，就能把梯度重新写成期望。对任意函数 f：

```text
∇_θ E_{x~p_θ}[f(x)] = Σ_x ∇_θ p_θ(x) · f(x)
                    = Σ_x p_θ(x) · ∇_θ log p_θ(x) · f(x)        因为 ∇p = p · ∇log p
                    = E_{x~p_θ}[ f(x) · ∇_θ log p_θ(x) ]
```

∇_θ log p_θ(x) 叫得分函数（score function）。这一步只要求 p_θ 对 θ 可导，不要求 f 可导。所以奖励可以是规则、单元测试、人类打分，什么都行。

把 x 换成轨迹 τ，f 换成回报：

```text
log P(τ; θ) = log p(s_0) + Σ_t log π_θ(a_t|s_t) + Σ_t log P(s_{t+1}|s_t, a_t)
∇_θ log P(τ; θ) = Σ_t ∇_θ log π_θ(a_t|s_t)                     环境项与 θ 无关，导数为 0

∇_θ J = E_τ [ (Σ_t γ^t r_t) · Σ_t ∇_θ log π_θ(a_t|s_t) ]
```

环境转移在求导时消失了。这是“无模型”（model-free）RL 的来源：不需要知道环境怎么转移，只需要能从中采样。

### 3.2 因果性：动作只为它之后的奖励负责

上式里，第 t 步的得分函数乘的是整条轨迹的回报，包括 t 之前的奖励。这部分期望为 0。对 k < t：

```text
E[ r_k · ∇log π_θ(a_t|s_t) ] = E_{s_t, r_k} [ r_k · E_{a_t~π(·|s_t)}[ ∇log π_θ(a_t|s_t) ] ]
                             = E[ r_k · Σ_a ∇π_θ(a|s_t) ] = E[ r_k · ∇ Σ_a π_θ(a|s_t) ] = E[ r_k · ∇1 ] = 0
```

第一步成立，是因为 r_k 在 a_t 采样之前就确定了，可以条件在 s_t 上把它提出去。去掉这些项，得到策略梯度定理的常用形式：

```text
∇_θ J = E_τ [ Σ_t γ^t · G_t · ∇_θ log π_θ(a_t|s_t) ]          G_t = Σ_{k≥t} γ^{k−t} r_k
```

用这个式子的单样本估计做梯度上升，就是 REINFORCE（[Sutton 和 Barto 教材](http://incompleteideas.net/book/the-book-2nd.html)第 13.3 节，更新式 13.8）。γ^t 这个因子在实践中常被省掉，相当于优化一个稍有不同的目标；LLM 里 γ = 1，不存在这个问题。

### 3.3 Ψ_t 的几种选择

[Schulman 等，GAE，arXiv 1506.02438](https://arxiv.org/abs/1506.02438) 第 2 节把策略梯度写成统一形式：

```text
g = E [ Σ_t Ψ_t · ∇_θ log π_θ(a_t | s_t) ]
```

Ψ_t 有多种选择，方差从高到低大致是：

| Ψ_t | 含义 | 特点 |
|---|---|---|
| Σ_{t=0}^∞ r_t | 整条轨迹的总回报 | 无偏，方差最大：动作要为它之前的奖励负责 |
| Σ_{t'≥t} r_{t'} | 动作之后的回报（reward-to-go） | 去掉了和当前动作无关的过去奖励（3.2 节） |
| Σ_{t'≥t} r_{t'} − b(s_t) | 减去基线 | 仍无偏，方差更小（3.4 节） |
| Q^π(s_t, a_t) | 动作价值 | 需要估计 |
| A^π(s_t, a_t) = Q − V | 优势 | 论文说它给出“几乎最低”的方差 |
| r_t + V(s_{t+1}) − V(s_t) | TD 残差 | 用 V 估计优势，V 不准时有偏，方差低（第 4 节） |

### 3.4 基线：为什么不改变期望

只要 b 只依赖状态、不依赖动作：

```text
E_{a~π} [ ∇_θ log π_θ(a|s) · b(s) ] = b(s) · Σ_a ∇_θ π_θ(a|s) = b(s) · ∇_θ Σ_a π_θ(a|s) = b(s) · ∇_θ 1 = 0
```

和 3.2 节是同一个论证：得分函数的期望为 0，乘上任何“在选动作之前就确定”的量，期望仍为 0。基线如果依赖当前动作，b 不能提到求和外面，这个推导就不成立，梯度会有偏。

**优势的直觉**：不减基线时，所有奖励为正的动作都被推高，只是推多推少的区别，信号很嘈杂。减去 V(s) 后，比平均好的动作被推高，比平均差的被压低。

### 3.5 基线怎样降低方差，哪个基线最好

记单样本估计 g = (R − b)·∇log π(a)，R 是这个动作拿到的回报。期望不随 b 变，所以最小化方差等于最小化二阶矩。只看单个参数分量 i，记 s_i(a) = ∂log π(a)/∂θ_i：

```text
E[g_i²] = E[ (R − b)² · s_i² ]
d/db E[g_i²] = −2 · E[ (R − b) · s_i² ] = 0
b*_i = E[ R · s_i² ] / E[ s_i² ]
```

对所有分量求和、用一个共同的常数 b，就得到 b* = E[R·‖∇log π‖²] / E[‖∇log π‖²]（推导）。这是按得分函数范数加权的平均回报。它和 V = E[R] 不完全相同，但通常很接近。实践里用 V(s) 做基线，因为它好学、含义清楚，而且接近最优。

基线不能消除所有方差。剩下的方差来自两处：奖励本身的噪声，以及同一状态下不同动作的回报差异（优势本身的大小）。

### 3.6 实现：bandit 上验证无偏和降方差

一个三臂 bandit：策略是 softmax(z)，奖励 R = μ[a] + 高斯噪声，所有奖励都为正。对 softmax 策略，∇_z log π(a) = onehot(a) − π，可以精确枚举每个动作，算出估计量的期望和方差，不用蒙特卡洛。

```python
import numpy as np

# 三臂 bandit：策略 π = softmax(z)，动作 a 的奖励 R = μ[a] + 噪声（方差 σ²）
# 所有奖励都为正：不减基线时每个被采到的动作都会被推高
z = np.array([0.5, 0.0, -0.5])
mu = np.array([1.0, 2.0, 3.0])
sigma2 = 0.25
pi = np.exp(z) / np.exp(z).sum()
K = len(z)
score = np.eye(K) - pi                    # score[a] = ∇_z log π(a) = onehot(a) − π，(K, K)

# 真梯度：J(z) = Σ_a π_a μ_a，用中心差分求
def J(z):
    p = np.exp(z) / np.exp(z).sum()
    return p @ mu
h = 1e-6
true_grad = np.array([(J(z + h * e) - J(z - h * e)) / (2 * h) for e in np.eye(K)])

def mean_and_var(b):
    """单样本估计 g = (R − b(a))·∇log π(a) 的精确期望和总方差（协方差矩阵的迹）。
    b 可以是标量（只依赖状态）或长度 K 的数组（依赖动作）。"""
    b = np.broadcast_to(np.asarray(b, float), (K,))
    mean = sum(pi[a] * (mu[a] - b[a]) * score[a] for a in range(K))
    second = sum(pi[a] * ((mu[a] - b[a]) ** 2 + sigma2) * score[a] @ score[a] for a in range(K))
    return mean, second - mean @ mean

g0, v0 = mean_and_var(0.0)               # 不减基线
gV, vV = mean_and_var(pi @ mu)           # 基线 = V = E[R]
s2 = (score ** 2).sum(axis=1)            # ‖∇log π(a)‖²
b_star = (pi * mu * s2).sum() / (pi * s2).sum()   # 推导出的方差最优常数基线
gS, vS = mean_and_var(b_star)

# 1) 只依赖状态的基线不改变期望
for g in (g0, gV, gS):
    assert np.allclose(g, true_grad, atol=1e-8)
# 2) 方差：不减基线 > 减 V > 减 b*，且 b* 是扫描网格上的最小值
assert v0 > vV > vS
grid = np.linspace(0, 4, 401)
assert abs(grid[np.argmin([mean_and_var(b)[1] for b in grid])] - b_star) < 0.01
# 3) 依赖动作的基线 b(a) = μ[a]：期望变成 0，梯度有偏
g_bad, _ = mean_and_var(mu)
assert np.allclose(g_bad, 0) and not np.allclose(g_bad, true_grad)

# 4) 蒙特卡洛核对：采 20 万个样本，经验方差和上面的闭式一致
rng = np.random.default_rng(0)
a = rng.choice(K, size=200_000, p=pi)
R = mu[a] + rng.normal(0, np.sqrt(sigma2), size=a.size)
for b, v in [(0.0, v0), (pi @ mu, vV)]:
    G = (R - b)[:, None] * score[a]
    assert np.allclose(G.mean(0), true_grad, atol=0.02)
    assert abs(G.var(0).sum() - v) / v < 0.02
print(f"V = {pi @ mu:.3f}, b* = {b_star:.3f}")
print(f"总方差：无基线 {v0:.3f}，减 V {vV:.3f}，减 b* {vS:.3f}")
```

结果：V = 1.680，b* = 2.000。总方差从不减基线的 2.800 降到减 V 的 0.405，再降到减 b* 的 0.342，三者期望都等于真梯度。依赖动作的“基线” b(a) = μ[a] 把期望变成了 0，梯度完全错了。

reward-to-go 的作用也可以这样验证。下面是一个两步回合，每步从同一个 softmax 策略选动作，第 t 步奖励 μ[a_t]：

```python
import itertools
import numpy as np

# 两步回合：每步从同一个 π = softmax(z) 选动作，第 t 步奖励 r_t = μ[a_t]
# 比较 Ψ_t = 整条轨迹回报 和 Ψ_t = reward-to-go 两种估计
z = np.array([0.3, -0.2, 0.1])
mu = np.array([1.0, 2.0, 4.0])
pi = np.exp(z) / np.exp(z).sum()
K = len(z)
score = np.eye(K) - pi

true_grad = 2 * pi * (mu - pi @ mu)      # ∇_z [2 Σ π μ]，softmax 的解析梯度

stats = {}
for name in ("total", "to_go"):
    m1, m2 = np.zeros(K), 0.0
    for a0, a1 in itertools.product(range(K), repeat=2):
        p = pi[a0] * pi[a1]
        r0, r1 = mu[a0], mu[a1]
        if name == "total":
            g = (r0 + r1) * (score[a0] + score[a1])
        else:                               # 第 0 步之后的回报是 r0 + r1，第 1 步之后只有 r1
            g = (r0 + r1) * score[a0] + r1 * score[a1]
        m1 += p * g
        m2 += p * g @ g
    stats[name] = (m1, m2 - m1 @ m1)

# 两者期望相同：去掉的 r0·∇log π(a1) 一项期望为 0（a1 在 r0 之后才采样）
assert np.allclose(stats["total"][0], true_grad)
assert np.allclose(stats["to_go"][0], true_grad)
# 这个例子里 reward-to-go 的方差更小
assert stats["to_go"][1] < stats["total"][1]
print({k: round(v[1], 3) for k, v in stats.items()})
```

两种估计的期望都等于真梯度。总方差从 32.27 降到 19.58。

## 4 Actor-Critic 与 GAE

### 4.1 Actor-Critic：基线从哪来

**问题**：REINFORCE 用蒙特卡洛回报 G_t 估计 Q，回合越长，G_t 里累加的随机性越多，方差越大。基线 V(s) 也要有人提供。

**做法**：同时训练两个网络。Actor 是策略 π_θ。Critic 是价值函数 V_φ，用回归拟合回报目标，最小化 (V_φ(s_t) − V_t^targ)²。Critic 有两个用处：做基线，以及用 bootstrap（用估计值代替真实的后续回报）缩短需要累加的随机奖励。

TD 残差把一步真实奖励和下一状态的估计价值拼起来：

```text
δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
```

如果 V 恰好等于真实的 V^π，δ_t 是优势的无偏估计：

```text
E[ δ_t | s_t, a_t ] = E[ r_t + γ V^π(s_{t+1}) | s_t, a_t ] − V^π(s_t) = Q^π(s_t, a_t) − V^π(s_t) = A^π(s_t, a_t)
```

δ_t 只含一步奖励的随机性，方差小。代价是 V 不准时，误差直接进入优势估计，产生偏差。

### 4.2 GAE：k 步估计的指数加权平均

在“一步 TD”和“完整蒙特卡洛”之间，可以只看 k 步真实奖励，再 bootstrap：

```text
Â_t^(1) = δ_t                                   = r_t + γV(s_{t+1}) − V(s_t)
Â_t^(2) = δ_t + γ δ_{t+1}                       = r_t + γ r_{t+1} + γ²V(s_{t+2}) − V(s_t)
Â_t^(k) = Σ_{l=0}^{k−1} γ^l δ_{t+l}             = Σ_{l=0}^{k−1} γ^l r_{t+l} + γ^k V(s_{t+k}) − V(s_t)
```

右边的等号是裂项相消：相邻 δ 里的 +γV(s_{t+l+1}) 和下一个 δ 里的 −V(s_{t+l+1})（乘 γ 后）互相抵掉。k 越大，用的真实奖励越多，偏差越小，方差越大。

GAE 不挑一个 k，而是对所有 k 做指数加权平均，权重 (1−λ)λ^{k−1}（GAE 论文式 16）：

```text
Â_t^GAE = (1 − λ) · ( Â_t^(1) + λ Â_t^(2) + λ² Â_t^(3) + … )
        = (1 − λ) · ( δ_t (1 + λ + λ² + …) + γ δ_{t+1} (λ + λ² + …) + γ² δ_{t+2} (λ² + …) + … )
        = (1 − λ) · ( δ_t / (1−λ) + γ δ_{t+1} · λ/(1−λ) + γ² δ_{t+2} · λ²/(1−λ) + … )
        = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}
```

第二行按 δ 归并：δ_{t+l} 出现在所有 k > l 的项里，系数是 γ^l (λ^l + λ^{l+1} + …) = γ^l λ^l / (1−λ)。

**两个极限**（GAE 论文式 17、18）：

```text
λ = 0：Â_t = δ_t                                  只看一步；V = V^π 时无偏，否则有偏；方差低
λ = 1：Â_t = Σ_l γ^l δ_{t+l} = Σ_l γ^l r_{t+l} − V(s_t)    蒙特卡洛回报减基线；不管 V 准不准都无偏；方差高
```

λ = 1 时所有中间的 V 都被裂项消掉，只剩 −V(s_t) 这个基线，所以 V 的误差不进入期望。

论文的说法：0 < λ < 1 在偏差和方差之间折中；γ < 1 不管价值函数准不准都会引入偏差，λ < 1 只在价值函数不准时引入偏差；实验中最优的 λ 明显小于最优的 γ。PPO 论文的 MuJoCo 实验用 γ = 0.99、λ = 0.95（附录表 3）。

**γ 和 λ 的区别**：γ 改变了被优化的目标本身（远期奖励打折）。λ 不改目标，只改优势估计在偏差和方差之间的位置。LLM 里 γ 常取 1，λ 常取 0.95 或 1。

### 4.3 从后往前递推，以及 critic 的回归目标

Σ (γλ)^l δ_{t+l} 可以从最后一步往前一次算完：

```text
A_{T} = 0                                      T 是这段数据之后的位置
A_t = δ_t + γλ · A_{t+1}
回报目标 V_t^targ = A_t + V(s_t)                ← 用来训练 critic
```

λ = 1 时 V_t^targ 就是蒙特卡洛回报，和 V 无关；λ < 1 时目标里含有 V 自己的预测，属于半梯度（semi-gradient）方法。

### 4.4 终止和截断：done 不是一个标志

回合可能因为两种原因结束，处理方式不同：

| | 终止（terminated） | 截断（truncated） |
|---|---|---|
| 例子 | 生成了 EOS；游戏结束；任务完成 | 达到最大 token 数；步数上限；rollout 分段 |
| 之后还有回报吗 | 没有，V(s_{t+1}) 按 0 算 | 有，只是没观察到 |
| δ_t | r_t − V(s_t) | r_t + γV(s_{t+1}) − V(s_t)，用估计值 bootstrap |
| 递推 | A_t = δ_t，切断 | 本段内 A_t = δ_t，下一段的优势不传过来 |

[Pardo 等，Time Limits in Reinforcement Learning，arXiv 1712.00378](https://arxiv.org/abs/1712.00378) 区分了两种情况。如果时间上限只是训练时的工程设置、不属于任务本身，应该在截断处用状态价值 bootstrap（论文叫 partial-episode bootstrapping）。如果时间上限属于任务，剩余时间应该放进状态里，否则同一个状态会对应不同的回报，破坏马尔可夫性。

把截断当终止，相当于告诉 critic “这里之后价值为 0”。长回答被截断时，critic 会学到“写长了就没价值”，这是一种隐性的长度惩罚。反过来，把终止当截断，会把下一条轨迹开头的价值传进当前轨迹。

LLM 里的截断还有第二层含义：没写完的回答通常拿不到可信的奖励（答案没写出来，或奖励模型只在 EOS 上训练过）。实践中有三种处理：判错或给固定惩罚、把这条样本 mask 掉、按超长程度软惩罚。DAPO 的做法见 [training/03 第 4.4 节](../../tracks/training/03-grpo-family/README.md#44-overlong-reward-shaping截断样本怎么处理)。

### 4.5 实现：GAE 与两个极限

```python
import numpy as np

def gae(rewards, values, next_values, terminated, gamma, lam):
    """rewards, values, next_values, terminated: (T,)。
    values[t] = V(s_t)，next_values[t] = V(s_{t+1})。
    terminated[t] = 1：s_{t+1} 是真正的终止状态，V(s_{t+1}) 按 0 算，且优势不再往前传。
    截断（时间上限）不算终止：next_values 照常填 V(s_{t+1})，bootstrap 由它完成。
    一段 rollout 里拼了多个回合时，回合边界要切断递推：真终止由 terminated 切，
    截断边界由调用方切（每个回合单独调用本函数，见下面的例子）。"""
    T = len(rewards)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - terminated[t]
        delta = rewards[t] + gamma * next_values[t] * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values                     # 优势、critic 的回归目标

rng = np.random.default_rng(0)
T, gamma = 6, 0.9
r = rng.normal(size=T)
V = rng.normal(size=T + 1)                       # V(s_0) … V(s_T)，故意随便给，代表不准的 critic
term = np.zeros(T); term[-1] = 1                 # 最后一步真正结束
nv = V[1:].copy()

delta = r + gamma * nv * (1 - term) - V[:-1]
# 1) λ = 0：优势就是一步 TD 残差
A0, _ = gae(r, V[:-1], nv, term, gamma, 0.0)
assert np.allclose(A0, delta)
# 2) λ = 1：优势 = 折扣蒙特卡洛回报 − V(s_t)；回归目标就是蒙特卡洛回报，和 V 无关
mc = np.array([sum(gamma ** (k - t) * r[k] for k in range(t, T)) for t in range(T)])
A1, ret1 = gae(r, V[:-1], nv, term, gamma, 1.0)
assert np.allclose(A1, mc - V[:-1])
assert np.allclose(ret1, mc)
# 3) 递推等于显式求和 Σ (γλ)^l δ_{t+l}
lam = 0.95
A, _ = gae(r, V[:-1], nv, term, gamma, lam)
explicit = [sum((gamma * lam) ** (k - t) * delta[k] for k in range(t, T)) for t in range(T)]
assert np.allclose(A, explicit)
# 4) critic 不准时 λ 的作用：给 V 加偏差，λ=1 的回归目标不变，λ=0 的目标跟着偏
_, ret0_a = gae(r, V[:-1], nv, term, gamma, 0.0)
_, ret0_b = gae(r, V[:-1] + 1.0, nv + 1.0, term, gamma, 0.0)
_, ret1_b = gae(r, V[:-1] + 1.0, nv + 1.0, term, gamma, 1.0)
assert np.allclose(ret1_b, ret1)                 # λ=1：对 V 的误差免疫
assert not np.allclose(ret0_b, ret0_a)           # λ=0：r + γV(s') 里带着 V 的误差

# 5) 终止 vs 截断：同一条 3 步轨迹，最后一步之后状态的价值是 V(s_3) = 5
r3, v3 = np.zeros(3), np.zeros(3)
nv3 = np.array([0.0, 0.0, 5.0])
A_term, _ = gae(r3, v3, nv3, np.array([0, 0, 1.0]), 1.0, 1.0)
A_trunc, _ = gae(r3, v3, nv3, np.array([0, 0, 0.0]), 1.0, 1.0)
assert np.allclose(A_term, 0)                    # 真终止：后面没有价值
assert np.allclose(A_trunc, 5)                   # 截断：用 V(s_3) bootstrap，价值传回每一步

# 6) 两个回合拼在一个 buffer 里，忘了在边界切断：后一回合的奖励漏进前一回合
rA, rB = np.zeros(3), np.array([0, 0, 10.0])
cat = np.concatenate
z6 = np.zeros(6)
A_bug, _ = gae(cat([rA, rB]), z6, z6, np.zeros(6), 1.0, 1.0)            # 漏标边界
A_ok, _ = gae(cat([rA, rB]), z6, z6, np.array([0, 0, 1, 0, 0, 1.0]), 1.0, 1.0)
assert A_bug[0] == 10 and A_ok[0] == 0

# 7) LLM 场景：奖励只在最后一个 token，V ≡ 0（例如 critic 刚初始化），γ = 1
#    第 t 个 token 拿到的信号是 λ^(T−1−t)·R，回答越长前面的 token 越接近 0
Tl, R = 200, 1.0
rl = np.zeros(Tl); rl[-1] = R
tl = np.zeros(Tl); tl[-1] = 1
for lam_, first in [(1.0, 1.0), (0.95, 0.95 ** 199)]:
    Al, _ = gae(rl, np.zeros(Tl), np.zeros(Tl), tl, 1.0, lam_)
    assert np.isclose(Al[0], first)
print("λ=0.95、200 token 时第一个 token 的信号：", round(0.95 ** 199, 6))
```

第 7 个检查对应 LLM 的一个具体问题。奖励只在末尾、critic 还没学好时，λ = 0.95 下第一个 token 只收到 0.95^199 ≈ 3.7×10⁻⁵ 的信号。[VC-PPO，arXiv 2503.01491](https://arxiv.org/abs/2503.01491)（2025-03）第 3.3 节把这叫奖励信号衰减（reward signal decay），并让 critic 的回归目标用 λ = 1、策略的优势用 λ = 0.95（Decoupled-GAE）。[VAPO，arXiv 2504.05118](https://arxiv.org/abs/2504.05118)（2025-04）第 4.2 节进一步让策略的 λ 随回答长度 l 变化：λ_policy = 1 − 1/(αl)，α 是超参数。

### 4.6 Critic 的代价

在 LLM 上，critic 通常和策略同等规模，常从奖励模型初始化。它多一份参数、梯度和优化器状态。只在结尾有奖励时，它要学会给每个中间 token 估值，很难训准。VC-PPO 第 3.2 节分析了从奖励模型初始化 critic 的问题。奖励模型只在 EOS 位置上训练过。越靠前的前缀越不完整，它给的分越低。critic 的任务却是估计每个前缀的期望奖励。于是每一步的 δ_t 都带正偏差，沿轨迹累加，越靠前的 token 优势越偏正。论文报告，这让模型倾向于更早结束，输出长度塌缩。它们的处理是先用固定的 SFT 策略采样，用 λ = 1 的回报离线预训练 critic，直到 value loss 和 explained variance 达标（Value-Pretraining）。

## 5 PPO：限制每次更新的幅度

出处：[Schulman 等，Proximal Policy Optimization Algorithms，arXiv 1707.06347](https://arxiv.org/abs/1707.06347)，2017-07。

### 5.1 问题：一批数据只能用一次

策略梯度的期望是在当前策略 π_θ 下取的。更新一步之后，手里的数据就成了“旧策略”采的，严格说不能再用。LLM 里采样一批回答很贵（每条回答都是一次完整推理），只用一次很浪费。

PPO 论文第 2.1 节说，在同一批轨迹上对 L^PG = Ê[log π_θ(a|s)·Â] 做多步优化“没有充分依据”，经验上常导致破坏性的过大更新。

### 5.2 重要性采样：用旧策略的样本估计新策略的期望

对任意 f，有

```text
E_{a~π_θ}[ f(a) ] = Σ_a π_θ(a) f(a) = Σ_a π_old(a) · (π_θ(a)/π_old(a)) · f(a) = E_{a~π_old}[ ρ(a) · f(a) ]
```

ρ = π_θ/π_old 就是重要性比率。代入优势，得到 TRPO 和 PPO 共用的代理目标（PPO 论文式 6，CPI 指 conservative policy iteration）：

```text
L^CPI(θ) = Ê_t [ ρ_t(θ) · Â_t ]          ρ_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)，ρ_t(θ_old) = 1
```

这个式子只修正了动作分布，没修正状态分布：状态仍然是 π_old 访问到的。[TRPO，arXiv 1502.05477](https://arxiv.org/abs/1502.05477) 第 2 节把这种近似记作 L_π(π̃)，并说明它和真实目标在 θ_old 处一阶相等（式 4）：

```text
∇_θ L^CPI |_{θ=θold} = Ê[ ∇_θ ρ_t · Â_t ] = Ê[ ρ_t ∇log π_θ · Â_t ] |_{ρ=1} = Ê[ ∇log π_θ · Â_t ] = 策略梯度
```

所以离 θ_old 近时，优化 L^CPI 和优化真实目标是一回事；走远了，状态分布的变化被忽略，L^CPI 就不可信了。TRPO 定理 1 给出一个下界：真实目标 ≥ L_π(π̃) − C·(max_s D_TV)²，C = 4·max|A|·γ/(1−γ)²。新旧策略差得越多，代理目标越不可信。这就是“信任域”的来源。

TRPO 用 KL 约束 Ê[KL(π_old, π_θ)] ≤ δ 加共轭梯度求解。PPO 想要一个只用一阶优化器、能做多轮 minibatch 更新的版本。

### 5.3 clip 目标

PPO 论文式 7：

```text
L^CLIP(θ) = Ê_t [ min( ρ_t(θ)·Â_t ,  clip(ρ_t(θ), 1−ε, 1+ε)·Â_t ) ]
```

论文建议 ε = 0.2；在其 MuJoCo 对比里，ε = 0.2 的平均归一化分数最高（表 1：ε = 0.1、0.2、0.3 分别为 0.76、0.82、0.70，不 clip 也不加惩罚为 −0.39）。

**为什么叫悲观下界**。min 的第一项就是 L^CPI。所以对任意 ρ 和 Â，逐项都有

```text
min( ρÂ, clip(ρ)·Â ) ≤ ρÂ
```

在 [1−ε, 1+ε] 内两者相等，在 θ_old 附近一阶相同。论文第 3 节的原话是：最终目标是未截断目标的下界（pessimistic bound）；只在比率变化会让目标变好时忽略它，在变化会让目标变坏时保留它。

拆开看一个 token：

- **Â > 0**：目标希望 ρ 变大。ρ > 1+ε 后，clip 项 (1+ε)Â 更小，min 取它，它是常数，梯度为 0。ρ < 1−ε 时（概率反而被压低了），ρÂ 更小，min 取未截断项，梯度照常，可以拉回来。
- **Â < 0**：目标希望 ρ 变小。ρ < 1−ε 后，clip 项 (1−ε)Â 更小（更负），min 取它，梯度为 0。ρ > 1+ε 时（概率反而被推高了），ρÂ 更负，min 取它，梯度照常。

| 情况 | 目标想做什么 | clip 和 min 的效果 |
|---|---|---|
| Â > 0 | 提高这个动作的概率，ρ 变大 | ρ 超过 1+ε 后目标不再增长，梯度为 0，不会推得更高 |
| Â > 0 且 ρ < 1−ε | 概率已经被压低了 | min 取未 clip 的项，梯度照常，可以拉回来 |
| Â < 0 | 降低这个动作的概率，ρ 变小 | ρ 低于 1−ε 后梯度为 0，不会压得更低 |
| Â < 0 且 ρ > 1+ε | 概率反而被推高了 | min 取未 clip 的项，梯度照常，可以纠正 |

一句话：clip 只拦“朝着有利方向走太远”的更新；min 保证目标是未 clip 目标的下界，变坏的方向不被拦。

梯度可以写成一个式子。对 log π_θ 求导，dρ/dlog π = ρ：

```text
∂L^CLIP_t / ∂log π_θ(a_t|s_t) = ρ_t · Â_t · 1[ 未截断项被 min 选中 ]
                              = ρ_t · Â_t · 1[ not ( (Â_t > 0 且 ρ_t > 1+ε) 或 (Â_t < 0 且 ρ_t < 1−ε) ) ]
```

### 5.4 clip 不是硬信任域

clip 只是去掉了“继续推”的梯度，没有任何力把 ρ 拉回区间。同一批数据跑多个 epoch、多个 minibatch，参数是共享的：别的样本的梯度照样会把这个样本的 ρ 推出区间。[Engstrom 等，Implementation Matters in Deep Policy Gradients，arXiv 2005.12729](https://arxiv.org/abs/2005.12729)（ICLR 2020）第 4 节指出，第一步更新时所有 ρ = 1，这一步和优化未截断目标完全相同；步子有多大取决于优化过程，而不是目标本身。论文图 2（MuJoCo Humanoid-v2）显示，PPO 训练中每步的最大比率持续超出 1+ε；平均 KL 倒是控制住了。

所以实践中会加两道保险：

- **KL 早停**：每个 minibatch 后估计 KL(π_old, π_θ)，超过阈值就停止本轮剩余的更新。[OpenAI Spinning Up 的 PPO 实现](https://spinningup.openai.com/en/latest/algorithms/ppo.html) 默认 target_kl = 0.01，近似 KL 超过 1.5 × target_kl 时停止策略更新。
- **监控 clip 比例**（clipfrac）：被截断的 token 占比。它持续偏高，说明学习率太大或 epoch 太多。

### 5.5 完整 loss：价值损失和熵

PPO 论文式 9，策略和价值共享参数时：

```text
L(θ) = Ê_t [ L^CLIP_t(θ) − c1 · (V_θ(s_t) − V_t^targ)² + c2 · S[π_θ](s_t) ]          要最大化
```

c1 是价值损失系数，c2 是熵奖励系数，S 是策略在 s_t 处的熵 −Σ_a π log π。论文 Atari 实验用 c1 = 1、c2 = 0.01（表 5）；MuJoCo 实验不共享参数，也不用熵奖励。

各项的作用：

- **价值损失**：训练 critic 拟合 4.3 节的回报目标。不共享参数时它是单独的 loss，c1 不起作用。
- **熵奖励**：防止策略过早变成确定性的，保留探索。LLM 里很少用它，防退化主要靠对参考模型的 KL（第 6 节）。
- **价值裁剪**（value clipping）：论文没写，但很多实现有。Engstrom 等第 3 节列出 OpenAI baselines 的做法：L^V = max[(V_θ − V^targ)², (clip(V_θ, V_old − ε, V_old + ε) − V^targ)²]，ε 和策略的 clip 范围相同。[Huang 等，The N+ Implementation Details of RLHF with PPO，arXiv 2403.17031](https://arxiv.org/abs/2403.17031)（2024-03）表 7 的 RLHF 默认配置也打开了价值裁剪，ε̂ = 0.2，c1 = 0.1。

### 5.6 KL 惩罚版本

PPO 论文第 4 节给了 clip 的替代方案（式 8）：

```text
L^KLPEN(θ) = Ê_t [ ρ_t(θ)·Â_t − β · KL[π_θold(·|s_t), π_θ(·|s_t)] ]
每次策略更新后算 d = Ê_t[ KL[π_θold, π_θ] ]：
    d < d_targ / 1.5  →  β ← β / 2
    d > d_targ × 1.5  →  β ← β × 2
```

论文说 1.5 和 2 是启发式选的，算法对它们不敏感；β 的初值也不重要，会很快调整。论文实验里自适应 KL 的效果不如 clip（表 1：d_targ = 0.01 时为 0.74），只作为基线保留。

### 5.7 训练循环

PPO 论文算法 1：

```text
for iteration = 1, 2, …:
    N 个 actor 各用 π_θold 跑 T 步，收集数据
    用 GAE 算优势 Â_1 … Â_T
    在这 N·T 步数据上，用 minibatch 大小 M 做 K 个 epoch 的优化
    θ_old ← θ
```

论文 MuJoCo 设置是 T = 2048、10 个 epoch、minibatch 64、Adam 学习率 3×10⁻⁴（表 3）。

在 LLM 上，“跑 T 步”换成“对一批 prompt 各采一条回答”，一条回答是一条轨迹。第一个 minibatch 时 θ = θ_old，ρ = 1，clip 不起作用。只有在第二个及以后的 minibatch 或 epoch，比率才会偏离 1。所以 K = 1、只有一个 minibatch 时，PPO 退化成普通的带优势的策略梯度。

### 5.8 实现：clip 目标的梯度截断

下面的代码按 5.3 节的四种情况检查梯度，并验证悲观下界。

```python
import numpy as np

eps = 0.2

def clip_obj(r, A):
    """PPO 单个 token 的 clip 目标（要最大化）。r: 比率，A: 优势，可广播。"""
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)

def clip_grad_logp(r, A):
    """d clip_obj / d log π_θ。比率 r = exp(logp − logp_old)，所以 dr/dlogp = r。
    min 取到未截断项时梯度是 r·A；取到截断项时，截断项对 r 是常数，梯度为 0。"""
    unclipped_active = r * A <= np.clip(r, 1 - eps, 1 + eps) * A
    return np.where(unclipped_active, r * A, 0.0)

def num_grad(r, A, h=1e-6):
    lp = np.log(r)
    return (clip_obj(np.exp(lp + h), A) - clip_obj(np.exp(lp - h), A)) / (2 * h)

# 四种情况：A 的正负 × r 在区间哪一侧
cases = {                          # (r, A): 期望梯度是否为 0
    "A>0, r=1.5（已推高太多）": (1.5, 1.0, True),
    "A>0, r=0.5（反被压低）":   (0.5, 1.0, False),
    "A<0, r=0.5（已压低太多）": (0.5, -1.0, True),
    "A<0, r=1.5（反被推高）":   (1.5, -1.0, False),
    "A>0, r=1.1（区间内）":     (1.1, 1.0, False),
}
for name, (r, A, zero) in cases.items():
    g = clip_grad_logp(r, A)
    assert np.isclose(g, num_grad(r, A), atol=1e-5), name
    assert (g == 0) == zero, name

# 悲观下界：对任意 r 和 A，clip 目标 ≤ 未截断的 CPI 目标 r·A，在 [1−ε, 1+ε] 内相等
r = np.linspace(0.01, 3, 3000)
for A in (-2.0, -0.3, 0.7, 2.0):
    L_clip, L_cpi = clip_obj(r, A), r * A
    assert np.all(L_clip <= L_cpi + 1e-12)
    inside = (r >= 1 - eps) & (r <= 1 + eps)
    assert np.allclose(L_clip[inside], L_cpi[inside])
    # 有利方向走出区间后目标封顶：A>0 时 r>1+ε 目标恒为 (1+ε)A，A<0 时 r<1−ε 目标恒为 (1−ε)A
    cap = r > 1 + eps if A > 0 else r < 1 - eps
    assert np.allclose(L_clip[cap], (1 + eps if A > 0 else 1 - eps) * A)

# 在 r = 1（第一个 minibatch）处，clip 目标和 CPI、和普通策略梯度 log π·A 的梯度相同
for A in (-1.0, 1.0):
    assert np.isclose(clip_grad_logp(1.0, A), A)

# 对比：只做 clip、不取 min 的“错误写法”，在 A>0、r=0.5 时梯度也被清零，错误方向无法纠正
wrong = lambda r, A: np.clip(r, 1 - eps, 1 + eps) * A
lp = np.log(0.5)
g_wrong = (wrong(np.exp(lp + 1e-6), 1.0) - wrong(np.exp(lp - 1e-6), 1.0)) / 2e-6
assert g_wrong == 0 and clip_grad_logp(0.5, 1.0) == 0.5

# PPO 论文第 4 节的自适应 KL 系数：d < d_targ/1.5 时 β 减半，d > 1.5·d_targ 时加倍
def adapt_beta(beta, d, d_targ):
    if d < d_targ / 1.5:
        return beta / 2
    if d > d_targ * 1.5:
        return beta * 2
    return beta
assert adapt_beta(1.0, 0.001, 0.01) == 0.5
assert adapt_beta(1.0, 0.02, 0.01) == 2.0
assert adapt_beta(1.0, 0.012, 0.01) == 1.0
print("clip 梯度检查通过")
```

“只 clip 不取 min”的写法是一个常见错误：Â > 0 而 ρ 已经掉到 1−ε 以下时，它把梯度也清零了，策略在错误方向上走远后拉不回来。

## 6 KL 约束：三种“别走太远”

面试常把这几种混在一起。它们约束的对象和目的不同。

| 约束 | 相对谁 | 目的 | 出处 |
|---|---|---|---|
| clip | 本轮采样用的旧策略 π_old | 一批数据复用多轮时，每次更新别太大 | PPO 式 7 |
| KL 惩罚（PPO 的替代方案） | π_old | 同上；β 自适应：KL 小于 d_targ/1.5 时 β 减半，大于 d_targ×1.5 时 β 加倍 | PPO 式 8 |
| 对参考模型的 KL | 参考模型 π_ref（通常是 SFT 模型） | 防止策略跑出奖励模型可信的区域，防止 reward hacking 和语言退化 | Ziegler 等 2019；InstructGPT |

前两种管的是“一步”：每轮采样后 π_old 都会更新成当前策略，约束随之重置。第三种管的是“整个训练”：π_ref 固定不动，累计偏离多少都算。所以有了 clip，仍然需要对参考模型的 KL。

### 6.1 对参考模型的 KL 加进奖励

LLM 的 RLHF 里，对参考模型的 KL 一般加进奖励（[Ziegler 等，arXiv 1909.08593](https://arxiv.org/abs/1909.08593)，式 2）：

```text
R(x, y) = r(x, y) − β · log( π(y|x) / ρ(y|x) )        ρ 为初始策略
```

y 从 π 采样时，这一项的期望是 β·KL(π, ρ)。论文说这一项有几个作用：起熵奖励的作用，防止策略走出奖励模型 r 有效的范围；在风格续写任务里，它还是任务定义的一部分，靠它保持连贯和切题。

实现时把它拆到每个 token 上。因为 log π(y|x) = Σ_t log π(y_t | x, y_<t)：

```text
r_t = −β · ( log π(y_t|x, y_<t) − log ρ(y_t|x, y_<t) )         t < T
r_T = −β · ( log π(y_T|…) − log ρ(y_T|…) ) + r(x, y)           最后一个 token 再加奖励模型分数
Σ_t r_t = R(x, y)
```

这样每个 token 都有一个非零奖励，GAE 有了中间信号。Huang 等的图 9 展示了一条回答的逐 token 奖励：EOS 位置是奖励模型的分数，其余位置是逐 token 的 KL 惩罚。

**β 怎么定**。固定 β 时，不同随机种子训出的 KL 差别可能很大。Ziegler 等用了一个对数空间的比例控制器，让 KL 贴近目标值：

```text
e_t = clip( (KL(π_t, ρ) − KL_target) / KL_target, −0.2, 0.2 )
β_{t+1} = β_t · (1 + K_β · e_t)                K_β = 0.1
```

和 PPO 的自适应 KL 相比，它调的是对参考模型的 KL，而且每次只按比例小幅调整。Huang 等表 7 的 RLHF 配置用固定 β = 0.05。

GRPO 则把 KL 从奖励里拿出来直接放进 loss，见 [training/03](../../tracks/training/03-grpo-family/README.md)。两种放法的区别：放进奖励，KL 会经过 GAE 影响每个 token 的优势，也会进 critic 的回归目标；放进 loss，优势只由任务奖励决定。

### 6.2 KL 怎么估计

从 q 采样、r = p(x)/q(x)，估计 KL[q, p]（[Schulman，Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)，2020-03）：

| 估计量 | 公式 | 性质 |
|---|---|---|
| k1 | −log r | 无偏，方差大，单个样本可以为负 |
| k2 | (log r)² / 2 | 有偏但偏差小，方差低，恒非负 |
| k3 | (r − 1) − log r | 无偏，方差低，恒非负 |

GRPO 用的就是 k3 形式。k3 的无偏性和非负性推导、作为 loss 求导时对应哪个方向的 KL，见 [training/03 第 2.4 节](../../tracks/training/03-grpo-family/README.md#24-k3-kl-估计量为什么无偏且非负)。

上面逐 token 的 −β(log π − log ρ) 就是 k1 估计。它在单个 token 上可以为正，即 KL 惩罚变成奖励；求和后的期望才是 β·KL。监控时如果看到“KL 为负”，通常就是 k1 在少量样本上的波动。

## 7 LLM 场景下 RL 的特殊性

| 方面 | 经典控制（如 MuJoCo） | LLM |
|---|---|---|
| 动作空间 | 低维连续 | 词表大小的离散空间，每步一个 token |
| 轨迹长度 | 数百到数千步 | 数百到上万 token |
| 一条轨迹 | 一个回合 | 一个回答（Agent 场景是一次完整的多轮交互） |
| 奖励 | 每步都有 | 通常只在结尾，稀疏 |
| 策略初始化 | 随机 | 从 SFT 模型出发，已经很强 |
| 采样成本 | 模拟器便宜 | 每条回答都是一次完整推理，rollout 常是最大开销 |
| 价值模型 | 小网络 | 常和策略同规模，显存翻倍，且难训准 |
| 防退化 | 靠熵奖励鼓励探索 | 主要靠对参考模型的 KL，防止 reward hacking |

### 7.1 一个回答是一条轨迹：PPO 的各个量落在哪里

把第 4、5 节的量对到一条回答上。设 prompt 有 P 个 token，回答 y 有 T 个 token：

| 量 | 位置 | 形状 | 说明 |
|---|---|---|---|
| 状态 s_t | prompt + y_<t | — | 不存，模型前向时隐式给出 |
| log π_θ(y_t\|s_t) | 每个回答 token | (B, T) | 用 logits 在第 P+t−1 个位置的输出取 y_t 的 log 概率，错一位是常见 bug |
| log π_old、log π_ref | 同上 | (B, T) | rollout 后用训练引擎各算一次，之后固定 |
| r_t | 每个回答 token | (B, T) | 逐 token KL 惩罚；最后一个有效 token 再加奖励模型分数 |
| V(s_t) | 每个回答 token | (B, T) | critic 在同样位置上的标量输出 |
| Â_t、V^targ_t | 每个回答 token | (B, T) | GAE 沿 T 维从后往前算，γ 常取 1 |
| response mask | 每个回答 token | (B, T) | prompt、padding、工具结果为 0 |

回答长度不同，batch 里要补齐。padding 位置的 r、V、Â 都必须被 mask 掉，GAE 的递推要从每条回答的最后一个有效 token 开始，而不是从补齐后的末尾开始。

### 7.2 token 级与序列级：单位不一致

第 2 节说过，奖励是整条回答的，概率比和优势却是逐 token 的。这带来三个问题：

- **信用分配**：结尾一个奖励，怎么分给几百个 token？PPO 靠 critic 的 V(s_t) 差值；去掉 critic 的方法（RLOO、GRPO）让同一回答的所有 token 共享一个优势。
- **loss 怎么聚合**：按 token 平均还是按回答平均，会改变每个 token 的权重，产生长度偏差。见 [training/03 第 3.1 节](../../tracks/training/03-grpo-family/README.md#31-长度偏差1o_i-让-token-的权重依赖回答长度)。
- **比率的单位**：PPO 的比率逐 token 算，clip 也逐 token 做；GSPO 改成序列级比率。见 [training/03 第 5 节](../../tracks/training/03-grpo-family/README.md#5-gspo把重要性比率放到序列级)。

### 7.3 奖励只在末尾

奖励稀疏让 critic 难学（4.5、4.6 节）。也让一个现象更明显：回答越长，噪声累加越多。对策大致有三类：

- **去掉 critic，用组内基线**。同一 prompt 采 k 条，RLOO（[Ahmadian 等](https://arxiv.org/abs/2402.14740)）对每个样本用其余 k−1 个样本的平均奖励做基线；GRPO 用组内均值和标准差。代价是每个 prompt 要采多条。它为什么无偏，见 [training/03 第 2.2 节](../../tracks/training/03-grpo-family/README.md#22-组内均值为什么是合理的基线)。
- **保留 critic，但改 GAE**：VC-PPO、VAPO 的 Decoupled-GAE 和 Length-Adaptive GAE（4.5 节）。
- **加中间奖励**：过程奖励、分步验证，见 [training/05](../../tracks/training/05-reward-verifier/README.md)。

### 7.4 其他设计选择

- **非模型生成的 token 要屏蔽**。Agent 轨迹里的工具结果不是策略的动作，不能算进策略梯度，见 [training/04](../../tracks/training/04-agentic-rl/README.md)。
- **奖励从哪来**。奖励模型、规则、可执行的检查各有利弊，见 [training/05](../../tracks/training/05-reward-verifier/README.md)。
- **截断回答怎么给奖励**。Huang 等第 23 条细节记录了一种做法（EOS trick）：没生成 EOS 的回答拿不到奖励模型的可信分数，直接给固定奖励 −1，这也间接约束了长度。

## 8 实现：小 MDP 上跑 PPO

把第 4、5 节拼成一个完整的 PPO。环境是一个“两 token 回答”：词表 3 个 token，回答长度固定为 2，奖励只在末尾给。只有回答 (2, 1) 拿满分 1；(0, 0) 拿 0.3，是一个次优的诱饵；其余为 0。第一个 token 本身没有奖励，策略要靠 critic 学到“先选 2 是好的”。

策略是表格型的：4 个状态（空前缀、已生成 0、已生成 1、已生成 2），每个状态一组 logits。critic 也是表格型。期望奖励可以精确算，不依赖采样估计。

```python
import numpy as np

# 一个“两 token 回答”的小 MDP：词表 3 个 token，回答长度固定为 2，奖励只在末尾。
# 状态 0 = 空前缀；状态 1+a0 = 已生成第一个 token a0。转移确定（把 token 拼到末尾）。
# 奖励表 R[a0, a1]：只有 (2, 1) 拿满分，第一个 token 选 2 本身没有奖励，要靠信用分配学到。
V_SIZE = 3
R = np.zeros((V_SIZE, V_SIZE)); R[2, 1] = 1.0; R[0, 0] = 0.3

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def expected_reward(logits):
    p = softmax(logits)                               # (4, 3)，每个状态一行
    return sum(p[0, a0] * p[1 + a0, a1] * R[a0, a1] for a0 in range(3) for a1 in range(3))

def rollout(logits, n, rng):
    """返回 states, actions, logp_old, rewards，形状都是 (n, 2)。"""
    p = softmax(logits)
    a0 = np.array([rng.choice(3, p=p[0]) for _ in range(n)])
    a1 = np.array([rng.choice(3, p=p[1 + x]) for x in a0])
    S = np.stack([np.zeros(n, int), 1 + a0], 1)
    A = np.stack([a0, a1], 1)
    logp = np.log(p[S, A])
    rew = np.stack([np.zeros(n), R[a0, a1]], 1)       # 中间 token 奖励 0，末尾给 R
    return S, A, logp, rew

def gae_2step(rew, v, gamma=1.0, lam=0.95):
    """v: (n, 2) 为 V(s_0)、V(s_1)。第 2 步后回合终止。"""
    d1 = rew[:, 1] - v[:, 1]                          # 终止：V(s_2) = 0
    d0 = rew[:, 0] + gamma * v[:, 1] - v[:, 0]
    a1 = d1
    a0 = d0 + gamma * lam * a1
    adv = np.stack([a0, a1], 1)
    return adv, adv + v

def ppo(n_iters=30, n=64, epochs=4, mb=16, lr=1.0, eps=0.2, c2=0.01, use_clip=True, seed=0):
    rng = np.random.default_rng(seed)
    logits = np.zeros((4, V_SIZE))                    # 策略：每个状态一组 logits
    V = np.zeros(4)                                   # critic：表格型
    hist, kls, clipfracs = [expected_reward(logits)], [], []
    for _ in range(n_iters):
        S, A, logp_old, rew = rollout(logits, n, rng)
        p_old = softmax(logits)
        adv, ret = gae_2step(rew, V[S])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8) # batch 级优势标准化，常见实现
        S, A, logp_old, adv, ret = (x.reshape(-1) for x in (S, A, logp_old, adv, ret))
        for _ in range(epochs):
            perm = rng.permutation(len(S))
            for idx in np.split(perm, len(S) // mb):
                p = softmax(logits)
                s, a = S[idx], A[idx]
                ratio = np.exp(np.log(p[s, a]) - logp_old[idx])
                Ab = adv[idx]
                if use_clip:
                    active = ratio * Ab <= np.clip(ratio, 1 - eps, 1 + eps) * Ab
                    coef = np.where(active, ratio * Ab, 0.0)    # d L / d log π
                else:
                    coef = ratio * Ab                            # 不截断的 CPI 目标
                g = np.zeros_like(logits)
                onehot = np.eye(V_SIZE)[a]
                np.add.at(g, s, coef[:, None] * (onehot - p[s]))
                # 熵奖励 c2·H 对 logits 的梯度：−π(log π + H)
                H = -(p[s] * np.log(p[s])).sum(1, keepdims=True)
                np.add.at(g, s, c2 * (-p[s] * (np.log(p[s]) + H)))
                logits += lr * g / len(idx)                     # 梯度上升
                # critic：对回报目标做 MSE 回归
                gv = np.zeros(4)
                np.add.at(gv, s, V[s] - ret[idx])
                V -= 0.5 * gv / np.bincount(s, minlength=4).clip(1)
        # 本轮更新前后在采样状态上的平均 KL(π_old ‖ π_new)，以及最后的 clip 比例
        p_new = softmax(logits)
        kls.append(np.mean([(p_old[x] * np.log(p_old[x] / p_new[x])).sum() for x in S]))
        ratio = p_new[S, A] / np.exp(logp_old)
        clipfracs.append(np.mean(np.abs(ratio - 1) > eps))
        hist.append(expected_reward(logits))
    return np.array(hist), np.array(kls), np.array(clipfracs), logits, V

hist, kls, cf, logits, V = ppo()
p = softmax(logits)
# 1) 期望奖励（精确计算，不是采样估计）从 1/9·1.3 ≈ 0.144 提高到接近 1
assert np.isclose(hist[0], 1.3 / 9)
assert hist[-1] > 0.9 and hist[-1] > hist[0] + 0.7
# 2) 信用分配：第一个 token 本身没有奖励，策略仍学会先选 2，因为 critic 给了 V(s=“2”) 高值
assert p[0, 2] > 0.9 and p[3, 1] > 0.9
assert V[3] > 0.8
# 3) clip 在起作用：多 epoch 后有一部分样本的比率走出了 [1−ε, 1+ε]
assert cf[:5].max() > 0
# 4) 同样的学习率和 epoch 数，去掉 clip 后单轮 KL 更大：clip 在限制步长
_, kls_noclip, *_ = ppo(use_clip=False)
assert kls_noclip[:5].mean() > kls[:5].mean()
print("期望奖励：", hist[[0, 5, 10, 30]].round(3))
print("前 5 轮平均 KL：clip", kls[:5].mean().round(4), "不 clip", kls_noclip[:5].mean().round(4))
```

运行结果：期望奖励在第 0、5、10、30 轮分别为 0.144、0.848、0.982、0.999。策略没有停在诱饵 (0, 0) 上。前 5 轮的单轮平均 KL，有 clip 时约 0.035，去掉 clip 后约 0.155，多了 4 倍多。clip 比例在前几轮达到 0.8 左右：同一批数据跑 4 个 epoch，ρ 很快就走出了区间，这正是 5.4 节说的“clip 不是硬信任域”。换随机种子 0 到 7，这些结论都成立（在本地另行验证过，代码里只跑种子 0）。

几处和真实实现对应的地方：

- `adv` 在 batch 内标准化。PPO 论文没写，但多数实现都做；它让学习率对奖励尺度不敏感。
- `coef` 是 clip 目标对 log π 的导数（5.3 节），乘上 softmax 的得分函数 onehot − π 得到对 logits 的梯度。神经网络实现里这一步由自动求导完成。
- critic 的回归目标 `ret` 在本轮采样后算好，之后固定，不随 V 的更新而改变。

## 9 数值例子：一条三 token 回答

下面的数都能手算。代码只是核对。

**(a) GAE**。γ = 1，回答 3 个 token，奖励只在最后：r = (0, 0, 1)。critic 给出 V(s_0), V(s_1), V(s_2) = (0.2, 0.4, 0.6)，第 3 个 token 后终止，V(s_3) = 0。

```text
δ_0 = 0 + 0.4 − 0.2 = 0.2
δ_1 = 0 + 0.6 − 0.4 = 0.2
δ_2 = 1 + 0   − 0.6 = 0.4
```

| λ | Â_0 | Â_1 | Â_2 | 说明 |
|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0.4 | 就是 δ_t |
| 0.5 | 0.2 + 0.5·0.4 = 0.4 | 0.2 + 0.5·0.4 = 0.4 | 0.4 | 从后往前：Â_1 = δ_1 + 0.5·Â_2 |
| 1 | 0.8 | 0.6 | 0.4 | = 1 − V(s_t)，蒙特卡洛回报减基线 |

λ = 1 时三个 token 的优势之和是 1.8；λ = 0 时只有 0.8。V 偏低时，λ 越小，奖励越难传到前面的 token。

**(b) PPO clip**。ε = 0.2，某个 token 在旧策略下的概率 π_old = 0.5。

| Â | π_θ | ρ | ρÂ | clip(ρ)·Â | min | 梯度 |
|---|---|---|---|---|---|---|
| +1 | 0.65 | 1.3 | 1.3 | 1.2 | 1.2 | 0：已推高太多 |
| +1 | 0.30 | 0.6 | 0.6 | 0.8 | 0.6 | 照常：拉回来 |
| −1 | 0.30 | 0.6 | −0.6 | −0.8 | −0.8 | 0：已压低太多 |
| −1 | 0.65 | 1.3 | −1.3 | −1.2 | −1.3 | 照常：纠正 |

每一行 min 都不大于 ρÂ，这就是悲观下界。

**(c) 逐 token KL 奖励**。β = 0.05，奖励模型分数 4.5，4 个回答 token 上 π 和 π_ref 的概率分别是 (0.6, 0.3, 0.9, 0.5) 和 (0.5, 0.4, 0.8, 0.5)。逐 token 奖励 −β(log π − log π_ref) 约为 (−0.0091, +0.0144, −0.0059, 0)，最后一个再加 4.5。第 2 个 token 上 π < π_ref，KL 惩罚变成了正的奖励（6.2 节说的 k1 可以为负）。

```python
import numpy as np

# (a) 三步回答的 GAE 手算：γ = 1，奖励只在最后一步，第 3 步后终止
r = np.array([0.0, 0.0, 1.0])
V = np.array([0.2, 0.4, 0.6])                    # V(s_0), V(s_1), V(s_2)；V(s_3) = 0
delta = r + np.append(V[1:], 0.0) - V            # [0.2, 0.2, 0.4]
assert np.allclose(delta, [0.2, 0.2, 0.4])
def gae(lam):
    A, last = np.zeros(3), 0.0
    for t in reversed(range(3)):
        last = delta[t] + lam * last
        A[t] = last
    return A
assert np.allclose(gae(0.0), [0.2, 0.2, 0.4])    # 只看一步
assert np.allclose(gae(1.0), [0.8, 0.6, 0.4])    # = 1 − V(s_t)
assert np.allclose(gae(0.5), [0.4, 0.4, 0.4])

# (b) PPO clip 手算：ε = 0.2，π_old(a) = 0.5
eps = 0.2
def obj(p_new, A, p_old=0.5):
    r = p_new / p_old
    return min(r * A, np.clip(r, 1 - eps, 1 + eps) * A)
assert np.isclose(obj(0.65, +1.0), 1.2)          # r = 1.3：目标封顶在 1.2，不再随 π 增长
assert np.isclose(obj(0.70, +1.0), 1.2)
assert np.isclose(obj(0.30, +1.0), 0.6)          # r = 0.6：取未截断项 0.6，梯度照常
assert np.isclose(obj(0.30, -1.0), -0.8)         # A<0、r = 0.6：封顶在 −0.8
assert np.isclose(obj(0.65, -1.0), -1.3)         # A<0、r = 1.3：取未截断项，罚得更重

# (c) RLHF 的逐 token KL 奖励：每个 token 给 −β·(log π − log π_ref)，最后一个 token 再加 RM 分数
#     逐 token 求和 = 序列级的 R(x, y) = r(x, y) − β·log(π(y|x)/π_ref(y|x))
beta, rm = 0.05, 4.5
logp = np.log([0.6, 0.3, 0.9, 0.5])              # π 在 4 个回答 token 上的概率
logp_ref = np.log([0.5, 0.4, 0.8, 0.5])
tok_r = -beta * (logp - logp_ref)
tok_r[-1] += rm
assert np.isclose(tok_r.sum(), rm - beta * (logp.sum() - logp_ref.sum()))
assert tok_r[1] > 0                              # π < π_ref 的 token 上，KL 惩罚变成正奖励
print("逐 token 奖励：", tok_r.round(4))
```

## 10 工程细节与常见坑

每条按“现象 → 原因 → 排查 → 处理”写。PPO 训练建议至少监控：平均奖励、策略熵、对旧策略的近似 KL、clip 比例、对参考模型的 KL、value loss、explained variance（critic 解释了回报方差的多少比例）、优势的均值和方差、回答长度。GRPO 系特有的坑（长度偏差、零优势组、训推 log 概率不一致、MoE）见 [training/03 第 8 节](../../tracks/training/03-grpo-family/README.md#8-工程细节与常见坑)。

**1. 截断当成终止，或终止当成截断**

- 现象：回答长度被莫名压短；或者 critic 在回合开头的估值异常高。
- 原因：4.4 节。截断处没有 bootstrap，critic 学到“写到上限就没价值”；或者终止处漏标 done，下一条轨迹的价值被传进来。
- 排查：构造一条只在末尾有奖励的轨迹，分别按终止和截断算 GAE，对照 4.5 节代码第 5、6 个检查的结果。检查 rollout buffer 里回合边界的标志是否和环境的 terminated、truncated 一一对应。
- 处理：终止和截断分两个标志存；截断处用 V(s_{t+1}) bootstrap。LLM 里，长度上限截断的回答另外决定奖励怎么给（4.4 节）。

**2. clip 写成了只 clip 不取 min**

- 现象：策略在某些样本上越走越偏，clip 比例很高但 KL 仍在涨。
- 原因：loss 写成 `clip(ratio, 1−ε, 1+ε) * A`，漏了 `min`。Â > 0 而 ρ 已低于 1−ε 时，梯度被清零，错误方向纠正不回来（5.8 节最后一个检查）。
- 排查：构造 ρ = 0.5、Â = 1 的 token，看梯度是否为 0。
- 处理：按式 7 写 min。实现成 loss 时注意符号：要最小化的是 −min(ρÂ, clip(ρ)Â)，常见写法是 `max(−ρÂ, −clip(ρ)Â)`。

**3. 比率算错位置或用错 log 概率**

- 现象：第一个 minibatch 的比率不是 1，clip 比例在第一步就不为 0。
- 原因：logits 的第 i 个位置预测的是第 i+1 个 token，取 log 概率时错了一位；或者 π_old 用的是推理引擎返回的值，和训练引擎算的不一致；或者 dropout 没关。
- 排查：同一份参数、同一批数据，算两次 log 概率，比较 ρ 和 1 的差距。
- 处理：π_old 一律用训练引擎在 rollout 后重算；训练时关闭 dropout（Ziegler 等在策略训练中不用 dropout）。

**4. 多 epoch 后 KL 飙升，奖励突然下跌**

- 现象：每轮更新后的近似 KL 远超预期，clip 比例接近 1，随后奖励断崖式下降。
- 原因：5.4 节。clip 不是硬约束，学习率大或 epoch 多时，比率会走得很远。
- 排查：画出每个 minibatch 后的近似 KL 和 clip 比例，看是在第几个 epoch 开始失控。
- 处理：降学习率、减少 epoch；加 KL 早停（如 Spinning Up 的 1.5 × target_kl）。

**5. critic 不准导致优势有系统偏差**

- 现象：explained variance 接近 0 甚至为负；优势和 token 位置强相关（越靠前越正或越负）；回答长度单向漂移。
- 原因：critic 从奖励模型初始化，对中间 token 的估值没有意义（4.6 节）；或者 λ < 1 时奖励信号衰减，critic 的目标里混着自己的错误预测（4.5 节）。
- 排查：画优势随 token 位置的分布；看 value loss 和 explained variance 是否在下降。
- 处理：先预训练 critic；critic 的目标用 λ = 1；或者 critic 先训几步再开始更新策略。实在训不好，考虑去掉 critic，用组内基线。

**6. 优势没有标准化，或在错误的范围标准化**

- 现象：换一个奖励模型或奖励尺度，同样的学习率就训崩或学不动。
- 原因：梯度大小和优势尺度成正比。
- 排查：打印每个 batch 的优势均值和标准差。
- 处理：在 batch 内标准化优势（只在有效 token 上算均值和标准差，padding 不参与）。Huang 等第 24 条细节还提到可选的奖励白化（reward whitening）。注意标准化减均值会改变“好”和“坏”的分界，和 GRPO 的组内标准化不是一回事。

**7. KL 惩罚系数设置不当**

- 现象：β 太大，奖励几乎不涨，策略贴着参考模型；β 太小，奖励涨得很快，但输出开始重复、跑题，人工看质量下降（reward hacking）。
- 原因：β 决定了“奖励”和“离参考模型的距离”之间的换算率。
- 排查：同时画奖励和对参考模型的 KL；人工抽样看高奖励样本。
- 处理：先用固定 β 扫几个值，再考虑用 Ziegler 式的控制器让 KL 贴近目标值。

**8. 熵塌缩**

- 现象：策略熵在早期快速下降，同一 prompt 的多次采样几乎一样，奖励停在平台。
- 原因：从 SFT 出发的策略本来就集中，clip 的上界又按比例限制低概率 token 的提升。
- 排查：看熵曲线；分别统计被上界和下界截断的比例。
- 处理：经典 RL 里加熵奖励（c2）；LLM 里更常见的是 DAPO 的 Clip-Higher，见 [training/03 第 4.1 节](../../tracks/training/03-grpo-family/README.md#41-clip-higher为什么只调大上界)。

**9. padding 和非生成 token 漏 mask**

- 现象：loss 随 padding 长度变化；KL 在 π = π_ref 时不为 0；Agent 场景里模型开始“模仿”工具结果的格式。
- 原因：GAE、loss、KL、优势标准化有任何一处没乘 response mask。
- 排查：令 π_θ = π_ref，KL 必须为 0；改 padding 位置的 log 概率，loss 必须不变。
- 处理：所有逐 token 的量用同一个 mask；GAE 从每条回答最后一个有效 token 开始递推。

## 11 对比与取舍

### 11.1 Ψ_t 和优势估计

| 方法 | 优势估计 | 偏差 | 方差 | 需要 critic | 适用 |
|---|---|---|---|---|---|
| REINFORCE | G_t | 无 | 高 | 否 | 短回合、教学 |
| REINFORCE + 基线 | G_t − b(s_t) | 无 | 中 | b 可以是常数或均值 | bandit、序列级奖励 |
| RLOO | R_i − 其余 k−1 个样本的均值 | 无 | 中 | 否，但每个 prompt 采 k 条 | LLM，结果奖励 |
| Actor-Critic（一步 TD） | δ_t | V 不准时有偏 | 低 | 是 | 奖励稠密、回合很长 |
| GAE(λ) | Σ(γλ)^l δ_{t+l} | 随 λ 减小而增大 | 随 λ 减小而减小 | 是 | PPO 的默认 |

### 11.2 策略更新方式

| 方法 | 怎么限制步长 | 数据复用 | 优化器 | 代价 |
|---|---|---|---|---|
| 普通策略梯度 | 只靠学习率 | 一批一次 | 一阶 | 样本效率低 |
| TRPO | 硬 KL 约束 Ê[KL] ≤ δ | 一批一次 | 共轭梯度 + 线搜索 | 实现复杂，二阶信息 |
| PPO-clip | 截断比率，悲观下界 | 多 epoch | 一阶 | 不是硬约束（5.4 节） |
| PPO-KL | 自适应 KL 惩罚 | 多 epoch | 一阶 | 论文实验里不如 clip |

### 11.3 LLM 上用哪个

- **奖励稠密，或需要逐 token 的信用分配**，显存够放 critic：PPO + GAE。有中间奖励（多步工具调用、过程奖励）时，critic 的价值更大。
- **结果奖励、可验证任务、显存紧**：去掉 critic，用组内基线（RLOO、GRPO 系），见 [training/03 第 9 节](../../tracks/training/03-grpo-family/README.md#9-对比与取舍)。
- **长 CoT 上坚持用 PPO**：critic 要预训练，GAE 要解耦（critic 用 λ = 1），参考 VC-PPO 和 VAPO。

## 12 自测清单

- [ ] 把单轮生成和 Agent 多轮各写成 MDP，说出状态、动作、奖励、转移
- [ ] 从 ∇E[f] 出发写出对数导数技巧，说明环境转移为什么在求导时消失
- [ ] 证明 t 之前的奖励乘 ∇log π(a_t|s_t) 期望为 0，从而只保留 reward-to-go
- [ ] 证明减去只依赖状态的基线不改变梯度期望，推出方差最优的常数基线
- [ ] 写出 TD 残差，证明 V = V^π 时它是优势的无偏估计
- [ ] 从 k 步估计的加权平均推出 GAE，说出 λ = 0 和 λ = 1 各对应什么，以及为什么 λ = 1 对 V 的误差免疫
- [ ] 说出终止和截断在 δ_t 和递推上的区别
- [ ] 写出 PPO 的 clip 目标，证明它是 L^CPI 的下界，按 Â 的正负说出哪些梯度为 0
- [ ] 说明为什么 clip 不是硬信任域，实践中用什么补救
- [ ] 写出 PPO 的完整 loss 和自适应 KL 规则
- [ ] 说出 clip、对旧策略的 KL、对参考模型的 KL 的区别，写出逐 token KL 奖励
- [ ] 写出 k1、k2、k3 三种 KL 估计量，说出哪些无偏
- [ ] 说出 LLM RL 和经典 RL 的至少 4 点差别，以及为什么可以去掉 critic
- [ ] 不看代码，手算第 9 节的两张表
- [ ] 手写 PPO clip loss、价值 loss 和 GAE（见 [coding/README.md](../../coding/README.md) 的 code-model-009、code-model-010）

## 资料

- Schulman 等，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，arXiv 1707.06347，2017-07。式 6 到式 9，悲观下界见第 3 节，自适应 KL 见第 4 节，算法 1，超参数见表 1、表 3、表 5
- Schulman 等，[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)，arXiv 1506.02438，2015-06。Ψ_t 的六种形式见第 2 节式 1，GAE 推导见式 11 到式 18
- Schulman 等，[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)，arXiv 1502.05477，2015-02。代理目标 L_π 和一阶近似见第 2 节式 3、式 4，下界见定理 1
- Ziegler 等，[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)，arXiv 1909.08593，2019-09。KL 奖励见式 2，β 控制器见第 2.2 节
- Engstrom 等，[Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729)，arXiv 2005.12729，ICLR 2020。代码级优化见第 3 节，比率走出区间见第 4 节图 2
- Pardo 等，[Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378)，arXiv 1712.00378，ICML 2018。截断处 bootstrap
- Huang 等，[The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031)，arXiv 2403.17031，2024-03。超参数见表 7，逐 token 奖励见图 9，EOS trick 见第 23 条细节
- Yuan 等，[What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret](https://arxiv.org/abs/2503.01491)（VC-PPO），arXiv 2503.01491，2025-03。Value-Pretraining 见第 3.2 节，Decoupled-GAE 见第 3.3 节
- Yue 等，[VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)，arXiv 2504.05118，2025-04。Length-Adaptive GAE 见第 4.2 节式 5
- Ahmadian 等，[Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)，arXiv 2402.14740，2024-02
- John Schulman，[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)，博客，2020-03-07
- OpenAI，[Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)，文档，2026-11 访问。KL 早停
- Sutton、Barto，[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)，第 2 版，MIT Press，2018，第 13 章（13.2 策略梯度定理，13.3 REINFORCE，13.4 带基线的 REINFORCE，13.5 Actor-Critic）
