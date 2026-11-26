# 量化：权重、激活与 KV 量化

> 层级：垂类 · inference · 预计 200 分钟 · 题目：[questions.md](questions.md)
> 前置：[01 KV Cache 与显存估算](../01-kv-cache-memory/README.md)（显存构成、算术强度、Decode 的带宽瓶颈）

## 1 这一章解决什么问题

量化是 Agent 岗面试里推理部分的第二高频话题。面试官想确认三件事：你知道量化把什么换成了什么；你分得清 GPTQ、AWQ、FP8 这些名字不在同一层；你知道量化对 Agent 最敏感的能力（工具调用参数、长链推理）有什么影响，以及上线前怎么验证。

最朴素的做法是逐元素四舍五入到整数网格（RTN，round-to-nearest）。8 位时它对多数权重够用，但有两个缺陷。第一，激活里有系统性的离群值，一个大值就让整组的刻度变粗，8 位也会坏。第二，到 4 位、3 位，每个权重独立舍入的误差会在层输出上叠加，RTN 明显掉点。后面的方法都在补这两个缺口：SmoothQuant 和 LLM.int8() 处理离群值，GPTQ 和 AWQ 处理低位宽权重，QAT 让模型在训练时适应误差，KIVI 和 KVQuant 把同样的思路用到 KV Cache 上。

学完你应该能：

- 写出对称和非对称量化的公式，推出舍入误差的上界和期望，说清粒度怎么影响误差和存储开销；
- 解释激活为什么比权重难量化，推导 SmoothQuant 的平滑因子，说出 α = 0.5 平衡了什么；
- 写出 GPTQ 的逐列量化和误差补偿公式，手写一个简化实现，并说出 AWQ 为什么只放大显著通道；
- 写出伪量化和直通估计，知道 QAT 什么时候值得做；
- 讲清 KIVI 为什么 Key 按通道、Value 按 token，KVQuant 为什么在 RoPE 之前量化 Key；
- 用算术强度说明 W4A16 为什么加速 Decode、未必加速 Prefill，算出一个 72B 模型量化后的权重和 KV 预算；
- 说出量化对推理和 Agent 任务的已知影响，给出上线前的评测清单。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| x、x̂ | 原始浮点值、反量化后的值；误差 e = x − x̂ |
| b | 位宽，INT4 时 b = 4 |
| q | 量化后的整数 |
| s | 缩放因子（scale，也叫步长），相邻两个刻度之间的距离 |
| z | 零点（zero-point），浮点 0 对应的整数 |
| g | 分组大小：每 g 个连续元素共用一组 s、z |
| W | 线性层权重。GPTQ 记号下形状为 (d_row, d_col)，输出 = W X |
| X | 层输入（激活）。GPTQ 记号下形状为 (d_col, m)，m 为校准 token 数；SmoothQuant 记号下为 (T, C_in) |
| H | 层重构目标的 Hessian，H = 2XXᵀ |
| N | 参数量 |
| WxAy | 权重 x 位、激活 y 位，例如 W4A16、W8A8 |
## 2 量化的数学：对称、非对称与误差

### 2.1 两种映射

量化把浮点数 x 映射到 b 位整数 q，配一个缩放因子 s（有时还有零点 z），用时再反量化回 x̂。

```text
对称（零点固定为 0）：
  s = max|x| / (2^(b−1) − 1)          INT8 时分母是 127，INT4 时是 7
  q = clamp(round(x / s), −2^(b−1), 2^(b−1) − 1)
  x̂ = s · q

非对称（用 min 和 max 定范围）：
  s = (max(x) − min(x)) / (2^b − 1)    INT4 时分母是 15
  z = round(−min(x) / s)
  q = clamp(round(x / s) + z, 0, 2^b − 1)
  x̂ = s · (q − z)
```

两者的区别只在“刻度覆盖哪一段”。对称量化覆盖 [−max|x|, max|x|]，非对称覆盖 [min, max]。如果 x 全是正数（比如 ReLU 之后的激活），对称量化的负半边刻度一个都用不上，相当于少了 1 位。

不同论文和库对分母的写法不完全一样。AWQ 论文式 (1) 写的是 Δ = max|w| / 2^(N−1)，比上面的 2^(N−1) − 1 多一格，最大值会被截断到 2^(N−1) − 1 那一格。跨工具转换权重时，要先对齐这个定义。

| | 对称 | 非对称 |
|---|---|---|
| 适合 | 分布大致以 0 为中心，如多数权重 | 分布偏向一侧，如某些激活、KV |
| 精度 | 分布偏时浪费一半刻度 | 刻度用满，同位宽误差更小 |
| 开销 | 只存 s，计算简单 | 多存 z，反量化多一次减法 |

### 2.2 误差的上界和期望

记误差 e = x − x̂。只要 x 落在量化范围内（没有被 clamp 截断），round 最多差半格：

```text
|e| ≤ s / 2
```

再假设 x / s 的小数部分在 [0, 1) 上近似均匀（x 的分布相对步长足够平滑时成立），e 就近似服从 [−s/2, s/2] 上的均匀分布：

```text
E[e] = 0
E[e²] = ∫_{−s/2}^{s/2} e² · (1/s) de = s² / 12
```

这一个式子推出三条有用的结论。

1. **每多 1 位，均方误差降到约 1/4。** 范围不变时 s 和 2^b 成反比，s² 和 4^b 成反比。换成信噪比就是每位约 6 dB。
2. **误差由组里最大的那个数决定。** 对称量化的 s = max|x| / (2^(b−1) − 1)。组里只要有一个离群值，s 就被拉大，组里所有元素的期望误差一起变大。这就是第 3 节的离群值问题。
3. **截断换精度。** 把范围缩到 [−c, c]（c < max|x|）能让 s 变小，代价是 |x| > c 的元素误差变成 |x| − c，不再受 s/2 约束。AWQ 的权重裁剪、GPTQ 实现里的网格搜索都在这两项之间找平衡。LLM-QAT 论文则发现截断激活里的离群值对 LLM 有害：训练初期困惑度会超过 10000，所以他们改用不截断的对称 MinMax 量化。

### 2.3 量化粒度

同一组 s 覆盖的范围越小，越能贴合局部分布，精度越好，但要存的 s 越多。

| 粒度 | 一个 s 覆盖 | 常用于 | 代价 |
|---|---|---|---|
| per-tensor | 整个张量 | FP8 权重和激活、FP8 KV 的默认做法 | 最省，最怕离群值 |
| per-channel | 权重的一行或一列 | INT8 权重 | 每个输出通道一个 s |
| per-group | 连续 g 个权重（常见 g=128） | INT4 权重（GPTQ、AWQ） | 每个权重多 16/g 位（FP16 scale） |
| per-token | 激活或 KV 的一个 token 向量 | 动态激活量化、Value cache | 要在线算 s |

为什么粒度越细误差越小，可以从 2.2 节推出来。把一组拆成几个子组，每个子组的 max|x| 不会超过原来整组的 max|x|，所以每个元素的步长只会变小或不变。按 E[e²] = s²/12，期望误差也只会变小或不变。离群值越集中，收益越大：离群值只拉大它所在子组的 s，其他子组不受影响。

存储代价按“每个权重多几位”算：每组存一个 FP16 scale，多 16/g 位；再存一个 FP16 零点，再多 16/g 位。g=128 时合计 0.25 位，g=32 时合计 1 位。

### 2.4 手算一遍：4 个数的 INT4 量化

取 x = [−0.42, 0.13, 0.33, 1.4]，b = 4。

```text
对称：  s = 1.4 / 7 = 0.2
        q = round([−2.1, 0.65, 1.65, 7]) = [−2, 1, 2, 7]
        x̂ = [−0.4, 0.2, 0.4, 1.4]                       MSE = 0.00255

非对称：s = (1.4 − (−0.42)) / 15 = 0.12133
        z = round(0.42 / 0.12133) = round(3.46) = 3
        q = round(x / s) + 3 = [0, 4, 6, 15]
        x̂ = s · (q − 3) = [−0.364, 0.121, 0.364, 1.456]   MSE = 0.00188
```

非对称的步长小了约 40%，MSE 也更小。代价是零点取整之后，原来的最小值 −0.42 不再正好落在格点上。注意 x̂ 的最大值 1.456 超过了原始的 1.4：零点取整让整张网格平移了一点。

```python
import numpy as np

x = np.array([-0.42, 0.13, 0.33, 1.4])
b = 4

# 对称：s = max|x| / 7
s = np.abs(x).max() / (2 ** (b - 1) - 1)
q = np.clip(np.round(x / s), -8, 7)
x_sym = s * q
assert np.isclose(s, 0.2) and q.tolist() == [-2, 1, 2, 7]
assert np.allclose(x_sym, [-0.4, 0.2, 0.4, 1.4])

# 非对称：s = (max − min) / 15，z = round(−min / s)
s_a = (x.max() - x.min()) / (2 ** b - 1)
z = np.round(-x.min() / s_a)
q_a = np.clip(np.round(x / s_a) + z, 0, 15)
x_asym = s_a * (q_a - z)
assert np.isclose(s_a, 0.121333, atol=1e-6) and z == 3 and q_a.tolist() == [0, 4, 6, 15]

mse_sym = np.mean((x - x_sym) ** 2)
mse_asym = np.mean((x - x_asym) ** 2)
assert np.isclose(mse_sym, 0.00255) and abs(mse_asym - 0.00188) < 1e-5
# 两种方法的误差都没超过半个步长
assert np.all(np.abs(x - x_sym) <= s / 2 + 1e-12) and np.all(np.abs(x - x_asym) <= s_a / 2 + 1e-12)
print("对称 x̂ =", x_sym.round(4), " 非对称 x̂ =", x_asym.round(4))
print(f"MSE 对称 {mse_sym:.5f}  非对称 {mse_asym:.5f}")
```

### 2.5 实现：误差公式和粒度的验证

下面的代码验证 2.2 和 2.3 节的结论：误差上界 s/2、期望 s²/12、每位约 4 倍；偏态分布上非对称更好；粒度越细误差越小。

```python
import numpy as np

rng = np.random.default_rng(0)

def quant_sym(x, b, axis=None):
    """对称量化：零点为 0，s = max|x| / (2^(b-1) - 1)。axis 指定沿哪一维求 max（共用一个 s）。"""
    qmax = 2 ** (b - 1) - 1
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    s = np.where(amax == 0, 1.0, amax / qmax)
    q = np.clip(np.round(x / s), -qmax - 1, qmax)
    return q * s, s

def quant_asym(x, b, axis=None):
    """非对称量化：用 [min, max] 定范围，s = (max - min) / (2^b - 1)，z = round(-min / s)。"""
    qmax = 2 ** b - 1
    lo = np.min(x, axis=axis, keepdims=True)
    hi = np.max(x, axis=axis, keepdims=True)
    s = np.where(hi == lo, 1.0, (hi - lo) / qmax)
    z = np.round(-lo / s)
    q = np.clip(np.round(x / s) + z, 0, qmax)
    return s * (q - z), s

def quant_group(W, b, g, fn=quant_sym):
    """per-group：W 形状 (d_out, d_in)，沿输入维每 g 个连续元素一组。"""
    d_out, d_in = W.shape
    Wq, s = fn(W.reshape(d_out, d_in // g, g), b, axis=-1)
    return Wq.reshape(d_out, d_in), s

mse = lambda a, b: float(np.mean((a - b) ** 2))

# 1) 误差上界和期望：不截断时 |e| ≤ s/2，近似均匀分布时 E[e²] ≈ s²/12
x = rng.normal(size=200_000)
for fn in (quant_sym, quant_asym):
    xq, s = fn(x, 8)
    s = float(s.squeeze())
    assert np.max(np.abs(x - xq)) <= s / 2 + 1e-12
    assert abs(mse(x, xq) / (s ** 2 / 12) - 1) < 0.05

# 2) 每多 1 bit，步长减半，MSE 约降到 1/4（约 6 dB）
u = rng.uniform(-1, 1, size=200_000)
errs = [mse(u, quant_sym(u, b)[0]) for b in range(4, 9)]
ratios = np.array(errs[:-1]) / np.array(errs[1:])
assert np.all((ratios > 3.5) & (ratios < 4.6)), ratios

# 3) 偏向一侧的分布（如 SiLU/ReLU 之后的激活）：对称量化浪费一半刻度
a = np.abs(rng.normal(size=100_000))
r_one_side = mse(a, quant_sym(a, 4)[0]) / mse(a, quant_asym(a, 4)[0])
# 以 0 为中心的分布：两者差不多
w = rng.normal(size=100_000)
r_centered = mse(w, quant_sym(w, 4)[0]) / mse(w, quant_asym(w, 4)[0])
assert r_one_side > 3.5          # 步长比约 (15/7)² ≈ 4.6
assert 0.7 < r_centered < 1.3

# 4) 粒度：每行尺度不同、个别元素偏大的权重，粒度越细误差越小
d_out, d_in = 256, 1024
W = rng.normal(size=(d_out, d_in)) * np.exp(rng.normal(0, 0.5, size=(d_out, 1)))
W[rng.random(W.shape) < 0.001] *= 8                      # 0.1% 的离群权重
e_tensor = mse(W, quant_sym(W, 4)[0])
e_channel = mse(W, quant_sym(W, 4, axis=1)[0])          # 每个输出通道（每行）一个 s
e_g128 = mse(W, quant_group(W, 4, 128)[0])
e_g32 = mse(W, quant_group(W, 4, 32)[0])
assert e_tensor > e_channel > e_g128 > e_g32
print(f"INT4 MSE  per-tensor {e_tensor:.4f}  per-channel {e_channel:.4f}  "
      f"g128 {e_g128:.4f}  g32 {e_g32:.4f}")
print(f"偏态分布 sym/asym = {r_one_side:.2f}，居中分布 sym/asym = {r_centered:.2f}")
```

运行结果（INT4 MSE）：per-tensor 1.27，per-channel 0.18，g128 0.045，g32 0.021。这组权重每行的尺度不同，还混了 0.1% 的离群元素。per-tensor 让所有行共用最大的那个尺度，per-channel 解决了行间差异，per-group 又把离群元素隔离在各自的 128 个元素里。偏态分布上对称和非对称的 MSE 比是 4.58，接近步长比的平方 (15/7)² ≈ 4.6；居中分布上是 1.19。

## 3 离群值：为什么激活比权重难量化

### 3.1 LLM.int8() 的观察

权重的分布平坦、接近以 0 为中心，per-channel INT8 甚至 INT4 都容易量化。激活不一样。LLM.int8() 论文分析了 OPT 系列的隐藏状态，结论是：

- 幅度比其他维度大到 20 倍的特征，先出现在约 25% 的层里，模型变大后扩散到更多层。
- 约 6.7B 参数时出现“相变”：所有层、75% 的序列位置都受这类大值影响。
- 离群值是系统性的。6.7B 模型每条序列约有 15 万个离群值，但只集中在整个模型的 6 个特征维度上。
- 这些维度不能丢。把它们置零，注意力 softmax 的 top-1 概率质量下降超过 20%，验证困惑度变差 600% 到 1000%；随机置零同样多的维度，困惑度只变差约 0.1%。

LLM.int8() 的做法是**混合精度分解**：幅度超过阈值 α = 6.0 的特征维度拆出来走 16 位矩阵乘，其余 99.9% 以上的值走 8 位矩阵乘，最后相加。13B 以内的模型离群维度不超过 7 个，额外显存约 0.1%。代价是要在线找离群维度、拆矩阵，GPTQ 论文指出它的运行时间和 FP16 基线相同，省显存不省时间。

### 3.2 为什么 per-token 救不了

激活量化的粒度受矩阵乘 kernel 限制。Y = X W，X 的形状是 (T, C_in)，W 是 (C_in, C_out)。INT8 GEMM 在整数累加器里沿 C_in 求和，缩放因子只能放在**外维**上，累加完一次乘回去：

```text
Y = diag(s_X) · (X̄_INT8 · W̄_INT8) · diag(s_W)
     ↑ 每个 token 一个                 ↑ 每个输出通道一个
```

所以激活只能 per-tensor 或 per-token，权重只能 per-tensor 或 per-output-channel。离群值偏偏是**沿输入通道**（C_in，内维）分布的：某几个通道在每个 token 上都大。per-token 量化时，每个 token 那一行里都有离群通道，每行的 s 都被拉大，等于没解决问题。按输入通道量化激活能解决，但 s 落在内维上，没法在累加之后乘回去。

SmoothQuant 给了一个直观的量：通道 i 的最大幅度是 m_i，整个矩阵的最大值是 m，INT8 per-tensor 量化下通道 i 只有约 2^8 · m_i / m 个有效刻度。离群值比普通值大约 100 倍时，普通通道只剩 2 到 3 个刻度。

### 3.3 SmoothQuant：把难度从激活搬到权重

**等价变换。** 对每个输入通道 j 取一个平滑因子 s_j > 0，激活除以它，权重对应的行乘以它：

```text
Y = (X · diag(s)⁻¹) · (diag(s) · W) = X̂ · Ŵ
```

数学上完全等价。X̂ 的第 j 列是 X_j / s_j，Ŵ 的第 j 行是 s_j · W_j。X 通常来自前一个 LayerNorm 或线性层，diag(s)⁻¹ 可以离线并进前一层的参数里，运行时没有额外开销。

**怎么选 s。** 目标是让 X̂ 和 Ŵ 都好量化。

- s_j = max|X_j|：X̂ 的每个通道最大值都是 1，激活变得非常好量化。但离群值整个搬到了 W 的对应行，权重变得难量化。
- s_j = 1 / max|W_j|：反过来，难度全留在激活上。

SmoothQuant 用迁移强度 α 在两者之间插值：

```text
s_j = max|X_j|^α / max|W_j|^(1−α)
```

代进去看平滑后第 j 个通道两侧的最大值：

```text
max|X̂_j| = max|X_j| / s_j = max|X_j|^(1−α) · max|W_j|^(1−α)
max|Ŵ_j| = s_j · max|W_j| = max|X_j|^α · max|W_j|^α
```

α = 0.5 时两者相等，都是 √(max|X_j| · max|W_j|)。激活和权重在这个通道上分担同样的“难度”。论文发现 OPT、BLOOM 等多数模型上 α = 0.5 是很好的平衡点；激活离群更严重的模型（论文提到 GLM-130B 约 30% 是离群值）用更大的 α，如 0.75；Llama-2 的结果表里用的是 0.85。α 由在 Pile 验证集子集上的快速网格搜索决定。

**效果。** 论文报告 SmoothQuant 让 W8A8 在 OPT-175B、BLOOM-176B、GLM-130B、Llama-2 等模型上基本无损，相比 FP16 最多 1.56 倍加速、显存减半。它是“权重和激活都量化”这一类方法的代表，加速来自 INT8 矩阵乘单元，所以 Prefill 也能变快（见第 4 节）。

### 3.4 实现：离群通道和平滑

构造 8 个离群通道（每个 token 上都放大 60 倍）的激活，对比几种 W8A8 粒度，再做 SmoothQuant 平滑。

```python
import numpy as np

rng = np.random.default_rng(1)

def quant_sym(x, b, axis=None):
    qmax = 2 ** (b - 1) - 1
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    s = np.where(amax == 0, 1.0, amax / qmax)
    return np.clip(np.round(x / s), -qmax - 1, qmax) * s

def rel_err(Y, Yq):
    return float(np.linalg.norm(Y - Yq) / np.linalg.norm(Y))

# 激活 X: (T, C_in)，T 个 token、C_in 个输入通道。权重 W: (C_in, C_out)，Y = X W
T, C_in, C_out = 512, 1024, 1024
X = rng.normal(size=(T, C_in))
outlier = rng.choice(C_in, size=8, replace=False)        # 8 个固定离群通道，约 0.8%
X[:, outlier] *= 60                                      # 每个 token 上都大约 60 倍（LLM.int8() 的“系统性”离群）
W = rng.normal(size=(C_in, C_out)) / np.sqrt(C_in)       # 权重平坦，好量化
Y = X @ W

# 1) 离群值让普通通道只剩几个有效刻度：通道 i 的有效刻度数 ≈ 2^8 · m_i / m（SmoothQuant 第 3 节）
m_i = np.abs(X).max(axis=0)
levels = 2 ** 8 * m_i / m_i.max()
normal = np.setdiff1d(np.arange(C_in), outlier)
assert np.median(levels[normal]) < 10

# 2) W8A8，硬件友好的粒度：激活 per-tensor 或 per-token，权重 per-output-channel
def w8a8(X, W, act_axis):
    return quant_sym(X, 8, axis=act_axis) @ quant_sym(W, 8, axis=0)

e_tensor = rel_err(Y, w8a8(X, W, None))
e_token = rel_err(Y, w8a8(X, W, 1))                      # 每个 token 一个 s：离群通道在每个 token 里都有，救不了
# 对照：per-input-channel 激活量化误差很小，但 s 落在矩阵乘的内维，INT8 GEMM 没法在累加后再乘回去
e_chan = rel_err(Y, w8a8(X, W, 0))
assert e_chan < e_token / 3

# 3) SmoothQuant：s_j = max|X_j|^α / max|W_j|^(1-α)，X̂ = X diag(s)^-1，Ŵ = diag(s) W，X̂Ŵ = XW
def smooth(X, W, alpha):
    s = np.abs(X).max(axis=0) ** alpha / np.abs(W).max(axis=1) ** (1 - alpha)
    return X / s, W * s[:, None]

Xs, Ws = smooth(X, W, 0.5)
assert np.allclose(Xs @ Ws, Y)                           # 数学等价，只改了“难度”放在哪边
# α = 0.5 时，第 j 个通道在激活侧和权重侧的最大值相等，都是 sqrt(max|X_j| · max|W_j|)
assert np.allclose(np.abs(Xs).max(axis=0), np.abs(Ws).max(axis=1))
e_smooth = rel_err(Y, w8a8(Xs, Ws, 1))
assert e_smooth < e_token / 3

# 4) α 的两端都不好：α=0 难度全在激活上，α=1 全推给权重（权重出现离群行）
errs = {a: rel_err(Y, w8a8(*smooth(X, W, a), 1)) for a in (0.0, 0.25, 0.5, 0.75, 1.0)}
best = min(errs, key=errs.get)
assert 0 < best < 1
print(f"per-tensor {e_tensor:.4f}  per-token {e_token:.4f}  per-in-channel(不可用) {e_chan:.4f}")
print("SmoothQuant + per-token，按 α:", {a: round(e, 4) for a, e in errs.items()})
```

运行结果（输出相对误差）：per-tensor 0.101，per-token 0.048，per-input-channel 0.011。per-token 只改善一半，因为离群通道在每个 token 里都有。SmoothQuant 加 per-token：α = 0 时 0.051（就是不平滑），α = 0.5 时 0.0105，α = 1 时 0.050。α = 0.5 平滑后的误差和“不可用的 per-input-channel”相当，而它只用了硬件支持的粒度。两端都差，因为难度只是换了一边。

## 4 量化什么，在哪个阶段加速

### 4.1 三类方案

| 方案 | 例子 | 省什么 | 加速在哪里 |
|---|---|---|---|
| 只量化权重 | W4A16、W8A16 | 权重显存、读权重的带宽 | 小 batch Decode（带宽瓶颈）。矩阵乘仍是 16 位，要先反量化 |
| 权重和激活都量化 | W8A8（INT8 或 FP8） | 权重显存，并用 8 位矩阵乘单元 | Prefill 和大 batch Decode（算力瓶颈）也能加速 |
| KV 量化 | FP8 KV、INT4/2bit KV | KV 显存、读 KV 的带宽 | 长上下文和高并发 |

FP8 是格式不是算法。FP8 Formats 论文定义了两种编码：E4M3（4 位指数、3 位尾数，最大正规数 448，范围窄精度高）和 E5M2（5 位指数、2 位尾数，最大正规数 57344，范围宽）。浮点格式自带指数，刻度不均匀，对离群值比 INT8 宽容，但需要硬件原生支持 FP8 矩阵乘才有计算加速。

### 4.2 用算术强度推一遍

算术强度 I 的定义和 H100 的“算力 / 带宽”比值（约 295 FLOP/B）见 [inference/01 第 2.3 节](../01-kv-cache-memory/README.md#23-每步读写多少字节为什么-decode-受带宽限制)。这里只看线性层：一步同时处理 b 个 token（Decode 时是 batch 里的序列数，Prefill 时是 prompt 长度），权重每个元素占 w 位。

```text
计算     F = 2 · b · N
读权重   N · w / 8 字节          （激活的读写相对很小，忽略）
强度     I = 2b / (w / 8) = 16b / w
```

BF16 时 w = 16，I = b。W4A16 时 w ≈ 4.25（含分组开销），I ≈ 3.76b。推出三条结论：

1. **小 batch Decode 加速比约等于字节比。** b = 1 时 I 远低于 295，一步的时间几乎全是读权重。读的字节降到 4.25/16，时间也降到约 27%，即约 3.76 倍。
2. **W4A16 更早撞上算力墙。** W4A16 的矩阵乘仍以 BF16 算力执行。I = 295 时对应 b = 295 · 4.25 / 16 ≈ 78。超过这个 batch，再省字节也没用。
3. **Prefill 通常不加速。** prompt 2000 个 token 时 b = 2000，BF16 和 W4A16 都受算力限制，时间一样。真实 kernel 里反量化是额外计算，W4A16 还可能更慢。要让算力受限的阶段变快，需要 W8A8：H100 的 INT8/FP8 稠密算力约为 BF16 的 2 倍。

AWQ 论文的瓶颈分析给出了同样的图景：在 RTX 4090 上用 batch 1 跑 Llama-2-7B，生成阶段是带宽受限的，W4A16 把算术强度提高了约 4 倍。GPTQ 论文报告，3 位 OPT-175B 在单张 A100 上的每 token 延迟是 FP16（5 张卡）的约 1/3.25，几乎全部来自它的“量化矩阵 × 全精度向量”kernel。

长上下文 Agent 请求还有第二个瓶颈：每步要读全部 KV。权重量化不减少这部分字节，见 4.3 节最后一段和第 7 节。

### 4.3 实现：权重体积、加速比和留给 KV 的显存

用 Qwen2.5-72B-Instruct 的 config 把账算一遍。这个模型 80 层、64 个查询头、8 个 KV 头，适合说明“权重压下去之后 KV 能放多少”。

```python
import json

# 原样摘自 Qwen2.5-72B-Instruct 的 config.json（2026-11 查阅），只保留估算要用的字段
cfg = json.loads("""{"hidden_size": 8192, "intermediate_size": 29568, "num_attention_heads": 64,
  "num_hidden_layers": 80, "num_key_value_heads": 8, "vocab_size": 152064, "tie_word_embeddings": false}""")
d, f, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
n_h, n_kv, V = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
d_h = d // n_h

# 1) 数参数：量化的是每层的线性层；嵌入和输出层常保留 16 位
attn = d * n_h * d_h + 2 * d * n_kv * d_h + n_h * d_h * d       # Q、K、V、O 投影
mlp = 3 * d * f                                                  # gate、up、down
small = (n_h * d_h + 2 * n_kv * d_h) + 2 * d                     # QKV 偏置、两个 RMSNorm
N_linear = L * (attn + mlp)
N_embed = V * d * (1 if cfg["tie_word_embeddings"] else 2)
N_other = L * small + d
N = N_linear + N_embed + N_other
assert abs(N / 1e9 - 72.7) < 0.1                                 # 模型卡：72.7B

# 2) 每个权重的等效位宽：b 位整数 + 每组 g 个共享一个 scale（和可选的 zero）
def eff_bits(b, g=None, scale_bits=16, zero_bits=0):
    return b if g is None else b + (scale_bits + zero_bits) / g

bf16 = N * 2
w4 = N_linear * eff_bits(4, 128, 16, 16) / 8 + (N_embed + N_other) * 2   # INT4 g128，FP16 scale + FP16 zero
w4_naive = N * 4 / 8                                             # 只按 4 bit 算，漏掉组开销和 16 位的嵌入
assert eff_bits(4, 128, 16, 16) == 4.25 and eff_bits(4, 32, 16, 16) == 5.0
assert 42e9 < w4 < 42.5e9 and 36e9 < w4_naive < 36.5e9          # 组开销 +2.2 GB，16 位嵌入和输出层再 +3.7 GB
print(f"BF16 {bf16/1e9:.1f} GB  INT4 g128 {w4/1e9:.1f} GB（其中 16 位嵌入 {N_embed*2/1e9:.1f} GB）  "
      f"只按 4 bit 算 {w4_naive/1e9:.1f} GB")

# 3) 什么时候能加速：把一层线性层看成 (b × d_in) @ (d_in × d_out)，b 是一步同时处理的 token 数
#    FLOPs = 2·b·N_linear；读权重 = N_linear·bits/8（激活字节相对很小，忽略）
#    算术强度 I = 2b / (bits/8)。H100 SXM：BF16 稠密 ≈ 990 TFLOPS，带宽 3.35 TB/s，比值 ≈ 295（inference/02 第 4 节）
peak, bw = 990e12, 3.35e12
ridge = peak / bw
def step_time(b, bits):
    """忽略 KV 和反量化，只看线性层：max(算的时间, 读权重的时间)。W4A16 的矩阵乘仍以 BF16 算力计。"""
    return max(2 * b * N_linear / peak, N_linear * bits / 8 / bw)

for b in (1, 16):                                               # 小 batch Decode：受带宽限制，读得少就快
    sp = step_time(b, 16) / step_time(b, 4.25)
    assert abs(sp - 16 / 4.25) < 1e-9                            # 加速比 = 字节比 ≈ 3.76
b_ridge_w4 = ridge * 4.25 / 16                                   # W4A16 在 b ≈ 78 时就碰到算力墙
assert 75 < b_ridge_w4 < 82
assert step_time(2000, 4.25) == step_time(2000, 16)              # Prefill 2000 token：两者都受算力限制，不加速
# W8A8（INT8/FP8 矩阵乘单元算力约为 BF16 的 2 倍）才能让算力受限的阶段变快
assert step_time(2000, 16) / (2 * 2000 * N_linear / (2 * peak)) == 2
print(f"Decode b=1：BF16 {step_time(1,16)*1e3:.1f} ms → W4A16 {step_time(1,4.25)*1e3:.1f} ms；"
      f"W4A16 在 b≈{b_ridge_w4:.0f} 转为算力受限")

# 4) 省下的显存给 KV：80 GB 卡，可用 90%，预留 4 GB 给激活和运行时（假设值）
budget = 80e9 * 0.9 - 4e9
kv_per_token = 2 * L * n_kv * d_h * 2                            # K 和 V，每层，BF16：327,680 B = 320 KiB
assert kv_per_token == 327_680
assert bf16 > budget and N_linear * 1 + (N_embed + N_other) * 2 > budget   # BF16 和 W8A16 都放不下
kv_tokens = {name: int((budget - w4) // (kv_per_token * s)) for name, s in (("BF16", 1), ("FP8", 0.5))}
sessions = {k: v // 32_768 for k, v in kv_tokens.items()}
assert sessions == {"BF16": 2, "FP8": 4}                         # 32K 上下文的会话：FP8 KV 让并发翻倍
print("W4A16 后可放的 KV token：", kv_tokens, " 32K 会话数：", sessions)
```

运行结果：

- 参数 72.7B，和模型卡一致。其中线性层 70.2B，嵌入和输出层 2.5B（`tie_word_embeddings` 为 false，各一份）。
- BF16 权重 145.4 GB。INT4 g128（FP16 scale 和 zero）42.3 GB；如果只按 4 位乘总参数，得 36.4 GB，少算了约 6 GB：组开销 2.2 GB，16 位的嵌入和输出层 3.7 GB。
- Decode b = 1 时，BF16 每步读权重约 41.9 ms，W4A16 约 11.1 ms（只算线性层、只算读权重的下界）。
- 一张 80 GB 卡，可用 90%、预留 4 GB：BF16 和 INT8 权重都放不下；W4A16 后剩约 25.7 GB 给 KV。每 token KV 是 320 KiB（BF16），能放约 7.8 万 token，即 2 条 32K 会话。KV 改成 FP8 后翻倍到 4 条。

最后一条说明了 4.1 节表格的第三行：权重压到 4 位之后，限制并发的是 KV。

## 5 PTQ：GPTQ 与 AWQ

PTQ（训练后量化）不改训练，只用少量校准数据。最朴素的做法是逐组四舍五入（RTN），低位宽时误差大。GPTQ 和 AWQ 都是在 RTN 基础上减小误差的只量化权重方法。两者都把问题拆成逐层的子问题：对一个线性层，找量化后的 Ŵ，使层输出的误差最小。

```text
argmin_Ŵ ‖W X − Ŵ X‖²₂        （GPTQ 式 (1)，量化网格事先固定）
```

### 5.1 GPTQ：二阶信息和误差补偿

**目标函数是二次型。** 上式可以按 W 的行拆开，每一行 w（形状 1 × d_col）独立：

```text
‖(w − ŵ) X‖² = (w − ŵ) · X Xᵀ · (w − ŵ)ᵀ = ½ (w − ŵ) H (w − ŵ)ᵀ,     H = 2 X Xᵀ
```

H 的形状是 (d_col, d_col)，只取决于层输入 X，和权重无关，所以**所有行共用一个 H**。H 的非对角元表示输入通道之间的相关性：两个输入通道相关时，一个权重的误差可以由另一个权重的调整抵消掉一部分。RTN 完全忽略这一点。

**OBQ 的单步更新。** GPTQ 建立在 OBQ（Optimal Brain Quantization）上。OBQ 一次量化一个权重 w_q，然后调整还没量化的权重（集合 F）来补偿。对二次型求带约束的最小值，得到（GPTQ 式 (2)）：

```text
δ_F = − (w_q − quant(w_q)) / [H_F⁻¹]_qq · (H_F⁻¹)_{:,q}

按 (w − ŵ) H (w − ŵ)ᵀ 计，量化这一个权重的误差增量 = (w_q − quant(w_q))² / [H_F⁻¹]_qq
换成层输出误差 ‖(w − ŵ) X‖²，再乘 ½
```

推导思路：固定 w_q 必须变成 quant(w_q)，其余权重自由，这是一个带一个线性约束的二次规划。用拉格朗日乘子解出来，最优调整方向是 H⁻¹ 的第 q 列。量化完 w_q，把它从 F 里去掉，H_F⁻¹ 要相应删掉第 q 行第 q 列。这一步不用重新求逆，一步高斯消元即可（式 (3)）：

```text
H_{−q}⁻¹ = ( H⁻¹ − H⁻¹_{:,q} H⁻¹_{q,:} / [H⁻¹]_qq )_{−q}
```

OBQ 按贪心顺序挑下一个误差最小的权重，每行顺序不同，对 d_row × d_col 的矩阵复杂度是 O(d_row · d_col³)，到 1 亿参数就跑不动了。

**GPTQ 的三个改动。**

1. **固定列顺序。** 论文发现，大模型上贪心顺序比任意固定顺序好不了多少。所有行都按同一个列顺序量化，那么每一步的 F 对所有行相同，H_F⁻¹ 也相同，只需要更新 d_col 次而不是 d_row · d_col 次。复杂度降到 O(max{d_row · d_col², d_col³})。
2. **延迟批量更新。** 逐列更新整个矩阵，每个元素只做几次浮点运算，受带宽限制。第 i 列的舍入只受作用在它自己身上的更新影响，所以可以一次处理 B = 128 列：块内逐列量化、逐列补偿，块外的列攒到块结束时用一次矩阵乘更新。理论计算量不变，论文报告实际快了一个数量级。
3. **Cholesky 重写。** 大模型上反复做式 (3) 会累积数值误差，H_F⁻¹ 可能变得不定，把剩余权重往错的方向推。观察到每一步只需要 H_F⁻¹ 第 q 行从对角元往右的部分，这正好是 H⁻¹ 的 Cholesky 分解的上三角因子的第 q 行（差一个除以 √[H_F⁻¹]_qq 的常数）。所以开始前做一次 Cholesky，就拿到了全部需要的信息。另外在 H 的对角上加阻尼 λ，取平均对角元的 1%。

完整算法（Algorithm 1）：

```text
H⁻¹ ← Cholesky((2XXᵀ + λI)⁻¹)ᵀ                       上三角
for i = 0, B, 2B, ...:                                 每个块
    for j = i, ..., i+B−1:                             块内逐列
        Q[:, j] ← quant(W[:, j])
        E[:, j−i] ← (W[:, j] − Q[:, j]) / H⁻¹[j, j]     这一列的误差（已除以对角元）
        W[:, j:i+B] ← W[:, j:i+B] − E[:, j−i] · H⁻¹[j, j:i+B]      补偿块内剩下的列
    W[:, i+B:] ← W[:, i+B:] − E · H⁻¹[i:i+B, i+B:]     块结束，一次性补偿后面所有列
```

论文的设置和结果：校准数据是 C4 里随机 128 段、每段 2048 token 的文本；逐个 Transformer 块量化，每块的输入用“已经量化完的前面各块”的输出算，误差会被后面的块看到。论文报告一张 A100 约 4 个 GPU 小时量化 175B 模型到 3 或 4 位；4 位时 OPT-175B 和 BLOOM-176B 的困惑度只比全精度高不超过 0.25，3 位时 RTN 崩溃而 GPTQ 仍可用。分组能进一步改善精度：g1024（约多 0.02 位）平均改善约 0.2，g128（约多 0.15 位）再改善约 0.1。

### 5.2 实现：简化 GPTQ 对比 RTN

在一个 64 × 128 的小矩阵上实现 Algorithm 1，3 位、每行一个 scale。输入通道之间有相关性，这是误差补偿能起作用的前提。

```python
import numpy as np

rng = np.random.default_rng(2)

# 层：输出 = W X。W: (d_row, d_col)，X: (d_col, m)，m 个校准 token。GPTQ 的记号
d_row, d_col, m, bits = 64, 128, 2048, 3
A = rng.normal(size=(d_col, d_col)) / np.sqrt(d_col)
X = A @ rng.normal(size=(d_col, m)) + 0.3 * rng.normal(size=(d_col, m))   # 输入通道之间相关，更接近真实激活
W = rng.normal(size=(d_row, d_col))

# 量化网格在开始前定好：每行一个对称 scale（per-output-channel）
qmax = 2 ** (bits - 1) - 1
scale = np.abs(W).max(axis=1, keepdims=True) / qmax
def quant(w, s):
    return np.clip(np.round(w / s), -qmax - 1, qmax) * s

def out_err(Wq):
    return float(np.sum((W @ X - Wq @ X) ** 2))

# RTN：逐个元素四舍五入，互不相干
W_rtn = quant(W, scale)

def gptq(W, X, B=32, damp=0.01):
    """GPTQ Algorithm 1：固定列顺序 + 块内递归 + 块外延迟批量更新 + Cholesky 形式的 H^-1。"""
    W = W.copy()
    H = 2 * X @ X.T                                       # (d_col, d_col)，只和输入有关，所有行共用
    H += damp * np.mean(np.diag(H)) * np.eye(len(H))      # 阻尼 λ = 1% 平均对角元
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T         # 上三角 U，H^-1 = UᵀU
    Q = np.zeros_like(W)
    for i in range(0, W.shape[1], B):
        E = np.zeros((W.shape[0], B))
        for j in range(i, i + B):
            Q[:, j] = quant(W[:, j:j + 1], scale)[:, 0]
            E[:, j - i] = (W[:, j] - Q[:, j]) / Hinv[j, j]
            W[:, j:i + B] -= np.outer(E[:, j - i], Hinv[j, j:i + B])   # 只更新块内
        W[:, i + B:] -= E @ Hinv[i:i + B, i + B:]                     # 块结束后一次性更新后面所有列
    return Q

W_gptq = gptq(W, X)

def obq_fixed_order(W, X, damp=0.01):
    """不用 Cholesky 的写法：每量化一列，用式 (2) 补偿剩余列，再用式 (3) 从 H^-1 里消去这一列。"""
    W = W.copy()
    H = 2 * X @ X.T
    H += damp * np.mean(np.diag(H)) * np.eye(len(H))
    Hinv = np.linalg.inv(H)
    Q = np.zeros_like(W)
    for j in range(W.shape[1]):
        Q[:, j] = quant(W[:, j:j + 1], scale)[:, 0]
        err = (W[:, j] - Q[:, j]) / Hinv[j, j]
        W[:, j:] -= np.outer(err, Hinv[j, j:])                       # 式 (2)：δF = −(w_q − quant(w_q)) / [H^-1]_qq · H^-1_{:,q}
        Hinv = Hinv - np.outer(Hinv[:, j], Hinv[j, :]) / Hinv[j, j]   # 式 (3)：一步高斯消元
    return Q

# Cholesky 形式和逐列做式 (2)(3) 结果相同：上三角因子的第 j 行就是消元到第 j 步时 H^-1 的第 j 行（差一个常数）
assert np.allclose(W_gptq, obq_fixed_order(W, X))
# 分块只是改了更新的时机，不改数学：B=1 和 B=32 结果相同
assert np.allclose(W_gptq, gptq(W, X, B=1))
# 两者用的是同一张网格，GPTQ 只改了“每个权重舍入到哪一格”
assert np.allclose(np.round(W_gptq / scale), W_gptq / scale)
r = out_err(W_gptq) / out_err(W_rtn)
assert r < 0.7                                            # 这组数据上约 0.61
# 权重本身离原值反而更远：GPTQ 优化的是层输出，不是权重的 MSE
assert np.sum((W - W_gptq) ** 2) > np.sum((W - W_rtn) ** 2)

# OBQ 单步公式核对：只量化第 0 列、其余列按 δF 补偿后，输出误差的增量 = (w_q − quant(w_q))² / [H_F^-1]_qq
H = 2 * X @ X.T
Hi = np.linalg.inv(H)
w = W[0].copy()
q0 = quant(w[:1], scale[0])[0]
delta = -(w[0] - q0) / Hi[0, 0] * Hi[:, 0]               # 式 (2) 的 δF（含第 0 列本身，正好把它移到 q0）
w_new = w + delta
assert np.isclose(w_new[0], q0)
loss = lambda v: float((v - W[0]) @ H @ (v - W[0])) / 2   # ‖(v−w)X‖² = (v−w) H (v−w)ᵀ / 2
assert np.isclose(loss(w_new), (w[0] - q0) ** 2 / Hi[0, 0] / 2)
print(f"3-bit 输出误差 GPTQ / RTN = {r:.3f}")
```

运行结果：GPTQ 的层输出平方误差是 RTN 的 0.61 倍。几个断言各验证一件事：

- Cholesky 版本和逐列做式 (2)(3) 的结果完全相同，说明 Cholesky 只是换了一种数值更稳的算法，没有改数学。
- 块大小 B = 1 和 B = 32 结果相同，说明延迟批量更新只改了更新时机。
- GPTQ 的权重本身离原值**更远**（权重 MSE 更大），但层输出更准。它优化的是输出，不是权重。
- OBQ 单步的误差增量和式 (2) 给出的值一致。

输入通道越相关，补偿的空间越大。把代码里 X 的独立噪声从 0.3 调到 1.0（通道近似独立），比值升到约 0.87；调到 0.1，比值降到约 0.51。

**工程上的两个补充。** GPTQ 官方仓库后来加了 `--act-order`：按 H 对角元（对应输入通道的激活能量）从大到小的顺序量化列，让重要的列先量化、有更多剩余列帮它补偿。README 说这修复了 LLaMA-7B 上异常差的结果（Wiki2 困惑度 7.15 到 6.09）。act-order 和分组一起用时，组的划分跟着重排后的列走，推理时要按置换读 scale，可能变慢；`--static-groups` 在量化前按原始列顺序定好所有组的网格，避免这个问题。

### 5.3 AWQ：按激活保护显著通道

**观察。** AWQ 论文在 OPT-6.7B 上做 INT3-g128 实验：只把约 1% 的权重保留为 FP16，困惑度就从 RTN 的 43.2 降到 13.0。关键是怎么选这 1%：按**激活**幅度选（对应输入通道的平均 |x| 大）有效，按权重自身的大小选几乎没用。原因是输出误差 = 权重误差 × 激活，激活大的通道把权重误差放大得最多。

**混合精度不好用，改成等价缩放。** 1% FP16 要求 kernel 处理两种数据类型，硬件不友好。AWQ 改为把显著通道的权重乘以 s > 1，对应的激活除以 s：

```text
y = Q(w · s) · (x / s)
```

为什么能减误差。论文的推导（式 (1) 到 (3)）：

```text
原误差   Err(Q(w) x)          = Δ  · RoundErr(w / Δ) · x
缩放后   Err(Q(w·s) (x/s))    = Δ' · RoundErr(w·s / Δ') · x · (1/s)
```

其中 Δ 是组的步长，由组内最大值决定。论文给出三个经验事实：RoundErr 近似均匀分布在 [0, 0.5]，期望约 0.25，和 s 无关；只放大组里一个元素通常不改变组的最大值，所以 Δ' ≈ Δ；Δ 和 x 用 FP16 表示，本身没有量化误差。于是新旧误差之比约为 (Δ'/Δ) · (1/s)，显著权重的误差降到约 1/s。

**s 不能无限大。** s 变大后，越来越多组的最大值变成了被放大的显著权重，Δ' > Δ，同组的普通权重误差跟着变大。论文 Table 2：s = 2 时只有 8.2% 的组 Δ 变了，困惑度最好（23.54 降到 11.92）；s = 4 时 21.2% 的组 Δ 变了，困惑度反而回升到 12.36。

**搜索缩放因子。** 显著和非显著通道要一起考虑。AWQ 的目标是缩放后的层输出误差最小：

```text
s* = argmin_s ‖Q(W · diag(s)) (diag(s)⁻¹ · X) − W X‖
```

Q 不可导，论文说基于近似梯度的方法收敛不稳定，所以把搜索空间缩成一个标量：

```text
s = s_X^α,    α* = argmin_α L(s_X^α),    α ∈ [0, 1]
```

s_X 是每个输入通道的平均激活幅度。α = 0 就是不缩放（RTN），α = 1 是搜索空间里最激进的缩放。官方实现（`auto_scale.py`）在 α ∈ {0, 0.05, ..., 0.95} 这 20 个点上搜索，并把 s 除以 √(max(s) · min(s)) 做归一化。搜完再对权重做裁剪，进一步减小 MSE。diag(s)⁻¹ 并进前一个算子（LayerNorm 或前一个线性层），运行时没有额外开销。

**为什么对校准集不敏感。** AWQ 只用校准集算每个通道的平均幅度，不做反向传播，也不像 GPTQ 那样逐列重构。论文认为因此对校准集过拟合更少，在指令微调模型和多模态模型上泛化更好。

### 5.4 实现：AWQ 式通道缩放

构造 1% 输入通道激活特别大的层，INT4-g128，对比按激活和按权重保留 1% FP16、手动放大显著通道、按 α 网格搜索。

```python
import numpy as np

rng = np.random.default_rng(3)

def quant_group(W, b=4, g=128):
    """per-group 对称量化，W: (d_out, d_in)，沿输入维每 g 个一组。AWQ 论文式 (1)：Δ = max|w| / 2^(N-1)。"""
    d_out, d_in = W.shape
    Wg = W.reshape(d_out, d_in // g, g)
    delta = np.abs(Wg).max(axis=-1, keepdims=True) / 2 ** (b - 1)
    q = np.clip(np.round(Wg / delta), -2 ** (b - 1), 2 ** (b - 1) - 1)
    return (q * delta).reshape(d_out, d_in)

# 层：Y = W X。W: (d_out, d_in)，X: (d_in, m)
d_out, d_in, m = 256, 1024, 512
W = rng.normal(size=(d_out, d_in))
act_scale = np.exp(rng.normal(0, 0.3, size=d_in))
salient = rng.choice(d_in, size=10, replace=False)       # 约 1% 的输入通道激活特别大
act_scale[salient] *= 30
X = rng.normal(size=(d_in, m)) * act_scale[:, None]
Y = W @ X
err = lambda Wq, Xin=X: float(np.linalg.norm(W @ X - Wq @ Xin) / np.linalg.norm(Y))

e_rtn = err(quant_group(W))

# 1) 按激活找的 1% 通道保留 FP16，比按权重大小找的有效得多（AWQ Table 1 的结论）
def keep_fp16(idx):
    Wq = quant_group(W)
    Wq[:, idx] = W[:, idx]
    return err(Wq)
k = d_in // 100
by_act = np.argsort(-np.abs(X).mean(axis=1))[:k]
by_w = np.argsort(-np.linalg.norm(W, axis=0))[:k]
assert keep_fp16(by_act) < 0.7 * keep_fp16(by_w)

# 2) AWQ：s = s_X^α，W' = W diag(s)，X' = diag(s)^-1 X，W'X' = WX；α 在 [0,1] 上网格搜索
s_X = np.abs(X).mean(axis=1)                             # 每个输入通道的平均激活幅度
def awq_err(alpha):
    s = s_X ** alpha
    s = s / np.sqrt(s.max() * s.min())                   # 归一化，避免整体缩放带偏（不改相对大小）
    return err(quant_group(W * s), X / s[:, None])

grid = np.arange(20) / 20                                # 官方实现 n_grid = 20：0, 0.05, …, 0.95
errs = np.array([awq_err(a) for a in grid])
best = grid[errs.argmin()]
assert errs[0] == e_rtn or np.isclose(errs[0], e_rtn)     # α=0 就是 RTN
assert errs.min() < 0.9 * e_rtn
assert 0 < best < 0.5
assert errs[-1] > errs.min()                              # α 太大：过度放大，非显著通道的误差上升

# 3) 只放大 1% 显著通道 s 倍（AWQ Table 2 的实验）：s 小时大多数组的 Δ 不变，显著权重的相对误差约降到 1/s；
#    s 太大时显著权重撑大了组的 Δ，同组的普通权重误差上升
def deltas(Wm):
    return np.abs(Wm.reshape(d_out, -1, 128)).max(axis=-1)
res = {}
for s_val in (1.0, 1.25, 1.5, 2.0, 4.0, 8.0):
    s = np.ones(d_in); s[salient] = s_val
    changed = float(np.mean(~np.isclose(deltas(W * s), deltas(W))))
    res[s_val] = (changed, err(quant_group(W * s), X / s[:, None]))
    print(f"s={s_val}: Δ 变化的组占比 {changed:.1%}  输出误差 {res[s_val][1]:.4f}")
assert res[1.0][0] == 0 and res[2.0][0] < res[8.0][0]    # s 越大，越多组的 Δ 被改动
assert min(res[1.25][1], res[1.5][1], res[2.0][1]) < res[1.0][1]
assert res[8.0][1] > min(r[1] for r in res.values())     # 放大过头，误差回升

print(f"RTN {e_rtn:.4f}  1% FP16(按激活) {keep_fp16(by_act):.4f}  1% FP16(按权重) {keep_fp16(by_w):.4f}")
print(f"AWQ 最优 α={best:.2f}  误差 {errs.min():.4f}")
```

运行结果（输出相对误差）：RTN 0.1006；按激活选 1% 保留 FP16 是 0.029，按权重选是 0.1006，和 RTN 几乎一样。手动放大显著通道：s = 1.5 最好（0.085），s = 8 时 65% 的组 Δ 被改动，误差升到 0.121，比 RTN 还差。这和论文 Table 2 的趋势一致：先降后升。α 网格搜索的最优点是 0.10，误差 0.086。

这个例子里 1% FP16 比 AWQ 缩放好得多，而论文的 OPT 实验中两者接近（Table 3）。原因是这里的显著通道激活放大了 30 倍，比真实模型极端，缩放能保护的有限。玩具实验只用来验证机制，不用来比方法的优劣。

### 5.5 GPTQ 和 AWQ 对比

| | GPTQ | AWQ |
|---|---|---|
| 核心思路 | 逐列量化 + 误差补偿 | 按激活找显著通道 + 等价缩放 |
| 用到的信息 | 二阶（H = 2XXᵀ，含通道相关性） | 每个通道的平均激活幅度 |
| 改变了什么 | 每个权重舍入到哪一格，以及未量化权重的值 | 权重的缩放（量化前），舍入本身仍是 RTN |
| 对校准集 | 做重构，有过拟合风险 | 只统计幅度，依赖更弱 |
| 计算量 | 每层一次 Cholesky，O(d_col³) | 每层 20 次前向搜索 |
| 常见配置 | W4A16 / W3A16，分组 | W4A16，g=128 |

两者关注的是同一个目标的不同部分。AWQ 只决定量化前怎么缩放，GPTQ 只决定怎么舍入，所以原理上可以先缩放再用 GPTQ 舍入。是否值得这样做，要在自己的任务上测。

## 6 QAT：训练时就考虑量化

### 6.1 伪量化和直通估计

**问题**：PTQ 在 8 位上效果好，到 4 位及以下误差明显变大。PTQ 只能在“已经训练好的权重附近”找舍入方式，模型本身没有机会适应量化误差。

**做法**：训练（或继续训练）时在前向里插入伪量化（fake quant）：先量化再反量化，输出仍是浮点，但已经带上了量化误差。

```text
前向：ŵ = s · clamp(round(w / s), q_min, q_max)       用 ŵ 算 loss
反向：∂L/∂w = ∂L/∂ŵ · ∂ŵ/∂w
```

麻烦在 ∂ŵ/∂w。round 是阶梯函数，导数几乎处处为 0，在台阶处不存在。直接求导，梯度全为 0，参数一步也走不动。

直通估计（STE，straight-through estimator）把 round 的导数当成 1：

```text
∂ŵ/∂w ≈ 1    当 w / s 在 [q_min − 0.5, q_max + 0.5] 内
∂ŵ/∂w ≈ 0    在截断范围外（clamp 的导数本来就是 0）
```

名字出自 Bengio 等 2013 年的论文：反向传播时把这个不可导的单元当作恒等函数“直接穿过”。直观上，模型维护一份浮点的“潜在权重” w，梯度按量化后的 ŵ 算，但更新加在 w 上。w 慢慢移动，跨过格点边界时 ŵ 跳到相邻的一格。

### 6.2 实现：STE 在线性回归上的效果和振荡

一个 64 维的线性回归，部署时权重用 2 位网格。对比 PTQ（直接舍入最优解）和 QAT（从最优解出发，用 STE 继续训练）。

```python
import numpy as np

rng = np.random.default_rng(5)

# 任务：线性回归 y = X w*，X: (n, d)。部署时 w 要用 2 bit 对称网格，scale 固定
n, d, bits = 4096, 64, 2
C = rng.normal(size=(d, d)) / np.sqrt(d)
X = rng.normal(size=(n, d)) @ C + 0.5 * rng.normal(size=(n, d))   # 特征相关
w_star = rng.normal(size=d)
y = X @ w_star
qmin, qmax = -2 ** (bits - 1), 2 ** (bits - 1) - 1
scale = np.abs(w_star).max() / qmax

def fake_quant(w):
    """伪量化：前向里先量化再反量化，输出仍是浮点。"""
    return np.clip(np.round(w / scale), qmin, qmax) * scale

loss = lambda w: float(np.mean((X @ w - y) ** 2))

# 1) round 的导数几乎处处为 0：直接对伪量化求梯度，参数一步也走不动
w0 = w_star + 0.01 * rng.normal(size=d)
eps = 1e-6
fd = np.array([(loss(fake_quant(w0 + eps * e)) - loss(fake_quant(w0 - eps * e))) / (2 * eps)
               for e in np.eye(d)])
assert np.count_nonzero(fd) == 0

# 2) PTQ：先训练全精度权重（这里直接取最优解 w*），再四舍五入
loss_ptq = loss(fake_quant(w_star))

# 3) QAT + STE：前向用 fake_quant(w)，反向把 ∂round/∂w 当成 1（落在截断范围外时为 0）
w = w_star.copy()                                         # 从全精度权重开始继续训练
lr, steps = 3e-3, 2000
hist, grids = [], []
for step in range(steps):
    wq = fake_quant(w)
    g_wq = 2 * X.T @ (X @ wq - y) / n                     # 对伪量化权重的梯度
    inside = (w / scale >= qmin - 0.5) & (w / scale <= qmax + 0.5)
    w = w - lr * g_wq * inside                            # STE：梯度原样穿过 round
    hist.append(loss(fake_quant(w)))
    grids.append(np.round(w / scale))
hist = np.array(hist)
best = int(hist.argmin())
assert hist[best] < 0.85 * loss_ptq                       # 训练中最好的检查点比 RTN 好（真实训练里用验证集选）

# 4) 常见坑：权重振荡。潜在权重卡在两个格点的分界附近，STE 梯度把它来回推过边界，
#    训练末期部署 loss 仍大幅跳动，“最后一步”的权重不一定是好的
tail = hist[-200:]
flips = int(np.sum(np.diff(np.array(grids[-200:]), axis=0) != 0))
assert flips > 100 and tail.max() > 2 * tail.min()
print(f"2-bit 部署 MSE：PTQ(RTN) {loss_ptq:.1f}  QAT 最佳检查点 {hist[best]:.1f}（第 {best} 步）  "
      f"最后一步 {hist[-1]:.1f}")
print(f"最后 200 步：格点翻转 {flips} 次，部署 MSE 在 {tail.min():.1f} 到 {tail.max():.1f} 之间跳")
```

运行结果：

- 对伪量化直接做数值求导，64 个分量全是 0，验证了 round 不可导的问题。
- PTQ 部署后 MSE 73.7；QAT 训练中最好的检查点 59.0，在第 40 步。
- 训练末期权重还在两个格点之间来回跳：最后 200 步格点翻转 4631 次，部署 MSE 在 90 到 369 之间变化，最后一步是 196.6，比 PTQ 还差。

最后一条是 QAT 的典型问题，叫**权重振荡**。Nagel 等（2022）专门研究过：潜在权重停在两个格点的分界附近，STE 梯度把它推过边界，ŵ 跳格后梯度反向，又推回来。他们发现这在 4 位及以下的低位宽 QAT 里尤其明显，并提出振荡阻尼和迭代冻结权重两种处理。工程上更简单的对策是学习率衰减、按验证集选检查点，而不是取最后一步。

### 6.3 LLM 上的 QAT

- LLM-QAT（Liu 等，2023-05）：用预训练模型自己生成的数据做蒸馏，不需要原始训练数据；同时量化权重、激活和 KV Cache，最低到 4 位，论文报告低位宽时明显好于免训练方法。它用对称 MinMax 量化（不截断离群值），权重 per-channel，激活和 KV per-token。
- 模型方直接发布 QAT 版本：例如 gpt-oss 的模型卡写明，MoE 权重在后训练阶段就用 MXFP4 量化，所有评测也在 MXFP4 下完成，因此 120b 模型能放进单张 80 GB GPU。

| | PTQ | QAT |
|---|---|---|
| 成本 | 分钟到小时，少量校准数据 | 要训练算力和数据 |
| 低位宽精度 | 4 位以下下降明显 | 更好 |
| 谁来做 | 部署方自己做 | 通常由模型方做；部署方直接用官方量化权重 |

Agent 团队的实际选择：优先用模型方发布的官方量化版本；没有时用 PTQ；只有在 4 位以下仍有硬需求时才考虑自己做 QAT。

## 7 KV Cache 量化

### 7.1 问题和难点

长上下文和高并发时，KV 是显存大头（第 4.3 节的例子里，W4A16 之后只能放 2 条 32K 会话）。每个元素从 2 字节降到 1 字节或更低，并发就能翻倍。

KV 量化和权重量化有两点不同：

1. **KV 是流式产生的。** 每步 Decode 新增一个 token 的 K、V。GPTQ 这类需要整块数据做优化的方法用不上，量化必须轻量、在线完成。
2. **K 和 V 的用法不同。** K 用来和 q 点积算注意力分数，V 被注意力权重加权求和。误差的影响路径不同，最好的量化方向也不同。

### 7.2 KIVI：Key 按通道，Value 按 token

记一个头的缓存 K、V 形状为 (T, d_h)。“按 token”是每个 token 的 d_h 个通道分组共用 s、z；“按通道”是每个通道沿 token 方向每 G 个分组。

**Key 按通道。** KIVI 观察到 Key 里有固定的离群通道，某几个通道在所有 token 上都大，这和 LLM.int8() 观察到的激活离群通道一致。按 token 量化时，每个 token 那一行都包含离群通道，s 被拉大，普通通道的精度全丢了（和 3.2 节 per-token 激活量化失效是同一个原因）。按通道量化把离群值关在自己的组里。论文 Table 2 报告，per-token 量化 Key 的注意力分数相对误差约是 per-channel 的 5 倍。

**Value 按 token。** Value 没有明显的通道离群模式，单看重建误差，两种分法差不多。KIVI 的解释看注意力输出：

```text
[A X_V]_i = Σ_j A_ij · [X_V]_j        第 i 行输出是各 token 的 Value 按注意力权重加权
```

注意力很稀疏，输出主要由少数几个重要 token 的 Value 决定。按 token 量化把误差限制在每个 token 自己的向量里，其他 token 的量化不会影响重要 token 的精度。论文报告按 token 量化 Value 时，注意力输出的相对误差比按通道小近 15 倍。

**流式实现。** 按 token 量化天然适合流式：新 token 来了直接量化、追加。按通道量化要攒够 G 个 token 才能组成一组。KIVI 把缓存分成两部分：已分组量化的部分，和最近 R 个 token 的全精度“残差”部分。新 token 先进残差；残差攒满 R 个就整组量化，并入量化部分。注意力分数分两段算再拼接：

```text
A_g = q · Q(X_Kg)ᵀ         量化部分，用融合反量化的矩阵乘
A_r = q · X_Krᵀ            残差部分，全精度
A = Concat([A_g, A_r])
```

论文设置：2 位非对称，组大小 G = 32，残差长度 R = 128（R 要能被 G 整除）。残差部分最多 R 个 token，序列远长于 R 时显存开销可以忽略。论文报告 KIVI 让峰值显存（含模型权重）降为 1/2.6，batch 最多增大 4 倍，吞吐提升 2.35 到 3.47 倍。

### 7.3 KVQuant：在 RoPE 之前量化 Key

KVQuant 也按通道量化 Key，另加三点：

- **RoPE 之前量化。** RoPE 按位置把通道两两旋转，不同位置的旋转角度不同。旋转后，原本固定在某些通道上的离群值被混到相邻通道里，而且混合程度随位置变化，通道的幅度不再一致。所以 KVQuant 缓存旋转前的 K，反量化之后再在线施加 RoPE（用融合 kernel）。论文报告这一项在 3 位 LLaMA-7B 上让 Wikitext-2 困惑度改善 0.82。
- **非均匀数据类型（nuqX）。** 刻度不等距，按每层的敏感度离线在校准集上求出。
- **按向量的稠密加稀疏分解。** 每个向量（Key 的每个通道、Value 的每个 token）用自己的离群阈值，超过阈值的约 1% 元素单独以稀疏格式全精度存储。

论文报告 3 位下困惑度上升小于 0.1，在单张 A100-80GB 上让 LLaMA-7B 支持最长 100 万上下文，8 卡支持 1000 万。

### 7.4 实现：Key 和 Value 的分组方向

构造带 4 个离群通道的 Key 和没有通道模式的 Value，2 位非对称，组大小 32。

```python
import numpy as np

rng = np.random.default_rng(4)

def quant_asym(x, b, axis):
    """非对称量化，沿 axis 求 min/max（同一组共用一个 s 和 z）。"""
    lo = x.min(axis=axis, keepdims=True)
    hi = x.max(axis=axis, keepdims=True)
    s = (hi - lo) / (2 ** b - 1)
    s = np.where(s == 0, 1.0, s)
    q = np.clip(np.round((x - lo) / s), 0, 2 ** b - 1)
    return q * s + lo

def per_token(x, b, G=32):
    """x: (T, d)。每个 token 内沿通道每 G 个一组。"""
    T, d = x.shape
    return quant_asym(x.reshape(T, d // G, G), b, axis=-1).reshape(T, d)

def per_channel(x, b, G=32):
    """x: (T, d)。每个通道内沿 token 每 G 个一组（KIVI 里 Key 的做法）。"""
    T, d = x.shape
    return quant_asym(x.reshape(T // G, G, d), b, axis=1).reshape(T, d)

rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(a))

# 一个头的缓存：T 个 token，head_dim d
T, d, bits = 1024, 128, 2
K = rng.normal(size=(T, d))
K_out = rng.choice(d, size=4, replace=False)
K[:, K_out] = K[:, K_out] * 2 + 12                       # Key：固定几个通道整体偏大（KIVI 图 2 的模式）
V = rng.normal(size=(T, d))                              # Value：没有通道模式
q = rng.normal(size=d) * 0.5
q[K_out] = 0.0                                           # 让打分主要由普通通道决定，便于看误差
q_scores = q @ K.T / np.sqrt(d)
A = np.exp(q_scores - q_scores.max()); A /= A.sum()      # 注意力权重 (T,)
O = A @ V

def scores_err(Kq):
    Aq = np.exp(q @ Kq.T / np.sqrt(d)); Aq /= Aq.sum()
    return rel(A, Aq)

# 1) Key：per-token 时离群通道撑大了每个 token 的范围；per-channel 把离群值关在自己的组里
eK_tok, eK_ch = scores_err(per_token(K, bits)), scores_err(per_channel(K, bits))
assert eK_ch < eK_tok / 2

# 2) Value：没有通道模式，两种分法的重建误差在同一量级。玩具数据复现不出 KIVI 表 2 里的 15 倍差距
#    （那来自真实模型的稀疏注意力），这里只验证“Value 不需要按通道”
eV_tok = rel(V, per_token(V, bits))
eV_ch = rel(V, per_channel(V, bits))
assert 0.5 < eV_tok / eV_ch < 2

# 3) 流式实现：最近 R 个 token 保持全精度，攒满再按 G 分组量化；结果和离线一次性分组一致
G, R = 32, 128
done, buf = [], []
for t in range(T):
    buf.append(K[t])
    if len(buf) == R:
        done.append(per_channel(np.array(buf), bits, G)); buf = []
K_stream = np.vstack(done + ([np.array(buf)] if buf else []))
assert np.allclose(K_stream, per_channel(K, bits, G))

# 4) 位宽账：2 bit，组大小 32，每组一个 FP16 scale 和一个 FP16 zero，等效 2 + 32/32 = 3 bit
eff_bits = bits + 2 * 16 / G
assert eff_bits == 3.0
print(f"Key 打分误差 per-token {eK_tok:.3f}  per-channel {eK_ch:.3f}")
print(f"Value 重建误差 per-token {eV_tok:.3f}  per-channel {eV_ch:.3f}  等效位宽 {eff_bits}")
```

运行结果：Key 的注意力分数相对误差 per-token 1.03，per-channel 0.20，差 5 倍左右。Value 的重建误差两种分法都是 0.39，验证了“Value 没有通道模式”。这组玩具数据复现不出论文 15 倍的输出误差差距，那需要真实模型的稀疏注意力和 Value 分布。流式实现和离线一次性分组的结果完全一致，因为 R = 128 是 G = 32 的整数倍。最后一行是位宽账：2 位加上每 32 个元素一组 FP16 scale 和 zero，等效 3 位，即名义上的“2 bit”实际只压到 BF16 的 3/16。

### 7.5 FP8 KV 和代价

工程上最常用的是 FP8 KV。vLLM 的 `--kv-cache-dtype fp8`（也可以指定 `fp8_e4m3` 或 `fp8_e5m2`）把 K、V 以 FP8 存储。除了下表里的要点，vLLM 文档还说明：Flash Attention 后端配合校准可以按注意力头缩放；校准推荐用 llm-compressor；敏感层用 `--kv-cache-dtype-skip-layers` 跳过。

| 方法 | 做法 | 要点 |
|---|---|---|
| FP8 KV（vLLM `--kv-cache-dtype fp8`） | K、V 以 FP8（E4M3 或 E5M2）存 | 默认 per-tensor 缩放；不做校准时缩放因子是 1.0，官方推荐用校准数据估计；文档提示滑动窗口等层对 KV 量化更敏感，可以跳过 |
| KIVI（2024-02） | 2 bit 非对称；Key 按通道量化，Value 按 token 量化 | 最近 R 个 token（论文 R=128）保持全精度，凑满一组再量化 |
| KVQuant（2024-01） | Key 按通道、并且在 RoPE 之前量化；非均匀数据类型；离群值单独存 | 论文报告 3 bit 下困惑度上升小于 0.1 |

**代价**：

- 解码时 attention kernel 要读低位 KV 并反量化，没有对应 kernel 就没有加速。
- MLA 这类结构缓存的是潜向量，不是多头 K、V，量化方案和 kernel 要专门适配。
- 误差会在长序列上累积，长上下文检索类任务（如大海捞针）要单独回归。

## 8 量化对 Agent 能力的影响

### 8.1 公开研究的结论

以下都是论文报告，换模型和任务不一定成立：

- Liu 等，Quantization Hurts Reasoning?（2025-04）：在 DeepSeek-R1 蒸馏系列、QwQ-32B、Qwen3-8B 上，W8A8 量化在各任务和尺寸上基本无损，W4A16 或 KV 量化接近无损（论文总结为下降不超过 1%，但 1.5B 小模型上超过 1%）；更低位宽风险明显。任务越难掉得越多，AIME-120 的下降最多是 GSM8K 的 4 倍。KV 降到 3 bit 时，1.5B 和 7B 模型下降超过 5%。轻微掉点的量化模型没有出现输出变长，激进的低位宽量化在小模型上会让输出变长。论文还发现，量化后的大模型在精度和延迟的权衡上好于未量化的小模型；对推理模型来说，校准数据的领域对精度影响很大，而且影响程度取决于量化方法，这和以前在预训练模型上“对校准数据领域不敏感”的结论不同。
- Dong 等，ACBench（2025-05）：评测 GPTQ、AWQ 和剪枝对 Agent 能力的影响。4 bit 量化下工作流生成和工具调用只下降 1% 到 3%，但真实应用类任务下降 10% 到 15%。

### 8.2 为什么 Agent 更敏感

- 工具调用参数要求逐字符精确。一个数字、一个 ID 错了，调用就失败或者调错对象。困惑度是所有 token 的平均，这类错误对它几乎没影响。
- 多步任务里错误会传递。单步成功率 p 的任务跑 k 步，整体约 p^k：p 从 0.99 降到 0.98，20 步的成功率从 82% 降到 67%（推导，假设各步独立）。
- 长链推理和长上下文依赖 KV 精度，KV 量化的误差随长度累积。

### 8.3 上线前的评测清单

1. 在自己的任务集上和 BF16 版本对比，不只看困惑度和通用榜单。
2. 单独统计工具调用的 JSON 合法率、函数名正确率、参数完全匹配率。
3. 按难度和轨迹长度分桶看任务成功率，长轨迹单独看。
4. 长上下文检索单独回归，尤其开了 KV 量化时。
5. 同时看输出长度和步数是否变多，量化省下的成本可能被更多 token 吃掉。
6. 自己做 PTQ 时，校准集混入带工具调用格式和长推理的样本，并和通用校准集各量化一版对比。

## 9 工程细节与常见坑

**1. 体积估算漏掉组开销和不量化的层**

- 现象：按“参数量 × 4 / 8”估的显存放得下，实际加载 OOM；或者量化文件比预期大 10% 到 20%。
- 原因：g128 加 FP16 scale 和 zero 是 4.25 位不是 4 位；嵌入层和输出层通常保持 16 位。第 4.3 节的 72B 例子里两项合计差约 6 GB。GPTQ 论文的 3 位 OPT-175B 也把嵌入和输出层保留为 FP16，总共约 63 GB。
- 排查：直接看量化后的 safetensors 文件总大小；按 `quantization_config` 里的 bits、group_size、是否有 zero 和不量化的模块列表重算。

**2. W4A16 上线后吞吐没提高，Prefill 还变慢了**

- 现象：单请求 Decode 延迟明显下降，但压测的总吞吐和 TTFT 没改善，长 prompt 时 TTFT 反而变差。
- 原因：第 4.2 节，W4A16 在 batch 约 78 以上就受算力限制，Prefill 一直受算力限制。反量化在主循环里做，是额外计算。
- 排查：按 prompt 长度和并发分档压测，分别看 TTFT 和每 token 延迟。高并发或长 prompt 为主的负载考虑 W8A8（FP8 或 INT8），或者至少把 Prefill 节点和 Decode 节点的量化方案分开（PD 分离见 [inference/02](../02-batching-pd/README.md)）。

**3. 激活 per-tensor 静态量化，某些输入上输出突然变差**

- 现象：校准集上评测正常，线上部分请求输出乱码或重复，集中在某类输入（长数字串、特殊格式）上。
- 原因：静态量化的激活 scale 来自校准集。线上出现更大的离群值时会被截断，误差不再受 s/2 约束（第 2.2 节第 3 条）。
- 排查：记录各层激活的 absmax，和校准时的值比较；换成 per-token 动态量化，或者用 SmoothQuant 平滑后再量化；扩大校准集的覆盖面。

**4. GPTQ 某一层误差爆炸，整个模型输出崩溃**

- 现象：量化日志里个别层的重构误差比其他层大几个数量级；量化后困惑度极高。
- 原因：H 接近奇异（例如死神经元让某些输入通道恒为 0，H 的对角元为 0），或者阻尼太小，H⁻¹ 数值不稳。GPTQ 论文指出大模型上不做 Cholesky 重写几乎必然有几层出问题。官方实现把对角元为 0 的通道置 1、对应权重置 0。
- 排查：打印每层的重构误差；增大阻尼比例（默认平均对角元的 1%）；增加校准样本数，让 H 满秩。

**5. 校准集和业务分布不匹配**

- 现象：通用评测接近无损，但工具调用的参数正确率或长推理成功率掉得明显。
- 原因：GPTQ 用校准集的 H 做重构，会贴合校准分布；Quantization Hurts Reasoning? 报告推理模型对校准数据领域敏感。AWQ 只统计平均幅度，依赖更弱，但不是零依赖。
- 排查：用通用校准集和“通用 + 业务轨迹”校准集各量化一版，在第 8.3 节的清单上对比。

**6. FP8 KV 没做校准**

- 现象：开了 `--kv-cache-dtype fp8` 后短任务正常，长上下文检索和长轨迹任务掉点。
- 原因：vLLM 不做校准时缩放因子是 1.0，K、V 的实际范围和 FP8 的表示范围（E4M3 最大 448）没有对齐，小值的相对精度变差；滑动窗口等层对 KV 量化更敏感。
- 排查：用 llm-compressor 校准 scale；用 `--kv-cache-dtype-skip-layers` 跳过敏感层；单独回归长上下文任务。

**7. RoPE 之后再按通道量化 Key**

- 现象：自己实现 Key 按通道量化，误差比预期大，长序列上更明显。
- 原因：RoPE 按位置旋转通道对，把离群通道的能量混到相邻通道，而且混合方式随位置变化（第 7.3 节）。
- 排查：缓存旋转前的 K，反量化之后再施加 RoPE；或者把离群通道成对处理。

**8. 不同工具的量化定义不一致**

- 现象：把一个工具导出的量化权重换到另一个引擎，输出明显变差，但没有报错。
- 原因：对称量化的分母是 2^(b−1) − 1 还是 2^(b−1)（第 2.1 节）；零点存成整数还是浮点；act-order 的列置换有没有被读取；分组方向是沿输入维还是输出维。
- 排查：用一层权重做反量化对拍，和原始 BF16 权重比较最大误差，应在半个步长以内（第 2.2 节）。

**9. QAT 最后一步的检查点比 PTQ 还差**

- 现象：QAT 训练 loss 后期不再下降而是来回跳；最终检查点部署后不如直接 PTQ。
- 原因：权重振荡（第 6.2 节）。
- 排查：统计每步有多少权重跨过格点边界；使用学习率衰减，按验证集选检查点；必要时冻结长期振荡的权重。

## 10 对比与取舍

### 10.1 按方案

| 方案 | 代表 | 适合 | 主要风险 |
|---|---|---|---|
| W8A16 / INT8 权重 | per-channel RTN | 想要几乎无损、只省一半权重显存 | 加速有限 |
| W4A16 | GPTQ、AWQ | 小 batch、显存紧、Decode 为主 | 高并发和 Prefill 不加速；小模型、难任务掉点 |
| W8A8 INT8 | SmoothQuant | 高并发、长 prompt，硬件有 INT8 Tensor Core | 激活离群值；需要校准 α |
| W8A8 FP8 | 原生 FP8 | Hopper 及以后的 GPU，高吞吐服务 | 硬件依赖；scale 校准 |
| KV FP8 | vLLM `fp8` | 长上下文、高并发，最省事的一步 | 不校准时长上下文掉点 |
| KV 2 到 4 bit | KIVI、KVQuant | 极长上下文 | kernel 支持；误差随长度累积 |
| QAT | LLM-QAT、官方 MXFP4 | 4 位以下的硬需求 | 训练成本；振荡 |

### 10.2 怎么选

1. 先算瓶颈：权重放不下，先做权重量化；权重放得下但并发上不去，先做 KV 量化（通常是 FP8 KV）。
2. 再看负载：小 batch Decode 为主用 W4A16；高并发或长 prompt 为主用 W8A8。
3. 有官方量化版本优先用官方的，尤其是 QAT 版本。
4. 无论选哪个，都按第 8.3 节的清单在自己的任务上验证。

## 11 自测清单

- [ ] 写出对称和非对称量化公式，推出误差上界 s/2 和期望 s²/12，说出每多 1 位误差降多少
- [ ] 解释为什么粒度越细误差越小，算出 INT4 g128 带 scale 和 zero 的等效位宽
- [ ] 说出激活为什么比权重难量化，以及为什么 per-token 解决不了通道离群值
- [ ] 推导 SmoothQuant 的平滑因子，说出 α = 0.5 时两侧最大值相等
- [ ] 写出 GPTQ 的单步补偿公式，说出固定列顺序、延迟批量更新、Cholesky 各解决什么
- [ ] 讲清 AWQ 为什么按激活选显著通道，为什么 s 太大反而变差
- [ ] 写出 STE 的前向和反向，说出权重振荡的现象和对策
- [ ] 说出 KIVI 为什么 Key 按通道、Value 按 token，KVQuant 为什么在 RoPE 之前量化
- [ ] 用算术强度解释 W4A16 为什么加速 Decode、未必加速 Prefill，算出转折的 batch
- [ ] 算出一个 72B 模型 INT4 后的权重大小和单卡能放的 KV
- [ ] 说出两篇论文里量化对推理和 Agent 任务的结论，并给出自己的评测清单

## 资料

- 论文精读：[AWQ](../../../papers/13-awq.md)
- Frantar 等，[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)，arXiv 2210.17323，2022-10（式 (1) 到 (5)、Algorithm 1、第 4 节实验设置）
- IST-DASLab，[gptq 官方仓库](https://github.com/IST-DASLab/gptq)（README 的 `--act-order`、`--static-groups`；`gptq.py` 的阻尼和死神经元处理），main 分支，2026-11 查阅
- Lin 等，[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)，arXiv 2306.00978，2023-06（式 (1) 到 (5)、Table 1 到 3、图 3）
- MIT HAN Lab，[llm-awq 的 `auto_scale.py`](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)（n_grid = 20 和缩放归一化），main 分支，2026-11 查阅
- Xiao 等，[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)，arXiv 2211.10438，2022-11（第 3 节、式 (2) 到 (4)、Table 7）
- Dettmers 等，[LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)，arXiv 2208.07339，2022-08（第 1 节离群特征分析、第 3.2 节混合精度分解）
- Micikevicius 等，[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)，arXiv 2209.05433，2022-09（E4M3 和 E5M2 的最大正规数）
- Bengio 等，[Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432)，arXiv 1308.3432，2013-08（直通估计）
- Nagel 等，[Overcoming Oscillations in Quantization-Aware Training](https://arxiv.org/abs/2203.11086)，arXiv 2203.11086，2022-03，ICML 2022
- Liu 等，[LLM-QAT: Data-Free Quantization Aware Training for Large Language Models](https://arxiv.org/abs/2305.17888)，arXiv 2305.17888，2023-05
- Liu 等，[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)，arXiv 2402.02750，2024-02（第 3.2 节、Table 2、Algorithm 1、第 4 节设置）
- Hooper 等，[KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079)，arXiv 2401.18079，2024-01（第 3.2 到 3.4 节）
- Liu 等，[Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models](https://arxiv.org/abs/2504.04823)，arXiv 2504.04823，2025-04（第 1 节结论列表、第 4.2 节校准数据）
- Dong 等，[Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression](https://arxiv.org/abs/2505.19433)，arXiv 2505.19433，2025-05
- vLLM，[Quantized KV Cache 文档](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)，2026-11 查阅
- OpenAI，[gpt-oss-120b 模型卡](https://huggingface.co/openai/gpt-oss-120b)，Hugging Face，2025-08
- Qwen 团队，[Qwen2.5-72B-Instruct 模型卡与 config.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)，Hugging Face，2026-11 查阅
- NVIDIA，[H100 Tensor Core GPU 产品页](https://www.nvidia.com/en-us/data-center/h100/)（BF16、FP8、INT8 稠密算力和显存带宽），2026-11 查阅
