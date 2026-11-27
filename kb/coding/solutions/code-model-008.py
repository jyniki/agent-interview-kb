'''code-model-008：手写 top-k / top-p 采样。

题目：logits (B, V)、temperature、k、p，输出下一个 token。顺序为 温度 → top-k → top-p → softmax → 采样。
复杂度：时间 O(B·V log V)（排序；先 top-k 再 top-p 可降到 O(B·V + B·k log k)），空间 O(B·V)。
'''
import numpy as np


def top_k_filter(logits, k):
    '''把第 k 大以下的 logit 置 −inf。k 为 None 或 ≥ V 时不动。'''
    if k is None or k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    kth = np.partition(logits, -k, axis=-1)[..., -k][..., None]
    return np.where(logits < kth, -np.inf, logits)


def top_p_filter(logits, p):
    '''按概率降序取累计到 p 的最小集合：remove = cumsum(probs) − probs >= p。
    这样跨过 p 的那个 token 一定保留；p 很小时退化成贪心，集合不会为空。'''
    if p is None or p >= 1.0:
        return logits
    p = max(float(p), 1e-12)          # p <= 0 时避免把第一个 token 也删掉、softmax 得到 nan
    probs = softmax(logits)
    order = np.argsort(-probs, axis=-1)
    sorted_p = np.take_along_axis(probs, order, axis=-1)
    keep = np.cumsum(sorted_p, axis=-1) - sorted_p < p
    keep[..., 0] = True               # 概率最大的 token 永远保留
    out = np.full_like(logits, -np.inf)
    np.put_along_axis(out, order, np.where(keep, np.take_along_axis(logits, order, axis=-1), -np.inf), axis=-1)
    return out


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    s = e.sum(axis=axis, keepdims=True)
    return e / np.where(s > 0, s, 1.0)


def sample_probs(logits, temperature=1.0, k=None, p=None):
    '''返回截断并重新归一化后的 (B, V) 概率；temperature = 0 时返回 one-hot argmax。'''
    logits = np.atleast_2d(np.asarray(logits, dtype=np.float64))
    if temperature is not None and temperature == 0:
        out = np.zeros_like(logits)
        out[np.arange(logits.shape[0]), logits.argmax(axis=-1)] = 1.0
        return out
    scaled = logits / temperature            # 温度先于截断
    scaled = top_k_filter(scaled, k)
    scaled = top_p_filter(scaled, p)
    return softmax(scaled)


def sample(logits, temperature=1.0, k=None, p=None, rng=None):
    '''输出 (B,) 的 token id。'''
    probs = sample_probs(logits, temperature, k, p)
    if temperature is not None and temperature == 0:
        return probs.argmax(axis=-1)
    rng = np.random.default_rng(0) if rng is None else rng
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((probs.shape[0], 1))
    return np.minimum((cdf < u).sum(axis=-1), probs.shape[-1] - 1)


np.random.seed(0)

# 题目给的例子：概率 [0.5, 0.3, 0.15, 0.05]、p = 0.7 → 保留前两个，重新归一化为 [0.625, 0.375]
ex = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
pr = sample_probs(ex, p=0.7)
assert np.allclose(pr, [[0.625, 0.375, 0.0, 0.0]], atol=1e-12)
# 跨过 p 的那个 token 保留：p=0.51 时“之前已够 0.5、再加 0.3 才跨过 p”的那个被保留；
# 而 p=0.5 时第一个 token 自己就够 0.5，集合只含它（累计到 p 的最小集合）
assert np.allclose(sample_probs(ex, p=0.51), [[5 / 8, 3 / 8, 0, 0]], atol=1e-12)
assert np.allclose(sample_probs(ex, p=0.5), [[1, 0, 0, 0]], atol=1e-12)
assert np.allclose(sample_probs(ex, p=0.99), [[0.5, 0.3, 0.15, 0.05]], atol=1e-12)
# p 很小 → 退化成贪心，且不产生 nan（不会把第一个 token 也删掉）
tiny_p = sample_probs(ex, p=1e-9)
assert tiny_p[0, 0] == 1.0 and np.all(np.isfinite(tiny_p))
assert np.allclose(sample_probs(ex, p=0.0), tiny_p)

# top-k：k=1 等于贪心；k=2 只保留两个；k ≥ V 不截断
assert np.allclose(sample_probs(ex, k=1), [[1, 0, 0, 0]])
assert np.allclose(sample_probs(ex, k=2), [[0.625, 0.375, 0, 0]])
assert np.allclose(sample_probs(ex, k=None), [[0.5, 0.3, 0.15, 0.05]], atol=1e-12)
assert np.allclose(sample_probs(ex, k=10), [[0.5, 0.3, 0.15, 0.05]], atol=1e-12)

# 温度：0 → argmax，不除以 0；低温更尖，高温更平
assert np.allclose(sample_probs(ex, temperature=0), [[1, 0, 0, 0]])
low = sample_probs(ex, temperature=0.5)
high = sample_probs(ex, temperature=2.0)
assert low[0, 0] > 0.5 and high[0, 0] < 0.5
assert np.all(np.isfinite(sample_probs(ex, temperature=1e-8)))
# 温度放在截断之前：同样的 p，高温下保留集合更大
assert (sample_probs(ex, temperature=1.0, p=0.9)[0] > 0).sum() == 3
assert (sample_probs(ex, temperature=3.0, p=0.9)[0] > 0).sum() >= 3

# 下标还原：截断后按原下标还原，采到的 id 是原位置而不是排序后的位置
logits = np.full((1, 6), -10.0)
logits[0, 4] = 10.0
logits[0, 1] = 9.0
ids = {int(sample(logits, p=0.5, rng=np.random.default_rng(s))[0]) for s in range(20)}
assert ids <= {4, 1} and 4 in ids
assert int(sample(logits, k=1)[0]) == 4
# 排序后位置 0 对应原下标 4
assert np.argmax(sample_probs(logits, k=1)[0]) == 4

# 批量 (B, V)：逐行独立，形状与 dtype
batch = np.random.randn(5, 11) * 3
out = sample(batch, temperature=0.8, k=5, p=0.9, rng=np.random.default_rng(1))
assert out.shape == (5,) and np.issubdtype(out.dtype, np.integer)
assert np.all((out >= 0) & (out < 11))
for b in range(5):
    allowed = np.where(sample_probs(batch[b][None], 0.8, 5, 0.9)[0] > 0)[0]
    assert out[b] in allowed

# 采样确实只在保留集合里（多次采样，统计支持集）
sup = set()
rng = np.random.default_rng(7)
for _ in range(500):
    sup.add(int(sample(ex, p=0.7, rng=rng)[0]))
assert sup == {0, 1}

# 数值稳定：极端 logits 不产生 nan/inf，概率和为 1
ext = np.array([[1e4, 1e4 - 1, -1e4, -np.inf]])
pe = sample_probs(ext)
assert np.all(np.isfinite(pe)) and abs(pe.sum() - 1) < 1e-12 and pe[0, 3] == 0
pe2 = sample_probs(ext, p=0.99)
assert np.all(np.isfinite(pe2)) and abs(pe2.sum() - 1) < 1e-12
