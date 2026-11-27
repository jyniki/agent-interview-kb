'''code-model-001：手写 Multi-Head Attention（含因果 mask）。

题目：Q、K、V 线性投影后拆成 h 个头，每头算 softmax(QKᵀ/√d_k + mask)V，拼接后再做一次输出投影。
复杂度：时间 O(B·T·D² + B·h·T²·d_k)，空间 O(B·h·T²)（注意力分数）。
追问已覆盖：GQA（num_kv_heads < num_heads）、KV Cache（past_kv）。
'''
import numpy as np


def softmax(x, axis=-1):
    '''稳定 softmax：先减去最大值再 exp。'''
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def multi_head_attention(x, Wq, Wk, Wv, Wo, num_heads, num_kv_heads=None,
                         causal=True, past_kv=None, return_kv=False):
    '''x: (B, T, D)；Wq: (D, h*d_k)；Wk、Wv: (D, n_kv*d_k)；Wo: (h*d_k, D)。
    past_kv: (K, V)，形状 (B, n_kv, T_past, d_k)。返回 (B, T, D)；return_kv 时另返回更新后的缓存。'''
    B, T, D = x.shape
    h = num_heads
    n_kv = h if num_kv_heads is None else num_kv_heads
    assert D % h == 0 and h % n_kv == 0
    d_k = D // h
    q = (x @ Wq).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
    k = (x @ Wk).reshape(B, T, n_kv, d_k).transpose(0, 2, 1, 3)
    v = (x @ Wv).reshape(B, T, n_kv, d_k).transpose(0, 2, 1, 3)
    if past_kv is not None:
        k = np.concatenate([past_kv[0], k], axis=2)
        v = np.concatenate([past_kv[1], v], axis=2)
    cache = (k, v)
    rep = h // n_kv
    k = np.repeat(k, rep, axis=1)
    v = np.repeat(v, rep, axis=1)
    T_k = k.shape[2]
    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    if causal:
        future = np.arange(T_k)[None, :] > (np.arange(T)[:, None] + T_k - T)
        scores = np.where(future, -np.inf, scores)
    out = softmax(scores) @ v
    out = out.transpose(0, 2, 1, 3).reshape(B, T, h * d_k) @ Wo
    return (out, cache) if return_kv else out


def _ref(x, Wq, Wk, Wv, Wo, h, scale=True):
    '''逐样本、逐头的朴素参考实现。'''
    B, T, D = x.shape
    d = D // h
    res = np.zeros((B, T, D))
    for b in range(B):
        heads = []
        for i in range(h):
            sl = slice(i * d, (i + 1) * d)
            s = (x[b] @ Wq[:, sl]) @ (x[b] @ Wk[:, sl]).T
            s = s / (np.sqrt(d) if scale else 1.0)
            for t in range(T):
                s[t, t + 1:] = -np.inf
            p = np.exp(s - s.max(axis=-1, keepdims=True))
            p = p / p.sum(axis=-1, keepdims=True)
            heads.append(p @ (x[b] @ Wv[:, sl]))
        res[b] = np.concatenate(heads, axis=-1) @ Wo
    return res


np.random.seed(0)
B, T, D, h = 2, 6, 16, 4
d_k = D // h
x = np.random.randn(B, T, D)
Wq, Wk, Wv, Wo = (np.random.randn(D, D) * 0.3 for _ in range(4))
out = multi_head_attention(x, Wq, Wk, Wv, Wo, h)

# 形状、头拆分与拼接顺序：和逐头循环逐元素一致（reshape 前漏 transpose 会在这里暴露）
assert out.shape == (B, T, D)
assert np.allclose(out, _ref(x, Wq, Wk, Wv, Wo, h), atol=1e-10)
# 缩放因子 1/sqrt(d_k) 确实生效
assert not np.allclose(out, _ref(x, Wq, Wk, Wv, Wo, h, scale=False), atol=1e-3)

# 因果 mask：改动位置 t 之后的 token，t 及之前的输出不变，之后的输出会变
for t in range(T - 1):
    x2 = x.copy()
    x2[:, t + 1:] = np.random.randn(B, T - t - 1, D) * 5
    out2 = multi_head_attention(x2, Wq, Wk, Wv, Wo, h)
    assert np.allclose(out2[:, :t + 1], out[:, :t + 1], atol=1e-10)
    assert not np.allclose(out2[:, t + 1:], out[:, t + 1:], atol=1e-3)

# 不加 mask 时未来 token 会影响前面位置
out_nc = multi_head_attention(x, Wq, Wk, Wv, Wo, h, causal=False)
x3 = x.copy()
x3[:, -1] = np.random.randn(B, D) * 5
out_nc3 = multi_head_attention(x3, Wq, Wk, Wv, Wo, h, causal=False)
assert not np.allclose(out_nc3[:, 0], out_nc[:, 0], atol=1e-6)
assert not np.allclose(out_nc, out, atol=1e-6)

# 退化情形：第 0 个位置只看自己；T = 1 时注意力权重恒为 1，输出等于 x·Wv·Wo
assert np.allclose(multi_head_attention(x[:, :1], Wq, Wk, Wv, Wo, h)[:, 0], out[:, 0], atol=1e-10)
assert np.allclose(multi_head_attention(x[:, :1], Wq, Wk, Wv, Wo, h), x[:, :1] @ Wv @ Wo, atol=1e-10)

# 数值稳定：极端输入不产生 nan/inf
big = multi_head_attention(x * 1e3, Wq, Wk, Wv, Wo, h)
assert np.all(np.isfinite(big))
p = softmax(np.array([[1000.0, 1001.0, -np.inf]]))
assert np.all(np.isfinite(p)) and abs(p.sum() - 1) < 1e-12 and p[0, 2] == 0

# GQA：K、V 头数变少，等价于把每个 KV 头连续复制 rep 次的 MHA；用 tile 的顺序会算错
n_kv = 2
rep = h // n_kv
Wk_g = np.random.randn(D, n_kv * d_k) * 0.3
Wv_g = np.random.randn(D, n_kv * d_k) * 0.3
out_g = multi_head_attention(x, Wq, Wk_g, Wv_g, Wo, h, num_kv_heads=n_kv)
assert out_g.shape == (B, T, D)
Wk_full = np.repeat(Wk_g.reshape(D, n_kv, d_k), rep, axis=1).reshape(D, D)
Wv_full = np.repeat(Wv_g.reshape(D, n_kv, d_k), rep, axis=1).reshape(D, D)
assert np.allclose(out_g, multi_head_attention(x, Wq, Wk_full, Wv_full, Wo, h), atol=1e-10)
Wk_tile = np.tile(Wk_g.reshape(D, n_kv, d_k), (1, rep, 1)).reshape(D, D)
Wv_tile = np.tile(Wv_g.reshape(D, n_kv, d_k), (1, rep, 1)).reshape(D, D)
assert not np.allclose(out_g, multi_head_attention(x, Wq, Wk_tile, Wv_tile, Wo, h), atol=1e-3)
out_mqa = multi_head_attention(x, Wq, Wk_g[:, :d_k], Wv_g[:, :d_k], Wo, h, num_kv_heads=1)
assert out_mqa.shape == (B, T, D)

# KV Cache：prefill 后逐 token decode，和整段前向一致（容差 1e-6）
T0 = 3
o_pre, kv = multi_head_attention(x[:, :T0], Wq, Wk, Wv, Wo, h, return_kv=True)
outs = [o_pre]
for t in range(T0, T):
    o, kv = multi_head_attention(x[:, t:t + 1], Wq, Wk, Wv, Wo, h, past_kv=kv, return_kv=True)
    assert kv[0].shape == (B, h, t + 1, d_k)
    outs.append(o)
assert np.allclose(np.concatenate(outs, axis=1), out, atol=1e-6)

# decode 只有一个 Q 位置，不加因果 mask 结果相同；GQA 的缓存只存 n_kv 个头
_, kv_p = multi_head_attention(x[:, :T - 1], Wq, Wk, Wv, Wo, h, return_kv=True)
o_a = multi_head_attention(x[:, T - 1:], Wq, Wk, Wv, Wo, h, past_kv=kv_p, causal=True)
o_b = multi_head_attention(x[:, T - 1:], Wq, Wk, Wv, Wo, h, past_kv=kv_p, causal=False)
assert np.allclose(o_a, o_b, atol=1e-10) and np.allclose(o_a, out[:, -1:], atol=1e-6)
_, kv_g = multi_head_attention(x, Wq, Wk_g, Wv_g, Wo, h, num_kv_heads=n_kv, return_kv=True)
assert kv_g[0].shape == (B, n_kv, T, d_k)
