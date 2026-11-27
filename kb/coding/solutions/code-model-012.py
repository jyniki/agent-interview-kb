'''code-model-012：手写 GQA 的 KV 头复制。

题目：K、V 形状 (B, n_kv, T, d)，复制成 (B, n_q, T, d)；每个 KV 头服务连续的 rep = n_q/n_kv 个 Q 头。
复杂度：复制后时间 O(B·n_q·T·d)、空间 O(B·n_q·T·d)；不复制、把 Q 拆成 (B, n_kv, rep, T, d) 广播的写法
空间 O(B·n_kv·T·d)，KV Cache 省下的就是 rep 倍。
'''
import numpy as np


def repeat_kv(x, n_q):
    '''插轴 + 广播 + reshape：每个 head 连续出现 rep 次。
    数学上等价 np.repeat(x, rep, axis=1)（即 repeat_interleave）。'''
    B, n_kv, T, d = x.shape
    assert n_q % n_kv == 0, 'n_q 必须能被 n_kv 整除'
    rep = n_q // n_kv
    y = np.broadcast_to(x[:, :, None, :, :], (B, n_kv, rep, T, d))
    return np.ascontiguousarray(y).reshape(B, n_q, T, d)


def repeat_kv_tile(x, n_q):
    '''错误示范：tile 得到的顺序是 [0, 1, 0, 1]，Q 头和 KV 头的对应关系错了。'''
    rep = n_q // x.shape[1]
    return np.tile(x, (1, rep, 1, 1))


def gqa_attention(q, k, v, causal=True):
    '''q: (B, n_q, T, d)，k、v: (B, n_kv, T, d)。不复制 K、V：
    把 Q reshape 成 (B, n_kv, rep, T, d)，直接与 K 广播相乘。'''
    B, n_q, T, d = q.shape
    n_kv = k.shape[1]
    assert n_q % n_kv == 0
    rep = n_q // n_kv
    qg = q.reshape(B, n_kv, rep, T, d)
    scores = np.einsum('bkrqd,bksd->bkrqs', qg, k) / np.sqrt(d)
    if causal:
        # 只挡 future（triu k=1）；把对角线也挡掉会让第一行全 -inf、softmax 出 nan
        scores = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, scores)
    mx = scores.max(axis=-1, keepdims=True)
    p = np.exp(scores - np.where(np.isfinite(mx), mx, 0.0))
    den = p.sum(-1, keepdims=True)
    p = p / np.where(den > 0, den, 1.0)
    return np.einsum('bkrqs,bksd->bkrqd', p, v).reshape(B, n_q, T, d)


def mha_reference(q, k, v):
    '''普通 MHA 的参考实现（要求 q、k、v 头数相同）。'''
    B, H, T, d = q.shape
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d)
    s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


np.random.seed(0)
B, n_kv, n_q, T, d = 2, 2, 8, 5, 4
rep = n_q // n_kv
k = np.random.randn(B, n_kv, T, d)
v = np.random.randn(B, n_kv, T, d)
q = np.random.randn(B, n_q, T, d)

# 形状，且与 repeat_interleave 完全一致
k_rep, v_rep = repeat_kv(k, n_q), repeat_kv(v, n_q)
assert k_rep.shape == (B, n_q, T, d) and v_rep.shape == (B, n_q, T, d)
assert np.array_equal(k_rep, np.repeat(k, rep, axis=1))
assert np.array_equal(v_rep, np.repeat(v, rep, axis=1))

# 头对应关系：Q 头 j 对应 KV 头 j // rep；每个 KV 头连续出现 rep 次
for j in range(n_q):
    assert np.array_equal(k_rep[:, j], k[:, j // rep])
assert np.array_equal(k_rep[:, :rep], np.broadcast_to(k[:, :1], (B, rep, T, d)))
assert np.array_equal(k_rep[:, rep:2 * rep], np.broadcast_to(k[:, 1:2], (B, rep, T, d)))
# tile 的顺序是 [0, 1, 0, 1]：head 1 被换成了 head 1（tile 下），不是 head 0
assert not np.array_equal(k_rep, repeat_kv_tile(k, n_q))
assert np.array_equal(k_rep[:, 1], k[:, 0]) and np.array_equal(repeat_kv_tile(k, n_q)[:, 1], k[:, 1])
# 不在 T 维上复制：序列长度不变
assert k_rep.shape[2] == T
assert not np.array_equal(k_rep, np.repeat(k, rep, axis=2))

# rep = 1 是 MHA：原样返回；n_kv = 1 是 MQA：一个 KV 头服务全部 Q 头
assert np.array_equal(repeat_kv(k, n_kv), k)
mqa = repeat_kv(k[:, :1], n_q)
assert mqa.shape == (B, n_q, T, d)
assert all(np.array_equal(mqa[:, j], k[:, 0]) for j in range(n_q))
# n_q 不能被 n_kv 整除时报错
try:
    repeat_kv(k, 5)
    raise SystemExit('应当断言失败')
except AssertionError:
    pass

# 数值等价：复制 K、V 头后跑普通 MHA，与不复制、广播的写法一致
assert np.allclose(gqa_attention(q, k, v), mha_reference(q, k_rep, v_rep), atol=1e-12)
# 与逐头循环一致
for j in range(n_q):
    single = mha_reference(q[:, j:j + 1], k_rep[:, j:j + 1], v_rep[:, j:j + 1])
    assert np.allclose(gqa_attention(q, k, v)[:, j:j + 1], single, atol=1e-12)

# 因果 mask：改最后一个 token 不影响前面位置
k2, v2 = k.copy(), v.copy()
k2[:, :, -1] = np.random.randn(B, n_kv, d) * 5
v2[:, :, -1] = np.random.randn(B, n_kv, d) * 5
full = gqa_attention(q, k, v)
pert = gqa_attention(q, k2, v2)
assert np.allclose(pert[:, :, :-1], full[:, :, :-1], atol=1e-12)
assert not np.allclose(pert[:, :, -1], full[:, :, -1], atol=1e-6)
# T = 1：每个位置只看自己，输出等于对应的 V 头
one = gqa_attention(q[:, :, :1], k[:, :, :1], v[:, :, :1])
assert np.allclose(one, v_rep[:, :, :1], atol=1e-12)

# 缓存只存 n_kv 个头：写入缓存发生在复制之前
cache = k[:, :, :3].copy()
assert cache.shape == (B, n_kv, 3, d)
assert repeat_kv(cache, n_q).shape == (B, n_q, 3, d)
assert k_rep.size == rep * k.size
assert cache.nbytes * rep == repeat_kv(cache, n_q).nbytes

# 数值稳定：大 logits 不产生 nan/inf
big = gqa_attention(q * 1e3, k, v)
assert np.all(np.isfinite(big))
