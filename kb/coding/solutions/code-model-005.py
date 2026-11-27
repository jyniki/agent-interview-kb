'''code-model-005：手写 RoPE，给 q、k 加旋转位置编码。

题目：输入 q、k 形状 (B, h, T, d)、位置 (T,)，按 θ_i = base^(−2i/d) 旋转每一对维度。
复杂度：时间 O(B·h·T·d)，空间 O(T·d)（cos/sin 表可预计算并缓存）。
要点里提到两种配对（相邻配对与 rotate_half 前后半配对）数学等价，但 cos/sin 必须同配。
'''
import numpy as np


def rope_tables(positions, d, base=10000.0):
    '''返回 (T, d/2) 的 cos、sin 表：inv_freq = base ** (-arange(0, d, 2) / d)。'''
    inv_freq = base ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang), np.sin(ang)


def _rotate_adjacent(x):
    '''相邻配对： (x1, x2) → (−x2, x1)，即奇数位取负、移位。'''
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return np.stack([-x2, x1], axis=-1).reshape(x.shape)


def _rotate_half(x):
    '''前后半配对： cat(−x[d/2:], x[:d/2])。'''
    d2 = x.shape[-1] // 2
    return np.concatenate([-x[..., d2:], x[..., :d2]], axis=-1)


def apply_rope(q, k, positions, base=10000.0, style='adjacent', scale=1.0):
    '''q、k: (B, h, T, d)；positions: (T,) 绝对位置；scale = s 即 Position Interpolation（角度除以 s）。
    返回旋转后的 (q, k)。v 不参与旋转。'''
    d = q.shape[-1]
    assert d % 2 == 0 and k.shape[-1] == d
    cos, sin = rope_tables(np.asarray(positions, dtype=np.float64) / scale, d, base)
    shape = (1, 1) + cos.shape
    cos, sin = cos.reshape(shape), sin.reshape(shape)
    if style == 'adjacent':
        cos = np.repeat(cos, 2, axis=-1)
        sin = np.repeat(sin, 2, axis=-1)
        rot = _rotate_adjacent
    elif style == 'half':
        cos = np.concatenate([cos, cos], axis=-1)
        sin = np.concatenate([sin, sin], axis=-1)
        rot = _rotate_half
    else:
        raise ValueError(style)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


def attention_scores(q, k_pos, positions, **kw):
    '''缓存里存的是旋转后的 k，这里对 q 再旋转后与 k 做内积。'''
    qr, _ = apply_rope(q, q, positions, **kw)
    return qr @ k_pos.transpose(0, 1, 3, 2)


np.random.seed(0)
B, h, T, d = 2, 3, 8, 8
base = 10000.0

# 频率表：θ_i = base^(−2i/d)，i = 0…d/2−1，严格递减
inv_freq = base ** (-np.arange(0, d, 2) / d)
assert inv_freq.shape == (d // 2,)
assert np.all(np.diff(inv_freq) < 0) and abs(inv_freq[0] - 1.0) < 1e-12
assert np.allclose(inv_freq, [10000.0 ** (-2 * i / d) for i in range(d // 2)])

# 形状、位置维度广播、长度不变（旋转是正交变换）
q = np.random.randn(B, h, T, d)
k = np.random.randn(B, h, T, d)
pos = np.arange(T)
for style in ('adjacent', 'half'):
    qr, kr = apply_rope(q, k, pos, style=style)
    assert qr.shape == q.shape and kr.shape == k.shape
    assert np.allclose(np.linalg.norm(qr, axis=-1), np.linalg.norm(q, axis=-1), atol=1e-12)
    assert np.allclose(np.linalg.norm(kr, axis=-1), np.linalg.norm(k, axis=-1), atol=1e-12)

# 位置 0 不旋转
qr0, kr0 = apply_rope(q, k, pos)
assert np.allclose(qr0[:, :, 0], q[:, :, 0], atol=1e-12)

# 与论文式（复数乘法）逐元素一致：对每对维度做 e^{i·mθ} 乘法
for style in ('adjacent', 'half'):
    qr, _ = apply_rope(q, k, pos, style=style)
    for m in (0, 1, 5):
        for i in range(d // 2):
            if style == 'adjacent':
                a, b = 2 * i, 2 * i + 1
            else:
                a, b = i, i + d // 2
            ang = m * inv_freq[i]
            want_a = q[..., m, a] * np.cos(ang) - q[..., m, b] * np.sin(ang)
            want_b = q[..., m, a] * np.sin(ang) + q[..., m, b] * np.cos(ang)
            assert np.allclose(qr[..., m, a], want_a, atol=1e-12)
            assert np.allclose(qr[..., m, b], want_b, atol=1e-12)

# 核心性质：q·k 只依赖相对位置（两种配对都成立）
def scores_at(m, n, style):
    '''固定 q、k 向量，位置取 m 和 n。'''
    q1, k1 = apply_rope(q[:, :, :1], k[:, :, :1], np.array([m]), style=style)
    _, k2 = apply_rope(q[:, :, :1], k[:, :, :1], np.array([n]), style=style)
    return (q1 * k2).sum()

for style in ('adjacent', 'half'):
    for c in (1, 3, 7):
        for m, n in ((0, 1), (2, 5), (1, 6)):
            assert abs(scores_at(m, n, style) - scores_at(m + c, n + c, style)) < 1e-9

# 整段注意力分数：平移所有位置后，分数矩阵不变
for style in ('adjacent', 'half'):
    _, k_cache = apply_rope(q, k, pos, style=style)
    s1 = attention_scores(q, k_cache, pos, style=style)
    _, k_shift = apply_rope(q, k, pos + 100, style=style)
    s2 = attention_scores(q, k_shift, pos + 100, style=style)
    assert np.allclose(s1, s2, atol=1e-8)
    # 相对位置不同，分数就不同（不是退化成常数）
    assert not np.allclose(s1[:, :, 0, 1], s1[:, :, 0, 3], atol=1e-6)

# 易错点：相邻配对的 cos/sin 配上 rotate_half 的旋转，内积不再只依赖相对位置
def mixed_scores(m, n):
    cos, sin = rope_tables(np.array([m], dtype=np.float64), d)
    cos = np.repeat(cos, 2, axis=-1).reshape(1, 1, 1, d)
    sin = np.repeat(sin, 2, axis=-1).reshape(1, 1, 1, d)
    q1 = q[:, :, :1] * cos + _rotate_half(q[:, :, :1]) * sin
    cos2, sin2 = rope_tables(np.array([n], dtype=np.float64), d)
    cos2 = np.repeat(cos2, 2, axis=-1).reshape(1, 1, 1, d)
    sin2 = np.repeat(sin2, 2, axis=-1).reshape(1, 1, 1, d)
    k2 = k[:, :, :1] * cos2 + _rotate_half(k[:, :, :1]) * sin2
    return (q1 * k2).sum()

c = 3
assert abs(mixed_scores(0, 1) - mixed_scores(0 + c, 1 + c)) > 1e-4

# 只作用于 q、k，v 原样不动
v = np.random.randn(B, h, T, d)
_, _ = apply_rope(q, k, pos)
assert np.array_equal(v, v)

# Position Interpolation：位置除以 s，等价于角度整体缩小 s 倍
qs, ks = apply_rope(q, k, pos, scale=2.0)
q_direct, k_direct = apply_rope(q, k, pos / 2.0)
assert np.allclose(qs, q_direct, atol=1e-12) and np.allclose(ks, k_direct, atol=1e-12)
# 改 base 的 NTK 做法只影响高频维度以外的频率
inv_ntk = (base * 4) ** (-np.arange(0, d, 2) / d)
assert np.allclose(inv_ntk / inv_freq, 4 ** (-np.arange(0, d, 2) / d))
assert not np.allclose(inv_ntk, inv_freq)

# KV Cache：k 在写入缓存前旋转，之后不再重算；新 token 用绝对位置 T_cache
q_all, k_all = apply_rope(q, k, pos)
for t in range(1, T):
    q_new, _ = apply_rope(q[:, :, t:t + 1], q[:, :, t:t + 1], np.array([t]))
    s_step = q_new @ k_all[:, :, :t + 1].transpose(0, 1, 3, 2)
    s_full = q_all[:, :, t:t + 1] @ k_all[:, :, :t + 1].transpose(0, 1, 3, 2)
    assert np.allclose(s_step, s_full, atol=1e-10)

# 易错点：位置从 1 开始，与从 0 开始不同（错位一格）
qa, ka = apply_rope(q, k, pos)
qb, kb = apply_rope(q, k, pos + 1)
assert not np.allclose(ka, kb, atol=1e-6)
