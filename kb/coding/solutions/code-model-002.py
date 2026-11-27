'''code-model-002：手写 GRPO 的组内优势和 token 级 loss。

题目：按问题分组算 (r − mean)/(std + eps)，广播到每个 token；loss 用 clip 目标，乘 response mask 后聚合。
复杂度：设 N = B·G 条回答、最长 T：时间 O(N·T)，空间 O(N·T)。
约定：std 取总体标准差（ddof=0）。题目未规定口径，torch.std 默认 ddof=1，只影响尺度。
'''
import numpy as np


def grpo_advantages(rewards, group_size, eps=1e-6, std_norm=True):
    '''rewards: (B·G,)，同一问题的 G 条回答连续排列。返回 (B·G,)。'''
    r = np.asarray(rewards, dtype=np.float64)
    assert r.ndim == 1 and r.size % group_size == 0
    g = r.reshape(-1, group_size)
    adv = g - g.mean(axis=1, keepdims=True)
    if std_norm:
        adv = adv / (g.std(axis=1, keepdims=True) + eps)
    return adv.reshape(-1)


def grpo_loss(rewards, logp_new, logp_old, mask, group_size, logp_ref=None,
              clip_eps=0.2, kl_coef=0.0, eps=1e-6, std_norm=True,
              norm='token', const_denom=None):
    '''logp_*、mask: (B·G, T)。
    norm=token：有效 token 损失求和 / 有效 token 总数（DAPO）；
    norm=sample：每条回答先按自身长度平均，再对所有回答平均（GRPO 原文）；
    norm=const：求和 / (N * const_denom)，分母为固定常数（Dr. GRPO，同时配 std_norm=False）。'''
    m = np.asarray(mask) > 0
    adv = grpo_advantages(rewards, group_size, eps, std_norm)[:, None]
    new = np.asarray(logp_new, dtype=np.float64)
    diff = np.where(m, new - np.asarray(logp_old, dtype=np.float64), 0.0)
    ratio = np.exp(diff)
    per_tok = -np.minimum(ratio * adv, np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    if logp_ref is not None and kl_coef != 0:
        d = np.where(m, np.asarray(logp_ref, dtype=np.float64) - new, 0.0)
        per_tok = per_tok + kl_coef * (np.exp(d) - d - 1)
    per_tok = np.where(m, per_tok, 0.0)
    n = m.shape[0]
    if norm == 'token':
        return per_tok.sum() / max(m.sum(), 1)
    if norm == 'sample':
        return (per_tok.sum(axis=1) / np.maximum(m.sum(axis=1), 1)).mean()
    if norm == 'const':
        return per_tok.sum() / (n * const_denom)
    raise ValueError(norm)


np.random.seed(0)

# 优势：形状、组内零均值、单位方差、仿射不变、组间互不影响
r = np.random.randn(12) * 3 + 1
adv = grpo_advantages(r, 4)
assert adv.shape == (12,)
ag = adv.reshape(3, 4)
assert np.allclose(ag.mean(axis=1), 0, atol=1e-9)
assert np.allclose(ag.std(axis=1), 1, atol=1e-4)
assert np.allclose(grpo_advantages(r * 10 + 3, 4, eps=1e-12), grpo_advantages(r, 4, eps=1e-12), atol=1e-6)
r2 = r.copy()
r2[4:] = np.random.randn(8)
assert np.allclose(grpo_advantages(r2, 4)[:4], adv[:4])

# 退化：组内奖励全相同（std = 0），加了 eps 不出 nan，优势为 0；不加 eps 会得到 nan
adv0 = grpo_advantages(np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]), 4)
assert np.all(np.isfinite(adv0)) and np.all(adv0[:4] == 0)
with np.errstate(all='ignore'):
    assert np.all(np.isnan(grpo_advantages(np.array([1.0, 1.0]), 2, eps=0.0)))

# 聚合口径：G = 2，奖励 [1, 0] 得优势 [1, -1]；两条回答的有效长度为 3 和 1；on-policy 时 ratio = 1
rw = np.array([1.0, 0.0])
lp = np.zeros((2, 3))
mk = np.array([[1, 1, 1], [1, 0, 0]])
kw = dict(group_size=2, eps=0.0)
assert np.ndim(grpo_loss(rw, lp, lp, mk, **kw)) == 0
assert abs(grpo_loss(rw, lp, lp, mk, norm='token', **kw) - (-0.5)) < 1e-12
assert abs(grpo_loss(rw, lp, lp, mk, norm='sample', **kw) - 0.0) < 1e-12
# Dr. GRPO：优势不除 std（[0.5, -0.5]），分母固定为 N * 4 = 8
assert abs(grpo_loss(rw, lp, lp, mk, norm='const', std_norm=False, const_denom=4, **kw) - (-0.125)) < 1e-12

# clip：ratio = 2。A > 0 被截到 1.2；A < 0 取未截断的 2·A（min 保留变坏方向的惩罚）
one = np.ones((2, 1))
assert abs(grpo_loss(rw, np.full((2, 1), np.log(2.0)), np.zeros((2, 1)), one, **kw) - 0.4) < 1e-12
# ratio = 0.5。A > 0 取 0.5·A（不截断）；A < 0 取 0.8·A（被截断）
assert abs(grpo_loss(rw, np.full((2, 1), np.log(0.5)), np.zeros((2, 1)), one, **kw) - 0.15) < 1e-12

# KL 项：k3 估计 exp(d) − d − 1，d = ref − new；ref == new 时为 0，其余恒 ≥ 0
rw_eq = np.array([1.0, 1.0])
new1 = np.zeros((2, 1))
ref1 = np.full((2, 1), np.log(2.0))
assert abs(grpo_loss(rw_eq, new1, new1, one, 2, logp_ref=ref1, kl_coef=1.0) - (1 - np.log(2.0))) < 1e-12
assert abs(grpo_loss(rw_eq, new1, new1, one, 2, logp_ref=new1, kl_coef=1.0)) < 1e-12
new_r = np.random.randn(8, 6) * 0.3
ref_r = np.random.randn(8, 6) * 0.3
ones_r = np.ones((8, 6))
assert grpo_loss(np.ones(8), new_r, new_r, ones_r, 4, logp_ref=ref_r, kl_coef=1.0) >= 0

# mask：被 mask 的位置（含 nan 和极大值）不影响 loss；多轮工具调用的工具返回段 mask 为 0
rw8 = np.random.randn(8)
new = np.random.randn(8, 6) * 0.1
old = new + np.random.randn(8, 6) * 0.1
ref = new + np.random.randn(8, 6) * 0.1
mk8 = (np.random.rand(8, 6) > 0.4).astype(int)
mk8[:, 0] = 1
mk8[0] = [1, 1, 0, 0, 1, 1]
for nm in ('token', 'sample'):
    base = grpo_loss(rw8, new, old, mk8, 4, logp_ref=ref, kl_coef=0.1, norm=nm)
    dirty = grpo_loss(rw8, np.where(mk8 > 0, new, np.nan), np.where(mk8 > 0, old, 1e3), mk8, 4,
                      logp_ref=np.where(mk8 > 0, ref, -1e3), kl_coef=0.1, norm=nm)
    assert np.isfinite(base) and abs(base - dirty) < 1e-12
# 全 0 mask：不除以 0
assert grpo_loss(rw8, new, old, np.zeros((8, 6)), 4) == 0.0

# on-policy 且无 KL：loss = −Σ(A·mask) / Σmask
adv8 = grpo_advantages(rw8, 4)[:, None]
expect = -(adv8 * mk8).sum() / mk8.sum()
assert abs(grpo_loss(rw8, new, new, mk8, 4) - expect) < 1e-10
