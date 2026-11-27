'''code-model-009：手写 PPO 的 clip 策略 loss 和价值 loss。

题目：ratio 为新旧概率比，策略 loss 取 −min(ratio·A, clip(ratio)·A)；价值 loss 为 (V−R)²，
可对 V 做 clip；都按 mask 平均。输入形状都是 (B, T)。
复杂度：时间 O(B·T)，空间 O(B·T)。
'''
import numpy as np


def ppo_loss(logp_new, logp_old, adv, v_new, v_old, ret, mask, clip_eps=0.2,
             vf_clip=0.2, vf_coef=0.5, entropy=None, ent_coef=0.0, adv_norm=True):
    '''返回 dict：pg_loss、v_loss、entropy、total、clipfrac、ratio。
    total = pg_loss + vf_coef * v_loss − ent_coef * entropy。'''
    m = np.asarray(mask) > 0
    logp_new = np.asarray(logp_new, dtype=np.float64)
    logp_old = np.asarray(logp_old, dtype=np.float64)      # 来自 rollout，必须视为常数
    a = np.asarray(adv, dtype=np.float64)
    if adv_norm and m.any():                               # 只在有效 token 上做均值方差归一化
        mu = a[m].mean()
        sd = a[m].std() + 1e-8
        a = (a - mu) / sd
    ratio = np.exp(np.clip(np.where(m, logp_new - logp_old, 0.0), -80, 80))
    unclipped = ratio * a
    clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps) * a
    pg = -np.where(m, np.minimum(unclipped, clipped), 0.0)
    denom = max(m.sum(), 1)
    pg_loss = pg.sum() / denom                             # 按有效 token 平均
    v = np.asarray(v_new, dtype=np.float64)
    v0 = np.asarray(v_old, dtype=np.float64)
    v_clip = v0 + np.clip(v - v0, -vf_clip, vf_clip)
    r = np.asarray(ret, dtype=np.float64)
    v_loss_pair = 0.5 * np.maximum((v - r) ** 2, (v_clip - r) ** 2)   # 取更差的那个，悲观下界
    v_loss = np.where(m, v_loss_pair, 0.0).sum() / denom
    ent = 0.0 if entropy is None else float(np.asarray(entropy)[m].sum() / denom)
    total = pg_loss + vf_coef * v_loss - ent_coef * ent
    clipfrac = float(((np.abs(ratio - 1) > clip_eps) & m).sum() / denom)
    return {'pg_loss': float(pg_loss), 'v_loss': float(v_loss), 'entropy': ent,
            'total': float(total), 'clipfrac': clipfrac, 'ratio': ratio}


def _pg_only(logp_new, logp_old, adv, mask, eps):
    return ppo_loss(logp_new, logp_old, adv, np.zeros_like(logp_new), np.zeros_like(logp_new),
                    np.zeros_like(logp_new), mask, clip_eps=eps, adv_norm=False)['pg_loss']


def grad_logp_new(logp_new, *args, h=1e-6):
    '''对 logp_new 的数值（中心差分）梯度，形状同输入。'''
    g = np.zeros_like(logp_new)
    for idx in np.ndindex(*logp_new.shape):
        p = logp_new.copy()
        p[idx] += h
        lp = _pg_only(p, *args)
        p[idx] -= 2 * h
        lm = _pg_only(p, *args)
        g[idx] = (lp - lm) / (2 * h)
    return g


np.random.seed(0)
B, T = 4, 6
mask = (np.random.rand(B, T) > 0.3).astype(int)
mask[:, 0] = 1
logp_old = np.random.randn(B, T) * 0.2
logp_new = logp_old + np.random.randn(B, T) * 0.05
adv = np.random.randn(B, T)
ret = np.random.randn(B, T)
v_new = ret + np.random.randn(B, T) * 0.1
v_old = v_new + np.random.randn(B, T) * 0.05

# 形状与 key
out = ppo_loss(logp_new, logp_old, adv, v_new, v_old, ret, mask)
assert set(out) == {'pg_loss', 'v_loss', 'entropy', 'total', 'clipfrac', 'ratio'}
assert out['ratio'].shape == (B, T)
assert all(np.isfinite(v) for k, v in out.items() if k != 'ratio')

# ratio = 1（on-policy）时，pg_loss = −mean(A)，clipfrac = 0
lp0 = np.random.randn(B, T) * 0.3
o = ppo_loss(lp0, lp0, adv, v_new, v_old, ret, mask)
assert abs(o['pg_loss']) < 1e-12                    # 归一化后的优势在有效 token 上均值为 0
o_raw = ppo_loss(lp0, lp0, adv, v_new, v_old, ret, mask, adv_norm=False)
assert abs(o_raw['pg_loss'] - (-adv[mask > 0].mean())) < 1e-12
assert o['clipfrac'] == 0.0
assert np.allclose(o['ratio'][mask > 0], 1.0)

# 手算：单 token，A = 1，ratio = 2 → 取 clip 的 1.2，pg_loss = −1.2
one = np.ones((1, 1))
o1 = ppo_loss(np.array([[np.log(2.0)]]), np.zeros((1, 1)), np.ones((1, 1)),
              np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), one, adv_norm=False)
assert abs(o1['pg_loss'] + 1.2) < 1e-12 and o1['clipfrac'] == 1.0
# A = −1，ratio = 2 → min 取未截断的 −2 → pg_loss = 2
o2 = ppo_loss(np.array([[np.log(2.0)]]), np.zeros((1, 1)), np.array([[-1.0]]),
              np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), one, adv_norm=False)
assert abs(o2['pg_loss'] - 2.0) < 1e-12
# A = 1，ratio = 0.5 → 取 0.5 → pg_loss = −0.5
o3 = ppo_loss(np.array([[np.log(0.5)]]), np.zeros((1, 1)), np.ones((1, 1)),
              np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), one, adv_norm=False)
assert abs(o3['pg_loss'] + 0.5) < 1e-12

# 梯度：A > 0 且 ratio > 1+ε 时对 logp_new 的梯度为 0；A > 0 且 ratio < 1−ε 时不为 0
lp = np.array([[np.log(2.0)]])
g = grad_logp_new(lp, np.zeros((1, 1)), np.ones((1, 1)), one, 0.2)
assert abs(g[0, 0]) < 1e-8
lp_low = np.array([[np.log(0.5)]])
g_low = grad_logp_new(lp_low, np.zeros((1, 1)), np.ones((1, 1)), one, 0.2)
assert abs(g_low[0, 0] + 0.5) < 1e-6          # d/dlogp_new(−r·A) = −r·A = −0.5
# A < 0 且 ratio > 1+ε 时，min 取未截断分支（−2 比 −1.2 更小），变坏的方向保留梯度：
# d/dlogp_new(−ratio·A) = −ratio·A = +2
g_neg = grad_logp_new(lp, np.zeros((1, 1)), np.array([[-1.0]]), one, 0.2)
assert abs(g_neg[0, 0] - 2.0) < 1e-6
# 区间内 (1−ε, 1+ε)：梯度 = −r·A
lp_mid = np.array([[np.log(1.05)]])
g_mid = grad_logp_new(lp_mid, np.zeros((1, 1)), np.ones((1, 1)), one, 0.2)
assert abs(g_mid[0, 0] + 1.05) < 1e-6

# 价值 clip：|v − v_old| > ε 时用 v_clip；再远也不会让 loss 变差（取 max）
o4 = ppo_loss(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
              np.array([[3.0]]), np.zeros((1, 1)), np.array([[0.0]]), np.ones((1, 1)),
              vf_clip=0.2, adv_norm=False)
# v = 3、v_old = 0、R = 0：v_clip = 0.2，max((3)², (0.2)²) 仍取未截断项，v_loss = 0.5·9
assert abs(o4['v_loss'] - 0.5 * 3.0 ** 2) < 1e-12
o5 = ppo_loss(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)),
              np.array([[0.1]]), np.zeros((1, 1)), np.array([[0.0]]), np.ones((1, 1)),
              vf_clip=0.2, adv_norm=False)
assert abs(o5['v_loss'] - 0.5 * 0.01) < 1e-12              # 未越界，等于 (v−R)²/2
assert o4['v_loss'] <= 0.5 * 3.0 ** 2 + 1e-12              # 悲观下界不会更差

# mask：被 mask 的位置不影响任何 loss；全 0 mask 不除以 0
dirty = np.where(mask > 0, logp_new, np.nan)
o6 = ppo_loss(dirty, logp_old, adv, np.where(mask > 0, v_new, 1e9), v_old,
              np.where(mask > 0, ret, -1e9), mask, adv_norm=False)
o7 = ppo_loss(np.where(mask > 0, logp_new, 0.0), logp_old, adv,
              np.where(mask > 0, v_new, 0.0), v_old, np.where(mask > 0, ret, 0.0), mask, adv_norm=False)
assert abs(o6['pg_loss'] - o7['pg_loss']) < 1e-12 and abs(o6['v_loss'] - o7['v_loss']) < 1e-12
oz = ppo_loss(logp_new, logp_old, adv, v_new, v_old, ret, np.zeros((B, T)))
assert oz['pg_loss'] == 0.0 and oz['v_loss'] == 0.0

# 聚合口径是 token 级平均：把同一个有效 token 复制一份，loss 不变
rep_ok = ppo_loss(np.repeat(logp_new, 2, axis=0), np.repeat(logp_old, 2, axis=0),
                  np.repeat(adv, 2, axis=0), np.repeat(v_new, 2, axis=0),
                  np.repeat(v_old, 2, axis=0), np.repeat(ret, 2, axis=0),
                  np.repeat(mask, 2, axis=0), adv_norm=False)
base = ppo_loss(logp_new, logp_old, adv, v_new, v_old, ret, mask, adv_norm=False)
assert abs(rep_ok['pg_loss'] - base['pg_loss']) < 1e-12

# 总目标：pg + c1·vf − c2·entropy
o8 = ppo_loss(logp_new, logp_old, adv, v_new, v_old, ret, mask,
              vf_coef=0.5, entropy=np.abs(np.random.randn(B, T)), ent_coef=0.01)
assert abs(o8['total'] - (o8['pg_loss'] + 0.5 * o8['v_loss'] - 0.01 * o8['entropy'])) < 1e-12

# logp_old 必须是常数：用同一份旧 logprob 反复算，结果相同（重新计算会让 ratio 恒为 1）
o9 = ppo_loss(logp_new, logp_old, adv, v_new, v_old, ret, mask, adv_norm=False)
assert abs(o9['pg_loss'] - base['pg_loss']) < 1e-15
# 数值稳定：极端 logp 差（ratio 巨大 / 极小）不溢出
o10 = ppo_loss(np.array([[800.0, -800.0]]), np.zeros((1, 2)), np.array([[1.0, -1.0]]),
               np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2)), adv_norm=False)
assert np.all(np.isfinite(o10['ratio'])) and np.isfinite(o10['pg_loss'])
