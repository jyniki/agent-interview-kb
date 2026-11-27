'''code-model-010：手写 GAE，输入 rewards、values、next_values、dones、truncated，输出优势和回报目标。

题目：从后往前递推 δ_t = r_t + γV(s_{t+1}) − V(s_t)，A_t = δ_t + γλA_{t+1}；
终止时不自举且切断递推，截断时仍自举但切断递推；R = A + V。
复杂度：时间 O(T)，空间 O(T)。
'''
import numpy as np


def gae(rewards, values, next_values, dones, truncated, gamma=0.99, lam=0.95):
    '''输入均为 (T,) 的 float 数组，dones / truncated 是 0/1。返回 (advantages, returns)。'''
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    nv = np.asarray(next_values, dtype=np.float64)
    d = np.asarray(dones, dtype=np.float64)
    tr = np.asarray(truncated, dtype=np.float64)
    T = r.shape[0]
    adv = np.zeros_like(r)
    last = 0.0
    for t in range(T - 1, -1, -1):
        # 终止：不自举；截断：仍用观测到的 V(s_{t+1}) 自举。两者都切断递推
        delta = r[t] + gamma * nv[t] * (1.0 - d[t]) - v[t]
        last = delta + gamma * lam * (1.0 - d[t]) * (1.0 - tr[t]) * last
        adv[t] = last
    return adv, adv + v


def gae_direct(rewards, values, next_values, dones, truncated, gamma=0.99, lam=0.95):
    '''定义式 Â_t = Σ_l (γλ)^l δ_{t+l}，遇到 done/truncated 就停。用于对照递推版。'''
    r, v, nv = map(lambda a: np.asarray(a, dtype=np.float64), (rewards, values, next_values))
    d, tr = np.asarray(dones, dtype=np.float64), np.asarray(truncated, dtype=np.float64)
    T = r.shape[0]
    delta = r + gamma * nv * (1 - d) - v
    out = np.zeros(T)
    for t in range(T):
        acc, w = 0.0, 1.0
        for l in range(T - t):
            acc += w * delta[t + l]
            if d[t + l] or tr[t + l]:
                break
            w *= gamma * lam
        out[t] = acc
    return out


np.random.seed(0)
T = 12
r = np.random.randn(T) * 0.5
v = np.random.randn(T)
nv = np.random.randn(T)
dones = np.zeros(T)
trunc = np.zeros(T)

# 形状与递推 / 定义式一致
adv, ret = gae(r, v, nv, dones, trunc)
assert adv.shape == (T,) and ret.shape == (T,)
assert np.allclose(adv, gae_direct(r, v, nv, dones, trunc), atol=1e-12)
assert np.allclose(ret, adv + v, atol=1e-12)

# λ = 0：A_t = δ_t（只看一步）
a0, r0 = gae(r, v, nv, dones, trunc, lam=0.0)
assert np.allclose(a0, r + 0.99 * nv - v, atol=1e-12)
# λ = 1 且无边界：A_t = 折扣回报（末尾自举）− V(s_t)。
# 这个恒等式要求 next_values 自洽：next_v[t] = V(s_{t+1})，即 nv[:-1] = v[1:]，
# 最后一步用末尾观测的价值自举。nv 给成独立随机数时它不成立（测试数据必须自洽）
nv_chain = np.concatenate([v[1:], nv[-1:]])
a1, r1 = gae(r, v, nv_chain, dones, trunc, lam=1.0)
g_t = np.zeros(T)
acc = nv_chain[-1]
for t in range(T - 1, -1, -1):
    acc = r[t] + 0.99 * acc
    g_t[t] = acc
assert np.allclose(a1, g_t - v, atol=1e-10)
assert np.allclose(r1, g_t, atol=1e-10)
# 不自洽的 next_values 下这个恒等式不成立，说明验证本身有区分力
assert not np.allclose(gae(r, v, nv, dones, trunc, lam=1.0)[0], g_t - v, atol=1e-3)
# λ 介于两者之间：满足同一递推（A_t = δ_t + γλ(1−d)(1−tr)A_{t+1}），
# 且不必逐项落在 λ=0 / λ=1 两个端点之间（含负奖励时不成立）
a_mid, _ = gae(r, v, nv, dones, trunc, lam=0.5)
delta = r + 0.99 * nv - v
for t in range(T - 1):
    assert abs(a_mid[t] - (delta[t] + 0.99 * 0.5 * a_mid[t + 1])) < 1e-12
assert abs(a_mid[-1] - delta[-1]) < 1e-12

# 终止（done）：不自举、切断递推。t=5 处 done，之后的价值不应影响 A_0…A_5
r2, v2, nv2 = r.copy(), v.copy(), nv.copy()
d2 = dones.copy()
d2[5] = 1.0
a2, _ = gae(r2, v2, nv2, d2, trunc)
assert abs(a2[5] - (r2[5] - v2[5])) < 1e-12                 # A_5 = δ_5 = r_5 − V(s_5)
nv3 = nv2.copy()
nv3[5] = 1e6                                                  # done 处 next_value 不该被用
a3, _ = gae(r2, v2, nv3, d2, trunc)
assert np.allclose(a2, a3, atol=1e-12)
v4 = v2.copy()
v4[6:] = 1e6                                                  # done 之后的状态不影响前面
a4, _ = gae(r2, v4, nv2, d2, trunc)
assert np.allclose(a2[:6], a4[:6], atol=1e-12)

# 截断（truncated）：仍自举，但切断递推。
# 用 next_values[5] 当 bootstrap：A_5 = r_5 + γ·next_v_5 − V_5，A_4 不含 A_5 的贡献
tr2 = trunc.copy()
tr2[4] = 1.0                                                  # 片段在 t=4 结束
a5, _ = gae(r2, v2, nv2, dones, tr2)
d4 = r2[4] + 0.99 * nv2[4] - v2[4]
assert abs(a5[4] - d4) < 1e-12                               # 截断处仍自举：A_4 = δ_4
d3 = r2[3] + 0.99 * nv2[3] - v2[3]
assert abs(a5[3] - (d3 + 0.99 * 0.95 * d4)) < 1e-12          # 不跨过边界的结果
# 递推在边界处切断：A_3 不把 A_4 的后续再乘一次 γλ；不切断时 A_3 会额外含 γλ·γλ·A_5 链
a_uncut, _ = gae(r2, v2, nv2, dones, trunc)
assert abs(a_uncut[3] - a5[3]) > 1e-6
assert abs(a_uncut[4] - a5[4]) > 1e-6                        # A_4 自身也因递推切断而不同
# 截断当 done 处理会把自举项丢掉，价值被系统性低估（差距恰好是 γ·next_v）
b5, _ = gae(r2, v2, nv2, tr2, trunc)                         # 错误做法：把截断当 done
assert abs(a5[4] - b5[4] - 0.99 * nv2[4]) < 1e-12

# 片段末尾 next_v 用最后一个观测的价值估计，不填 0：影响 A_{T−1}
r6, v6 = np.array([1.0]), np.array([0.0])
a7, _ = gae(r6, v6, np.array([1.0]), np.zeros(1), np.zeros(1), gamma=0.5, lam=1.0)
a8, _ = gae(r6, v6, np.zeros(1), np.zeros(1), np.zeros(1), gamma=0.5, lam=1.0)
assert abs(a7[0] - 1.5) < 1e-12 and abs(a8[0] - 1.0) < 1e-12

# 回报目标必须由优势 + 旧价值得到：先归一化优势再算 R 会改变目标（易错点）
lam_a, lam_r = gae(r, v, nv, dones, trunc)
norm_a = (lam_a - lam_a.mean()) / (lam_a.std() + 1e-8)
assert not np.allclose(norm_a + v, lam_r, atol=1e-6)
assert np.allclose(gae(r, v, nv, dones, trunc)[1], lam_r)

# 边界与退化：T = 1；全 done；无奖励时 A = γ·λ 链
a9, _ = gae(np.array([2.0]), np.array([1.0]), np.array([3.0]), np.array([0.0]), np.array([0.0]),
            gamma=0.9, lam=0.8)
assert abs(a9[0] - (2.0 + 0.9 * 3.0 - 1.0)) < 1e-12
a10, _ = gae(np.zeros(5), np.zeros(5), np.zeros(5), np.array([0, 0, 0, 0, 1.0]), np.zeros(5))
assert np.allclose(a10, 0) and np.all(np.isfinite(a10))
