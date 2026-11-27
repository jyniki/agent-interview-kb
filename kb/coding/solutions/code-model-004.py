'''code-model-004：手写 DPO loss。

题目：loss = −log σ(β·[(logπ(y_w) − logπ_ref(y_w)) − (logπ(y_l) − logπ_ref(y_l))])。
复杂度：设 batch 大小 B、回答长度 T：时间 O(B·T)，空间 O(B·T)（若离线预算好 logprob，在线只剩 O(B)）。
'''
import numpy as np


def sequence_logprob(token_logps, mask):
    '''整条回答的 logπ(y)：各 token logprob 在 response mask 上求和（不是求平均）。'''
    m = np.asarray(mask) > 0
    return (np.asarray(token_logps) * m).sum(axis=-1)


def dpo_loss(logp_chosen, logp_rejected, ref_logp_chosen, ref_logp_rejected, beta):
    '''输入为整条回答的 logπ、logπ_ref（标量或 (B,)）。
    返回 (loss, margin, chosen_reward, rejected_reward)。'''
    margin = beta * ((logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected))
    # −log σ(x) = log(1 + e^{−x})，用 logaddexp 保证大 |x| 时不溢出、不下溢
    loss = np.logaddexp(0.0, -margin)
    return loss, margin, beta * (logp_chosen - ref_logp_chosen), beta * (logp_rejected - ref_logp_rejected)


def dpo_grad_margin(margin):
    '''d loss / d margin = −σ(−margin) = −e^{−log(1+e^m)}，在 |margin| 很大时也稳定。'''
    return -np.exp(-np.logaddexp(0.0, margin))


def logsigmoid(x):
    return -np.logaddexp(0.0, -x)


np.random.seed(0)

# 与定义式一致（中段数值上用 log(σ) 直接对照）
m = np.array([-3.0, -0.5, 0.0, 0.7, 2.5])
assert np.allclose(np.logaddexp(0.0, -m), -np.log(1.0 / (1.0 + np.exp(-m))), atol=1e-12)
assert np.allclose(-np.logaddexp(0.0, -m), logsigmoid(m).real, atol=1e-12)
assert np.allclose(np.logaddexp(0.0, -m), -logsigmoid(m))

# 参考模型 = 策略：margin = 0，loss = log 2
lp = np.random.randn(4) * 0.1
l0, m0, rc, rj = dpo_loss(lp, lp, lp, lp, beta=0.5)
assert np.allclose(l0, np.log(2.0)) and np.all(m0 == 0)

# 手算一组：chosen 得分 2、rejected 得分 −1、β = 0.1 → margin = 0.3
l1, m1, rc1, rj1 = dpo_loss(2.0, -1.0, 0.0, 0.0, 0.1)
assert abs(m1 - 0.3) < 1e-12 and abs(rc1 - 0.2) < 1e-12 and abs(rj1 + 0.1) < 1e-12
assert abs(l1 - (-np.log(1 / (1 + np.exp(-0.3))))) < 1e-12

# β 的作用：β 越大 loss 随 margin 变化越陡；β → 0 时退化为 log 2
assert dpo_grad_margin(np.array(0.3)) == dpo_grad_margin(np.array(0.3))
assert abs(dpo_loss(2.0, -1.0, 0.0, 0.0, 1e-9)[0] - np.log(2.0)) < 1e-7
assert dpo_loss(3.0, -3.0, 0.0, 0.0, 0.1)[0] > dpo_loss(3.0, -3.0, 0.0, 0.0, 1.0)[0]

# 对称性：交换 chosen / rejected，loss 差一个 margin（L(−x) = L(x) + x）
x = np.random.randn(5)
assert np.allclose(np.logaddexp(0.0, -(-x)), np.logaddexp(0.0, -x) + x, atol=1e-12)
# 偏好排序正确时 loss 单调下降，排序反了 loss 上升
assert dpo_loss(1.0, 0.0, 0.0, 0.0, 0.5)[0] < np.log(2.0)
assert dpo_loss(0.0, 1.0, 0.0, 0.0, 0.5)[0] > np.log(2.0)

# 梯度（对 margin）数值校验
eps = 1e-6
for t in m:
    lp_p = np.logaddexp(0.0, -(t + eps))
    lp_m = np.logaddexp(0.0, -(t - eps))
    assert abs((lp_p - lp_m) / (2 * eps) - dpo_grad_margin(np.array(t))) < 1e-8

# 数值稳定：极端 margin 不产生 nan / inf
ext = np.array([-1e4, -800.0, 800.0, 1e4])
v = np.logaddexp(0.0, -ext)
assert np.all(np.isfinite(v))
assert np.allclose(v, np.maximum(0.0, -ext), atol=1e-12)   # 饱和时 loss = max(0, −x)
g = dpo_grad_margin(ext)
assert np.all(np.isfinite(g)) and g[0] == -1.0 and g[3] == 0.0

# mask 上的求和：padding 不参与；用 mean 会改变目标（易错点）
tok = np.random.randn(1, 6) * 0.1
mask = np.array([[1, 1, 1, 1, 0, 0]])
lp_sum = sequence_logprob(tok, mask)
assert abs(lp_sum - tok[:, :4].sum()) < 1e-12
tok_pad = np.concatenate([tok, np.full((1, 3), -1e3)], axis=1)
mask_pad = np.concatenate([mask, np.zeros((1, 3), dtype=int)], axis=1)
assert abs(sequence_logprob(tok_pad, mask_pad) - lp_sum) < 1e-12
assert sequence_logprob(tok, mask).shape == (1,)

# 长度偏差：同样的每 token 平均 logprob，更长的那条 logπ 绝对值更大
lp_long = np.full((1, 10), -0.1)
lp_short = np.full((1, 2), -0.1)
s_long = sequence_logprob(lp_long, np.ones((1, 10)))
s_short = sequence_logprob(lp_short, np.ones((1, 2)))
assert s_long < s_short
assert abs(sequence_logprob(lp_long, np.ones((1, 10))).item() - 10 * (-0.1)) < 1e-12   # 求和，随长度增长
assert abs(lp_long.ravel().mean() - (-0.1)) < 1e-12                                    # 求平均，与长度无关
assert not np.isclose(sequence_logprob(lp_long, np.ones((1, 10))).item(), lp_long.mean())

# 整条 batch 一起算，与逐条循环一致
B = 4
pc, pr, qc, qr = (np.random.randn(B) for _ in range(4))
lb, mb, _, _ = dpo_loss(pc, pr, qc, qr, 0.3)
for i in range(B):
    li, mi, _, _ = dpo_loss(pc[i], pr[i], qc[i], qr[i], 0.3)
    assert abs(lb[i] - li) < 1e-12 and abs(mb[i] - mi) < 1e-12
