'''code-model-003：手写 LoRA 线性层。

题目：冻结原权重 W，旁路加两个低秩矩阵 A (r×in)、B (out×r)，输出 Wx + (α/r)·BAx；B 初始化为 0。
复杂度：设 N 个 token，前向时间 O(N·in·out + N·r·(in+out))，可训练参数空间 O(r·(in+out))；合并后推理零额外开销。
'''
import numpy as np


def lora_init(in_features, out_features, r, rng):
    '''A 随机、B 全 0：训练开始时旁路输出为 0。'''
    A = rng.standard_normal((r, in_features)) / np.sqrt(in_features)
    B = np.zeros((out_features, r))
    return A, B


def lora_forward(x, W, A, B, alpha):
    '''x: (..., in)；W: (out, in)；返回 (..., out)。'''
    scale = alpha / A.shape[0]
    return x @ W.T + scale * ((x @ A.T) @ B.T)


def lora_merge(W, A, B, alpha):
    '''推理时把旁路并回 W：先用 (α/r)·BA 得到 delta，再叠加到 W 上。'''
    return W + (alpha / A.shape[0]) * (B @ A)


def lora_backward(x, grad_out, A, B, alpha):
    '''只返回 A、B 的梯度；W 冻结，不产生也不返回梯度。'''
    scale = alpha / A.shape[0]
    x2 = x.reshape(-1, x.shape[-1])
    g2 = grad_out.reshape(-1, grad_out.shape[-1])
    dB = scale * g2.T @ (x2 @ A.T)
    dA = scale * (g2 @ B).T @ x2
    return dA, dB


def lora_forward_multi(x, W, adapters, adapter_ids, alpha):
    '''多 LoRA 服务：x (N, in)，adapters 为 [(A, B)]，adapter_ids (N,)。
    底座共享，每个请求按 adapter ID 选 A、B，批内不同 adapter 分组计算。'''
    out = x @ W.T
    for a in np.unique(adapter_ids):
        idx = np.where(adapter_ids == a)[0]
        A, B = adapters[a]
        out[idx] += (alpha / A.shape[0]) * ((x[idx] @ A.T) @ B.T)
    return out


def _loss(x, y, W, A, B, alpha):
    d = lora_forward(x, W, A, B, alpha) - y
    return 0.5 * (d ** 2).sum(), d


rng = np.random.default_rng(0)
in_f, out_f, r, alpha = 12, 10, 4, 8.0
W = rng.standard_normal((out_f, in_f))
W.setflags(write=False)                     # W 冻结：任何写 W 的实现都会直接报错
A, B = lora_init(in_f, out_f, r, rng)
assert A.shape == (r, in_f) and B.shape == (out_f, r)
assert np.all(B == 0)

x3 = rng.standard_normal((3, 5, in_f))
x2 = x3.reshape(-1, in_f)

# 形状与 B = 0 时输出与原模型一致
assert lora_forward(x3, W, A, B, alpha).shape == (3, 5, out_f)
assert np.allclose(lora_forward(x3, W, A, B, alpha), x3 @ W.T, atol=1e-12)

# 合并等价：合并后不再有延迟
B1 = rng.standard_normal((out_f, r)) * 0.1
assert np.allclose(lora_forward(x2, W, A, B1, alpha), x2 @ lora_merge(W, A, B1, alpha).T, atol=1e-10)
assert np.allclose(lora_merge(W, A, B, alpha), W)          # B = 0 时合并就是原样

# 缩放 α/r：delta 关于 alpha 线性，且确实除以 r 而不是 1
d1 = lora_forward(x2, W, A, B1, alpha) - x2 @ W.T
d2 = lora_forward(x2, W, A, B1, 2 * alpha) - x2 @ W.T
assert np.allclose(d2, 2 * d1, atol=1e-10)
assert np.allclose(d1, (alpha / r) * (x2 @ A.T) @ B1.T, atol=1e-10)
A_s = rng.standard_normal((8, in_f)) / np.sqrt(in_f)
B_s = rng.standard_normal((out_f, 8)) * 0.1
d_s = lora_forward(x2, W, A_s, B_s, alpha) - x2 @ W.T
assert np.allclose(d_s, (alpha / A_s.shape[0]) * (x2 @ A_s.T) @ B_s.T, atol=1e-6)
assert not np.allclose(d_s, (alpha / 4) * (x2 @ A_s.T) @ B_s.T, atol=1e-9)

# 冻结原权重：前向和反向都不需要 W 的梯度，W 保持只读
assert not W.flags.writeable

# 梯度：数值梯度校验（B ≠ 0 时 A 的梯度才非零）
y = rng.standard_normal((x2.shape[0], out_f))
alpha_t = 2.0
A2, B2 = lora_init(in_f, out_f, r, rng)
B2 = rng.standard_normal((out_f, r)) * 0.1
_, g = _loss(x2, y, W, A2, B2, alpha_t)
dA, dB = lora_backward(x2, g, A2, B2, alpha_t)
assert dA.shape == A2.shape and dB.shape == B2.shape
h = 1e-6
num = np.zeros_like(dA)
for i in range(A2.shape[0]):
    for j in range(A2.shape[1]):
        old = A2[i, j]
        A2[i, j] = old + h
        lp, _ = _loss(x2, y, W, A2, B2, alpha_t)
        A2[i, j] = old - h
        lm, _ = _loss(x2, y, W, A2, B2, alpha_t)
        A2[i, j] = old
        num[i, j] = (lp - lm) / (2 * h)
assert np.allclose(dA, num, atol=1e-6)
num = np.zeros_like(dB)
for i in range(B2.shape[0]):
    for j in range(B2.shape[1]):
        old = B2[i, j]
        B2[i, j] = old + h
        lp, _ = _loss(x2, y, W, A2, B2, alpha_t)
        B2[i, j] = old - h
        lm, _ = _loss(x2, y, W, A2, B2, alpha_t)
        B2[i, j] = old
        num[i, j] = (lp - lm) / (2 * h)
assert np.allclose(dB, num, atol=1e-6)

# 初始化处的梯度：B = 0 时 B 有梯度（旁路能启动），但 A 的梯度也为 0；
# A、B 全零则两边梯度都是 0，训练永远不动（题目追问 1）
A0, B0 = lora_init(in_f, out_f, r, rng)
_, g0 = _loss(x2, y, W, A0, B0, alpha_t)
dA0, dB0 = lora_backward(x2, g0, A0, B0, alpha_t)
assert np.all(dA0 == 0) and np.any(dB0 != 0)
Z = np.zeros((r, in_f))
dAz, dBz = lora_backward(x2, g0, Z, B0, alpha_t)
assert np.all(dAz == 0) and np.all(dBz == 0)
# 一次更新后 A 的梯度恢复非零
B_after = B0 - 0.01 * dB0
dA_after, _ = lora_backward(x2, g0, A0, B_after, alpha_t)
assert np.any(dA_after != 0)

# 多 LoRA：批内混排不同 adapter，结果等于逐条单独计算
adapters = [(lora_init(in_f, out_f, r, rng)[0], rng.standard_normal((out_f, r)) * 0.1) for _ in range(3)]
ids = np.array([0, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1])
out_m = lora_forward_multi(x2, W, adapters, ids, alpha)
for i in range(x2.shape[0]):
    A_i, B_i = adapters[ids[i]]
    assert np.allclose(out_m[i], lora_forward(x2[i], W, A_i, B_i, alpha), atol=1e-10)
