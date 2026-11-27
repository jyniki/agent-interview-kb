'''code-model-006：手写 RMSNorm，输入 (B, T, D)。

题目：y = x / sqrt(mean(x², dim=-1, keepdim=True) + eps) * g，g 初始化为 1；不减均值、没有偏置。
复杂度：时间 O(B·T·D)，空间 O(D)（参数 g）；相对 LayerNorm 少一次均值规约。
'''
import numpy as np


def rms_norm(x, g, eps=1e-6):
    '''x: (B, T, D)；g: (D,)。统计量在最后一维算，平方和升到 FP64 累加，
    避免 FP16/FP32 下 x² 溢出或丢精度。'''
    xf = np.asarray(x)
    ms = np.mean(xf.astype(np.float64) ** 2, axis=-1, keepdims=True)
    y = xf.astype(np.float64) / np.sqrt(ms + eps)
    return (y * np.asarray(g, dtype=np.float64)).astype(x.dtype)


def layer_norm(x, g, beta, eps=1e-6):
    '''对照组：先减均值再除标准差，并有偏置 β。'''
    xf = np.asarray(x, dtype=np.float32)
    mu = xf.mean(axis=-1, keepdims=True)
    var = xf.var(axis=-1, keepdims=True)
    return ((xf - mu) / np.sqrt(var + eps) * g + beta).astype(x.dtype)


np.random.seed(0)
B, T, D = 2, 5, 32
x = np.random.randn(B, T, D).astype(np.float32) * 3
g = np.ones(D, dtype=np.float32)

y = rms_norm(x, g)
assert y.shape == x.shape and y.dtype == np.float32

# g 全 1 时，每个位置的 mean(y²) ≈ 1（keepdim 对了才会是逐位置，而不是整块）
assert np.allclose((y.astype(np.float64) ** 2).mean(axis=-1), 1.0, atol=1e-5)
assert (y.astype(np.float64) ** 2).mean(axis=-1).shape == (B, T)

# 口径是最后一维：同一行内改一个元素会改整行，别的行不受影响
y2 = rms_norm(np.concatenate([x[:1] + np.eye(D, dtype=np.float32)[None, None, 0] * 10, x[1:]], 0), g)
assert not np.allclose(y2[0, 0], y[0, 0], atol=1e-3)
assert np.allclose(y2[1:], y[1:], atol=1e-6)

# 尺度不变：x 乘常数，输出不变（eps 影响可忽略时）
assert np.allclose(rms_norm(x * 7.0, g), y, atol=1e-5)
# 平移不不变：RMSNorm 不减均值，这点和 LayerNorm 不同
assert not np.allclose(rms_norm(x + 5.0, g), y, atol=1e-3)
ln = layer_norm(x, g, np.zeros(D, dtype=np.float32))
assert np.allclose(layer_norm(x + 5.0, g, np.zeros(D, dtype=np.float32)), ln, atol=1e-5)
# LayerNorm 输出零均值，RMSNorm 不是（没有中心化）
assert np.allclose(ln.mean(axis=-1), 0, atol=1e-6)
assert not np.allclose(y.mean(axis=-1), 0, atol=1e-3)
# LayerNorm 无偏置时方差约 1，RMSNorm 是均方约 1
assert np.allclose(ln.var(axis=-1), 1.0, atol=1e-4)

# eps 放在开方里面：全零输入输出为 0，不产生 nan/inf
z = rms_norm(np.zeros((1, 1, D), dtype=np.float32), g)
assert np.all(np.isfinite(z)) and np.all(z == 0)
# 确认口径是 sqrt(mean + eps) 而不是 sqrt(mean) + eps：
# 取 x 很小、eps 起作用的输入，两种写法才分得开
xs = np.array([[[1e-4, 1e-4]]], dtype=np.float32)
gg = np.ones(2, dtype=np.float32)
assert np.allclose(rms_norm(xs, gg, eps=1e-2),
                   xs / np.sqrt(np.mean(xs * xs, axis=-1, keepdims=True) + 1e-2), atol=1e-7)
assert not np.allclose(rms_norm(xs, gg, eps=1e-2),
                       xs / (np.sqrt(np.mean(xs * xs, axis=-1, keepdims=True)) + 1e-2), atol=1e-7)
# eps 的作用：x 很小时被压小，不会被放大到爆炸
tiny = rms_norm(np.full((1, 1, 4), 1e-4, dtype=np.float32), np.ones(4, dtype=np.float32), eps=1e-6)
assert np.all(tiny < 1.0) and np.all(np.isfinite(tiny))

# g 参与缩放：g 的每个分量独立作用
g2 = np.linspace(0.5, 2.0, D).astype(np.float32)
yg = rms_norm(x, g2)
assert np.allclose(yg, y * g2, atol=1e-6)

# 混合精度：FP16 直接平方会溢出，先升 FP32 就没事
xh = np.array([[[300.0, -250.0, 200.0, 100.0]]], dtype=np.float16)
with np.errstate(all='ignore'):
    naive = (xh * xh)                                   # 9e4 → inf in fp16
assert not np.all(np.isfinite(naive.astype(np.float64)))
yh = rms_norm(xh, np.ones(4, dtype=np.float16))
assert np.all(np.isfinite(yh.astype(np.float64)))
assert np.allclose((yh.astype(np.float64) ** 2).mean(axis=-1), 1.0, atol=1e-2)

# 极端输入不产生 nan/inf
for bad_x in (np.full((1, 2, D), 1e30, dtype=np.float32),
              np.full((1, 2, D), -1e30, dtype=np.float32),
              np.full((1, 2, D), 1e-30, dtype=np.float32)):
    out = rms_norm(bad_x, g)
    assert np.all(np.isfinite(out.astype(np.float64)))

# Pre-Norm 与 Post-Norm：残差主干是否经过归一化
def sublayer(t):
    return np.tanh(t)                  # 代表任意子层（注意力 / FFN）

pre = x + sublayer(rms_norm(x, g))     # Pre-Norm：x + f(norm(x))，主干不经过归一化
post = rms_norm(x + sublayer(x), g)    # Post-Norm：norm(x + f(x))，整体再归一化
# Pre-Norm 的输出没有逐位置单位均方，Post-Norm 有
assert not np.allclose((pre.astype(np.float64) ** 2).mean(axis=-1), 1.0, atol=1e-3)
assert np.allclose((post.astype(np.float64) ** 2).mean(axis=-1), 1.0, atol=1e-5)
assert not np.allclose(pre, post, atol=1e-3)
