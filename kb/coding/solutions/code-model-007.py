'''code-model-007：手写带 KV Cache 的自回归解码循环。

题目：prefill 把整段 prompt 过一遍，每层存下 K、V；之后每步只喂上一步的新 token，
把 K、V 追加到缓存，用新 Q 和全部缓存做注意力，采样下一个 token，直到 EOS 或最大长度。
复杂度：prefill 时间 O(B·T_p²·d)；decode 每步 O(B·T_cache·d)，总共 O(B·T_new·T_cache·d)；
空间 O(L·B·n_kv·T_cache·d)（每层一份 K、V，不含 Q）。
'''
import numpy as np

PAD = 0


def softmax(x, axis=-1):
    '''稳定 softmax；全 -inf 的行返回全 0，不产生 nan。'''
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(x - m)
    s = e.sum(axis=axis, keepdims=True)
    return e / np.where(s > 0, s, 1.0)


def rope_tables(positions, d, base=10000.0):
    inv = base ** (-np.arange(0, d, 2) / d)
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang), np.sin(ang)


def rope_apply(x, cos, sin):
    '''x: (B, h, T, d)；cos/sin: (T, d/2)。相邻配对。'''
    c = np.repeat(cos[None, None], 2, axis=-1)
    s = np.repeat(sin[None, None], 2, axis=-1)
    rot = np.stack([-x[..., 1::2], x[..., 0::2]], axis=-1).reshape(x.shape)
    return x * c + rot * s


class TinyLM:
    '''最简 decoder-only 模型：embedding + (RoPE attention + FFN) × L，权重绑定输出头。'''

    def __init__(self, vocab=32, d_model=16, n_heads=4, n_layers=2, pad_id=PAD, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab, self.D, self.h, self.L = vocab, d_model, n_heads, n_layers
        self.d_k = d_model // n_heads
        self.pad_id = pad_id
        self.E = rng.standard_normal((vocab, d_model)) * 0.3
        self.W = [{k: rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
                   for k in ('q', 'k', 'v', 'o')} for _ in range(n_layers)]

    def forward(self, ids, cache=None, pos=None, use_causal=True):
        '''ids: (B, T) 或 (T,)。cache: {'kv': [(k, v)] * L, 'pad': (B, T_cache), 'len': T_cache}。
        返回 (logits (B, T, V), 新 cache)。'''
        ids = np.atleast_2d(ids)
        B, T = ids.shape
        past = 0 if cache is None else cache['len']
        pad = np.zeros((B, past + T), dtype=bool)
        if cache is not None:
            pad[:, :past] = cache['pad']
        pad[:, past:] = ids != self.pad_id
        # 位置编码用绝对位置：decode 时是 T_cache，不是 0
        positions = np.arange(past, past + T) if pos is None else pos + np.arange(T)
        cos, sin = rope_tables(positions, self.d_k)

        x = self.E[ids]
        kv = []
        for li, w in enumerate(self.W):
            q, k, v = x @ w['q'], x @ w['k'], x @ w['v']
            q = rope_apply(q.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3), cos, sin)
            k = rope_apply(k.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3), cos, sin)
            v = v.reshape(B, T, self.h, self.d_k).transpose(0, 2, 1, 3)
            if cache is not None:
                k = np.concatenate([cache['kv'][li][0], k], axis=2)
                v = np.concatenate([cache['kv'][li][1], v], axis=2)
            kv.append((k, v))
            Tc = k.shape[2]
            scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
            if use_causal and T > 1:
                future = np.arange(Tc)[None, :] > (np.arange(T)[:, None] + Tc - T)
                scores = np.where(future[None, None], -np.inf, scores)
            scores = np.where(pad[:, None, None, :], scores, -np.inf)   # 左填充的 pad 不可见
            y = softmax(scores) @ v
            x = x + y.transpose(0, 2, 1, 3).reshape(B, T, self.D) @ w['o']
        logits = x @ self.E.T
        return logits, {'kv': kv, 'pad': pad, 'len': past + T}


def generate(model, prompt, max_new, eos_id, rng=None):
    '''prefill 一次，再逐 token decode 到 EOS 或最大长度。返回每条序列的 token 列表。'''
    ids = np.atleast_2d(np.asarray(prompt))
    B = ids.shape[0]
    out = [list(int(t) for t in row) for row in ids]
    logits, cache = model.forward(ids, None)
    finished = np.zeros(B, dtype=bool)
    for _ in range(max_new):
        nxt = logits[:, -1].argmax(axis=-1) if rng is None else \
            np.array([rng.choice(model.vocab, p=softmax(logits[b, -1])) for b in range(B)])
        for b in range(B):
            if not finished[b]:
                out[b].append(int(nxt[b]))
                finished[b] = nxt[b] == eos_id
        if finished.all():
            break
        feed = np.where(finished, model.pad_id, nxt)[:, None]   # 结束的序列只喂 pad
        logits, cache = model.forward(feed, cache)
    return out


np.random.seed(0)
V, Tp, T_all = 32, 4, 9
model = TinyLM(vocab=V, d_model=16, n_heads=4, n_layers=2, seed=0)
ids = np.random.randint(1, V, size=(2, T_all))

# 缓存形状：每层一份 K、V，(B, h, T_cache, d_h)；不含 Q
_, cache = model.forward(ids[:, :Tp])
assert len(cache['kv']) == model.L
assert cache['kv'][0][0].shape == (2, model.h, Tp, model.d_k)
assert cache['kv'][0][1].shape == (2, model.h, Tp, model.d_k)
assert set(cache) == {'kv', 'pad', 'len'} and cache['len'] == Tp

# 核心验证：逐步 decode 与整段带因果 mask 一次前向逐位置一致（容差 1e-6）
full, _ = model.forward(ids)
logits, cache = model.forward(ids[:, :Tp])
steps = [logits]
for t in range(Tp, T_all):
    logits, cache = model.forward(ids[:, t:t + 1], cache)
    assert cache['len'] == t + 1 and cache['kv'][0][0].shape[2] == t + 1
    steps.append(logits)
step_logits = np.concatenate(steps, axis=1)
assert step_logits.shape == (2, T_all, V)
assert np.allclose(step_logits, full, atol=1e-6)
assert np.all(np.isfinite(step_logits))

# 不加因果 mask 的整段前向和加 mask 的不同（mask 真的起作用）；decode 时加不加一样
assert not np.allclose(model.forward(ids, use_causal=False)[0], full, atol=1e-3)
_, c1 = model.forward(ids[:, :T_all - 1])
a = model.forward(ids[:, T_all - 1:], c1, use_causal=True)[0]
b = model.forward(ids[:, T_all - 1:], c1, use_causal=False)[0]
assert np.allclose(a, b, atol=1e-10) and np.allclose(a, full[:, -1:], atol=1e-6)

# 位置编码用绝对位置：从 0 重新开始会算出不同的结果（易错点）
c2 = c1
wrong, _ = model.forward(ids[:, T_all - 1:], c2, pos=0)
assert not np.allclose(wrong, full[:, -1:], atol=1e-6)

# 批内左填充：与逐条去掉 pad 单独前向一致（RoPE 只看相对位置，pad 又不参与注意力）
lens = [2, 5]
padded = np.full((2, max(lens)), PAD)
for i, n in enumerate(lens):
    padded[i, max(lens) - n:] = ids[i, :n]          # 左填充
lp_logits, lp_cache = model.forward(padded)
assert lp_cache['pad'].sum() == sum(lens)
for i, n in enumerate(lens):
    single, _ = model.forward(ids[i, :n])
    assert np.allclose(lp_logits[i, max(lens) - n:], single[0], atol=1e-9)
# 同一个模型下，短序列 last logits 与单独前向一致
assert np.allclose(lp_logits[0, -1], model.forward(ids[0, :2])[0][0, -1], atol=1e-9)

# 生成循环：EOS 立即停（抬高 token 9 的输出行，保证贪心第一步就吐 eos）
E_BACKUP = model.E.copy()
model.E[9] = model.E[9] + 40.0                    # 让 token 9 的 logit 压倒性最大
res = generate(model, ids[:1, :3], max_new=5, eos_id=9)
assert res[0][:3] == list(ids[0, :3]) and len(res[0]) == 4 and res[0][-1] == 9
assert 9 not in res[0][3:-1]                      # 输出 EOS 之后不再追加（易错点）
res2 = generate(model, ids[:1, :3], max_new=5, eos_id=999)   # 永不 EOS，跑到上限
assert len(res2[0]) == 3 + 5
model.E = E_BACKUP                                # 还原，后面的假模型测试不依赖它
assert np.array_equal(model.E, E_BACKUP)


class _Scripted:
    '''固定脚本的假模型，用来测循环的 EOS / pad 记账。'''
    pad_id = PAD
    vocab = 10

    def __init__(self, script):
        self.script, self.t = script, 0
        self.feeds = []

    def forward(self, ids, cache=None, **kw):
        ids = np.atleast_2d(ids)
        B, T = ids.shape
        logits = np.zeros((B, T, self.vocab))
        if cache is None:                           # prefill：吐出脚本里的第一个 token
            for b in range(B):
                logits[b, -1, self.first[b]] = 10.0
        else:                                       # decode 步：推进脚本
            self.feeds.append(ids.copy())
            for b in range(B):
                logits[b, -1, self.script[b][self.t]] = 10.0
            self.t += 1
        return logits, {'kv': [], 'pad': np.ones((B, T), bool), 'len': T if cache is None else cache['len'] + T}


script = [[9, 9, 9], [8, 9, 9]]
sm = _Scripted(script)
sm.first = [5, 7]
out = generate(sm, [[1, 2], [3, 4]], max_new=5, eos_id=9)
assert out[0] == [1, 2, 5, 9]                       # 第一条在 EOS 处截断
assert out[1] == [3, 4, 7, 8, 9]
assert len(sm.feeds) == 2                           # 两条都结束后不再 decode
assert sm.feeds[1][0, 0] == PAD                     # 已结束的序列只喂 pad
assert sm.feeds[1][1, 0] == 8                       # 未结束的照常喂上一步 token
assert all(t != PAD for t in out[1])
# 输出 EOS 之后不再追加（易错点）：第一条不出现脚本里 EOS 之后的 token
