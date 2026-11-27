'''code-model-013：手写带 loss mask 的交叉熵，只在 assistant token 上算 loss。

题目：logits (B, T, V)、input_ids (B, T)、每个 token 的角色；logits 去尾、labels 去头做错位；
非 assistant 位置的 label 置 −100；交叉熵只在有效 token 上求和，再除以有效 token 数。
复杂度：时间 O(B·T·V)，空间 O(B·T·V)（log_softmax 的中间结果可只算有效位置，降到 O(B·T)）。
'''
import numpy as np


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return x - m - np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


def masked_cross_entropy(logits, input_ids, token_role, ignore_index=-100):
    '''token_role 与 input_ids 同形，assistant 为 1，system / user / 工具结果为 0。
    logits 位置 t 预测 t+1 的 token：logits[:, :-1] 对 labels[:, 1:]，mask 也取 [:, 1:]。
    返回 (loss, num_valid_tokens)。'''
    logits = np.asarray(logits, dtype=np.float64)
    B, T, V = logits.shape
    labels = np.asarray(input_ids).copy()
    mask = np.asarray(token_role).astype(bool)
    labels[mask == 0] = ignore_index
    sl = logits[:, :-1].reshape(-1, V)
    sl = sl - sl.max(axis=-1, keepdims=True)
    lp = sl - np.log(np.exp(sl).sum(axis=-1, keepdims=True))     # 稳定 log_softmax
    tgt = labels[:, 1:].reshape(-1)
    valid = tgt != ignore_index
    n = int(valid.sum())
    if n == 0:
        return 0.0, 0
    return float(-lp[np.arange(tgt.size), np.where(valid, tgt, 0)][valid].sum() / n), n


def masked_ce_naive(logits, input_ids, token_role):
    '''逐 token 循环的参考实现，用于对照。'''
    lg = np.asarray(logits, dtype=np.float64)
    ids = np.asarray(input_ids)
    role = np.asarray(token_role)
    B, T, V = lg.shape
    total, n = 0.0, 0
    for b in range(B):
        for t in range(T - 1):
            if role[b, t + 1] == 0:
                continue
            x = lg[b, t] - lg[b, t].max()
            lse = np.log(np.exp(x).sum())
            total += -(x[ids[b, t + 1]] - lse)
            n += 1
    return (total / n if n else 0.0), n


np.random.seed(0)
B, T, V = 3, 9, 17
logits = np.random.randn(B, T, V) * 2
ids = np.random.randint(0, V, size=(B, T))
role = np.zeros((B, T), dtype=int)
role[0, 2:6] = 1        # assistant 第一轮
role[0, 6] = 1          # 结束符也算 1
role[1, 4:8] = 1
role[2, 5:7] = 1
role[2, 8] = 1

loss, n = masked_cross_entropy(logits, ids, role)
naive, n2 = masked_ce_naive(logits, ids, role)
assert abs(loss - naive) < 1e-10 and n == n2 and n == role[:, 1:].sum()
assert np.isfinite(loss)

# 错位：位置 t 的 logits 预测 t+1 的 token，不是当前 token
off_by_one, _ = masked_cross_entropy(logits, ids, role)
same_pos = 0.0
cnt = 0
for b in range(B):
    for t in range(T - 1):
        if role[b, t + 1]:
            x = logits[b, t]
            x = x - x.max()
            same_pos += -(x[ids[b, t]] - np.log(np.exp(x).sum()))     # 故意不对齐
            cnt += 1
same_pos /= cnt
assert abs(off_by_one - same_pos) > 1e-3
# 错位方向：labels 取 [:, 1:]，mask 也按目标 token 对齐；最后一个位置的 logits 不参与
assert masked_cross_entropy(logits, ids, np.zeros((B, T), int))[0] == 0.0
role_only_first = np.zeros((B, T), int)
role_only_first[:, :1] = 1                       # 只有位置 0 标成 assistant，但位置 0 的 label 不参与
assert masked_cross_entropy(logits, ids, role_only_first)[1] == 0

# mask 规则：system / user / 工具结果为 0，不影响 loss；assistant 的回复、思考、工具调用、结束符为 1
# logits 的第 t 行对应目标 token t+1，所以污染范围是 logits[:, :-1]，index 是 role[:, 1:]
dirty = logits.copy()
dirty[:, :-1][role[:, 1:] == 0] = 1e6           # 污染所有非 assistant 目标位置
dirty[:, -1] = 1e6                              # 最后一个位置的 logits 本就不参与
assert abs(masked_cross_entropy(dirty, ids, role)[0] - loss) < 1e-12
# 多轮：每一轮 assistant 都算（不是只算最后一轮）。
# 去掉第一轮的 mask 后有效 token 数减少，loss 也随之变化
multi = role.copy()
n_before = masked_cross_entropy(logits, ids, multi)[1]
multi[0, 2:6] = 0
l_multi, n_multi = masked_cross_entropy(logits, ids, multi)
assert n_multi == n_before - 4
assert abs(l_multi - loss) > 1e-6

# 结束符必须算 loss：把最后那个 assistant 位置（结束符）mask 掉，有效 token 数减 1
no_eos = role.copy()
no_eos[0, 6] = 0
assert masked_cross_entropy(logits, ids, no_eos)[1] == n - 1

# 单元测试：单 token、logits 全是 0 → loss = log V
flat = np.zeros((1, 2, 5))
l_single, n_single = masked_cross_entropy(flat, np.array([[3, 1]]), np.array([[0, 1]]))
assert n_single == 1 and abs(l_single - np.log(5)) < 1e-12
# 目标 token 的 logit 越大，loss 越小
lg = np.zeros((1, 2, 5))
lg[0, 0, 2] = 5.0
l_hit, _ = masked_cross_entropy(lg, np.array([[3, 2]]), np.array([[0, 1]]))
l_miss, _ = masked_cross_entropy(lg, np.array([[3, 4]]), np.array([[0, 1]]))
assert l_hit < l_miss
# 返回值是“求和 / 有效 token 数”，不是先按样本平均：把一条样本复制两份，loss 不变
dup = np.concatenate([logits, logits], 0)
dup_ids = np.concatenate([ids, ids], 0)
dup_role = np.concatenate([role, role], 0)
assert abs(masked_cross_entropy(dup, dup_ids, dup_role)[0] - loss) < 1e-12

# 梯度累积：每个 micro-batch 先各自平均再平均 ≠ 按全局 token 数归一化（追问 1）
m1 = np.zeros((B, T), int)
m1[0] = role[0]
m2 = np.zeros((B, T), int)
m2[1:] = role[1:]
l1, n1 = masked_cross_entropy(logits, ids, m1)
l2, n2_ = masked_cross_entropy(logits, ids, m2)
assert n1 != n2_
assert abs((l1 + l2) / 2 - loss) > 1e-6                          # 等权平均口径不同
assert abs((l1 * n1 + l2 * n2_) / (n1 + n2_) - loss) < 1e-10     # 按全局 token 数正确

# 数值稳定：极端 logits 不产生 nan/inf，且与 float64 参考一致
ext = np.array([[[1e4, -1e4, 0.0], [0.0, 0.0, 0.0]]])
# target 位置取 ids[0, 1]：置 0 时目标是 logit 1e4 的 token，loss ≈ 0（关键是不溢出）
l_ext, _ = masked_cross_entropy(ext, np.array([[9, 0]]), np.array([[0, 1]]))
assert np.isfinite(l_ext) and abs(l_ext) <= 1e-9
l_ext2, _ = masked_cross_entropy(ext, np.array([[9, 1]]), np.array([[0, 1]]))
assert np.isfinite(l_ext2) and l_ext2 == 2e4                    # 目标 logit −1e4，loss 精确为 2e4
# 反例：先 softmax 再取 log。一行 [1e4, −1e4, 0] 里 exp 溢出、小概率被冲成 0，
# log 得到 −inf；log_softmax 不会
row = ext[0, 0]
with np.errstate(over='ignore', divide='ignore', invalid='ignore', under='ignore'):
    e = np.exp(row)
    bad = np.log(e / e.sum())
assert np.any(np.isinf(e)) and np.any(np.isinf(bad))
assert np.all(np.isfinite(log_softmax(ext[0, 1])))
