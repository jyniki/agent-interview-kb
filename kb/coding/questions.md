# 手撕题

> 面试里常见的现场编码题。每题给出要求、要点和追问，不给完整代码：先自己写，再对照要点自查。
> 题号前缀 `code-model-` 为模型侧，`code-eng-` 为工程侧。
> 写完再看参考实现：`solutions/<题号>.py`，每份都能单独运行（`python3 solutions/code-model-001.py`），结尾的 assert 覆盖了这道题的边界。先自己写，再看。

## code-model-001 手写 Multi-Head Attention（含因果 mask）
- level: 1
- tags: [core, training, inference]

**一句话答案**：Q、K、V 线性投影后拆成 h 个头，每头算 softmax(QKᵀ/√d_k + mask)V，拼接后再做一次输出投影。

**要点**：
- 形状：输入 (B, T, D) → 投影 → reshape 成 (B, h, T, d_k)，d_k = D/h。
- 缩放因子 √d_k，防止点积过大导致 softmax 饱和。
- 因果 mask：上三角位置填 −inf（或很大的负数），再 softmax。
- 最后 transpose 回 (B, T, h, d_k) 并 reshape 成 (B, T, D)。

**追问链**：
1. 改成 GQA 要动哪里？→ K、V 的头数变少，计算前把每个 KV 头重复给对应的一组 Q 头。
2. 加 KV Cache 怎么改？→ 解码时只算新 token 的 Q、K、V，把 K、V 拼到缓存后面，Q 只有一个位置，不再需要因果 mask。

**易错点**：
- reshape 前忘了 transpose，导致头和序列维度错位。

## code-model-002 手写 GRPO 的组内优势和 token 级 loss
- level: 2
- tags: [training]

**一句话答案**：按问题分组算 `(r − mean) / (std + eps)`，广播到每个 token；loss 用 clip 目标，乘 response mask，按 token 求和后除以总 token 数。

**要点**：
- 输入：rewards (B·G,)、新旧策略的 logprob (B·G, T)、response_mask (B·G, T)。
- 优势：reshape 成 (B, G) 求组内均值和标准差，再 reshape 回来并扩展到 T 维。
- ratio = exp(logp_new − logp_old)；loss = −min(ratio·A, clip(ratio)·A)。
- 聚合：`(loss * mask).sum() / mask.sum()` 是 token 级（DAPO 做法）；GRPO 原文是先按每条回答平均再按组平均。
- KL 项：`exp(ref − new) − (ref − new) − 1`，同样乘 mask。

**追问链**：
1. 怎么改成 Dr. GRPO？→ 优势不除 std；聚合分母换成固定常数。
2. 多轮工具调用时 mask 怎么构造？→ 只有模型生成的 token 为 1，工具返回、prompt 都为 0。

**易错点**：
- std 为 0 时没加 eps，产生 NaN。

## code-model-003 手写 LoRA 线性层
- level: 1
- tags: [core, training, inference]

**一句话答案**：冻结原权重 W，旁路加两个低秩矩阵 A (r×in)、B (out×r)，输出为 Wx + (α/r)·BAx；B 初始化为 0。

**要点**：
- B 初始化为 0，保证训练开始时输出与原模型一致。
- 缩放 α/r，让不同 r 下的学习率尺度相近。
- 合并：推理时可把 (α/r)·BA 加回 W，不增加延迟。

**追问链**：
1. 为什么 A 不能也初始化为 0？→ A、B 都为 0 时两者梯度都为 0，训练不起来。
2. 多 LoRA 服务怎么做？→ 底座共享，每个请求按 adapter ID 选 A、B，批内不同 adapter 分组计算。

**易错点**：
- 忘了冻结原权重。

## code-model-004 手写 DPO loss
- level: 1
- tags: [core, training]

**一句话答案**：loss = −log σ(β·[(logπ(y_w) − logπ_ref(y_w)) − (logπ(y_l) − logπ_ref(y_l))])。

**要点**：
- logπ(y) 是整条回答各 token logprob 在 response mask 上的求和。
- 参考模型冻结，可以离线预先算好 logprob。
- 用 `F.logsigmoid` 保证数值稳定。

**追问链**：
1. β 变大变小有什么影响？→ β 来自 RLHF 目标里的 KL 系数。β 越大，KL 约束越强，策略越贴近参考模型；β 越小，策略可以偏离参考模型更远去拟合偏好数据，也更容易过拟合。
2. 长度偏差从哪来？→ logπ 是求和，长回答的绝对值更大，偏好对里 chosen 往往更长，模型会学到“变长”。

**易错点**：
- 用 mean 代替 sum 却不说明，改变了目标的含义。

## code-eng-001 实现一个最小 Agent Loop
- level: 1
- tags: [core, harness]

**一句话答案**：循环调用模型，若输出工具调用就执行并把结果追加到消息，若输出最终答案就返回；设最大步数和异常处理。

**要点**：
- 消息列表是状态；每步把 assistant 消息和 tool 结果都追加进去。
- 工具注册表：名字 → 函数和参数 schema；参数先校验再执行。
- 终止：模型给出最终答案、达到最大步数、预算耗尽。
- 工具异常不要让循环崩溃，把错误信息作为工具结果返回给模型。

**追问链**：
1. 支持并行工具调用要改哪里？→ 模型一次返回多个调用时并发执行，按调用 ID 把结果对应回去。
2. 怎么检测死循环？→ 记录最近若干次（工具名, 参数）的哈希，重复出现时提示模型或终止。

**易错点**：
- 工具结果没带上调用 ID，多个调用时模型无法对应。

## code-eng-002 实现令牌桶限流器
- level: 1
- tags: [harness, inference]

**一句话答案**：桶容量 C、速率 r；每次请求先按距上次的时间补充 token（不超过 C），够就扣减放行，不够就拒绝或等待。

**要点**：
- 惰性补充：不用定时器，请求到来时按时间差计算。
- 线程安全：补充和扣减要在同一把锁内。
- 按 token 数限流（LLM 场景）时，扣减量是请求的预估 token 数。

**追问链**：
1. 分布式下怎么做？→ 用 Redis 加 Lua 脚本，保证补充和扣减原子执行。
2. 和漏桶的区别？→ 令牌桶允许突发（最多 C），漏桶以恒定速率输出。

**易错点**：
- 用整数秒计算时间差，精度不够。

## code-model-005 手写 RoPE：给 q、k 加旋转位置编码，输入 (B, h, T, d)、位置 (T,)
- level: 2
- tags: [core, training, inference]

**一句话答案**：按 θ_i = base^(−2i/d) 算每对维度的频率，角度为位置×θ_i；把 q、k 的每两维看成复数乘 e^(i·mθ)，即 x1·cos − x2·sin、x1·sin + x2·cos。

**要点**：
- 频率：i = 0…d/2−1，`inv_freq = base ** (-arange(0, d, 2) / d)`，base 原文为 10000。论文写作 θ_i = 10000^(−2(i−1)/d)，i 从 1 开始，两者一致。
- 角度表 `ang = pos[:, None] * inv_freq[None]`，形状 (T, d/2)；cos、sin 可以预计算并缓存。
- 论文式 34 的高效写法：`R·x = x ⊙ cos + rotate(x) ⊙ sin`，原文配对是相邻两维 (x1, x2)、(x3, x4)…，rotate 把每对变成 (−x2, x1)。
- 很多开源实现把前一半和后一半配对（rotate_half：`cat(-x[d/2:], x[:d/2])`），cos、sin 也要按同样方式拼成 (T, d)。两种配对数学等价，但权重必须和训练时一致。
- 只作用在 q、k 上，不作用在 v 上；旋转是正交变换，向量长度不变。
- 验证方法：位置 (m, n) 和 (m+c, n+c) 的 q·k 相等，说明内积只依赖相对距离。

**追问链**：
1. 解码时用了 KV Cache，新 token 的 RoPE 怎么算？→ 位置用当前绝对位置（缓存长度），k 在写入缓存前就旋转好，缓存里存的是旋转后的 k，之后不再重算。
2. Position Interpolation 在代码里改哪一行？→ 把位置下标除以扩展倍数 s 再乘频率，等价于角度整体缩小 s 倍；NTK 类做法改的是 base。
3. 为什么说低维转得快、高维转得慢？→ θ_i 随 i 指数变小，i 小的维度对每个位置转过的角度大，编码局部位置；i 大的维度周期很长，编码远距离。

**易错点**：
- 相邻配对的 cos、sin 和 rotate_half 的配对混用。实测混用后内积不再只依赖相对距离，模型输出会悄悄变差而不报错。
- 位置从 1 开始或 prefill 和 decode 的位置不连续。

## code-model-006 手写 RMSNorm，输入 (B, T, D)
- level: 1
- tags: [core, training, inference]

**一句话答案**：在最后一维算均方根 sqrt(mean(x²) + eps)，用它除 x，再乘可学习的缩放 g (D,)；不减均值，也没有偏置。

**要点**：
- `y = x / sqrt(mean(x**2, dim=-1, keepdim=True) + eps) * g`，g 初始化为 1。
- 和 LayerNorm 比：LayerNorm 先减均值再除标准差，并有偏置 β；RMSNorm 去掉了中心化和偏置，少一次规约，计算更省。
- eps 放在开方里面，常见取值 1e-6 或 1e-5。
- 混合精度下常把 x 转成 FP32 算方差，归一化后再转回原精度，最后乘 g。
- 验证：g 全为 1 时，输出每个位置的 mean(y²) 约等于 1。

**追问链**：
1. Pre-Norm 和 Post-Norm 的区别？→ Pre-Norm 是 `x + f(norm(x))`，残差主干不经过归一化，深层训练更稳；Post-Norm 是 `norm(x + f(x))`，深了需要 warmup 等技巧。
2. 为什么在 FP16 下直接算会出问题？→ x² 求和可能溢出或丢精度，所以先升到 FP32 算统计量。

**易错点**：
- 在 batch 或序列维度上求均值，变成了 BatchNorm 的口径。
- keepdim 没留，广播时形状对不上。

## code-model-007 手写带 KV Cache 的自回归解码循环：prefill 一次，再逐 token decode 到 EOS 或最大长度
- level: 2
- tags: [core, inference]

**一句话答案**：prefill 把整段 prompt 过一遍，每层存下 K、V；之后每步只喂上一步的新 token，算它的 Q、K、V，把 K、V 追加到缓存，用新 Q 和全部缓存做注意力，采样下一个 token。

**要点**：
- 缓存形状：每层一份 K、V，(B, n_kv, T_cache, d_h)，decode 每步在 T 维追加 1。
- prefill 需要因果 mask；decode 时 Q 只有一个位置，它本来就该看见全部历史，不再需要因果 mask。
- 位置编码用绝对位置 `pos = T_cache`，不是 0。
- 只缓存 K、V，不缓存 Q：之后只会用“新 Q 对历史 K、V”。
- 循环骨架：

```
logits, cache = model(prompt_ids, cache=None)        # prefill
for step in range(max_new):
    nxt = sample(logits[:, -1])
    if all finished: break
    logits, cache = model(nxt[:, None], cache=cache)  # decode，输入 (B, 1)
```

- 批内各条长度不同：prompt 左填充并配 attention mask，或者每条单独记录长度；已经输出 EOS 的序列之后只填 pad，不再更新结果。
- 验证：逐步 decode 的每一步输出，应和把完整序列一次性带因果 mask 前向的对应位置相同。

**追问链**：
1. 为什么 decode 阶段受显存带宽限制？→ 每步只算一个 token，算量小，却要读全部权重和整份 KV Cache。
2. 缓存预先分配和每步 concat 有什么差别？→ 每步 concat 会反复分配和拷贝，开销随长度增长；预分配最大长度后按下标写入更省，但浪费显存，这是 PagedAttention 要解决的问题。
3. 批里一条先结束了怎么办？→ 用 finished 标记，后续步骤给它填 pad 并忽略输出；服务端用连续批处理把它的位置让给新请求。

**易错点**：
- decode 时位置编码从 0 开始。
- 输出 EOS 后没停，或者把 EOS 之后的 pad 也算进结果。

## code-model-008 手写 top-k / top-p 采样：输入 logits (B, V)、temperature、k、p，输出下一个 token
- level: 1
- tags: [core, inference]

**一句话答案**：logits 先除温度；top-k 把第 k 大以下的置 −inf；top-p 按概率降序算累计和，只保留累计到 p 的最小集合，其余置 −inf；重新 softmax 后采样。

**要点**：
- 顺序：温度 → top-k → top-p → softmax → `multinomial`。
- top-k：`kth = topk(logits, k).values[..., -1:]`，`logits[logits < kth] = -inf`。
- top-p 的关键几行：

```
probs, idx = sort(softmax(logits), descending=True)
remove = cumsum(probs) - probs >= p      # 当前 token 之前已经够 p 了才删
logits_sorted[remove] = -inf，再按 idx 还原顺序
```

- 用“累计和减去自己”判断，保证跨过 p 的那个 token 被保留，概率最大的 token 永远保留，p 很小时退化成贪心。
- 例：概率 [0.5, 0.3, 0.15, 0.05]、p = 0.7，保留前两个，重新归一化为 [0.625, 0.375]。
- temperature = 0 单独走 argmax，不要除以 0。

**追问链**：
1. top-p 相比 top-k 好在哪？→ 保留集合大小随分布形状变化：分布尖时只留几个，平时留得多；top-k 的 k 固定。
2. 温度放在截断前后有区别吗？→ 有。温度改变分布形状，top-p 的保留集合会跟着变；常见实现是先温度后截断。
3. 批量实现时排序太贵怎么办？→ 先做 top-k 缩小候选，再在候选上做 top-p；或用不排序的阈值搜索。

**易错点**：
- 判断条件写成 `cumsum > p` 却在排序前比较，或把跨过 p 的那个 token 也删掉，导致集合为空、softmax 出 NaN。
- 截断后忘了按原下标还原，采到的 token id 是排序后的位置。

## code-model-009 手写 PPO 的 clip 策略 loss 和价值 loss：输入新旧 logprob、优势、旧价值、回报目标、mask，形状都是 (B, T)
- level: 2
- tags: [core, training]

**一句话答案**：ratio 为新旧概率比，策略 loss 取 −min(ratio·A, clip(ratio)·A)；价值 loss 为 (V−R)²，可对 V 做 clip；都按 mask 平均。

**要点**：
- 论文式 7：L^CLIP = E[min(r·Â, clip(r, 1−ε, 1+ε)·Â)]，要最大化，所以代码里取负号当 loss；论文建议 ε = 0.2。
- 论文式 9：总目标 L^CLIP − c1·L^VF + c2·S，L^VF = (V_θ(s_t) − V_t^targ)²，S 是熵。写成 loss 就是 `pg_loss + c1 * v_loss - c2 * entropy`。
- 价值 clip 是常见开源实现的做法，不是论文式 9 的内容：`v_clip = v_old + clip(v - v_old, -ε_v, ε_v)`，`v_loss = 0.5 * max((v - R)², (v_clip - R)²)`。
- 回报目标 R = A + V_old（GAE 算出来的），在 rollout 时算好，更新时是常数。
- 优势常在整个 batch 的有效 token 上做均值方差归一化，再进 loss。
- 验证：A > 0 且 ratio > 1+ε 时策略 loss 对 logp_new 的梯度为 0；A > 0 且 ratio < 1−ε 时梯度不为 0。
- 常加监控：`clipfrac = mean(|ratio − 1| > ε)` 和近似 KL。

**追问链**：
1. 为什么要取 min，而不是只 clip？→ min 让目标成为未 clip 目标的下界：朝有利方向走太远时被截断，变坏的方向保留梯度，可以纠正回来。
2. 为什么 ratio 用 logprob 相减再 exp？→ 直接用概率相除数值不稳，概率很小时会下溢。
3. LLM 的 PPO 里 KL 放在哪？→ 常见做法是把每个 token 的 KL 惩罚加进奖励，再算 GAE；GRPO 则把 KL 直接放进 loss。

**易错点**：
- logp_old 没有 detach，或者每个 epoch 都重新计算，ratio 恒为 1。
- 回报目标在更新时用当前价值重新算，目标随参数移动。

## code-model-010 手写 GAE：输入 rewards、values、next_values、dones、truncated，形状 (T,)，输出优势和回报目标
- level: 2
- tags: [core, training]

**一句话答案**：从后往前递推 δ_t = r_t + γV(s_{t+1}) − V(s_t)，A_t = δ_t + γλA_{t+1}；终止时不自举且切断递推，截断时仍自举但切断递推；R = A + V。

**要点**：
- 公式来自 GAE 论文：δ_t = r_t + γV(s_{t+1}) − V(s_t)，Â_t = Σ_l (γλ)^l δ_{t+l}。PPO 论文式 11、12 是长度为 T 的截断版本，最后一步用 V(s_T) 自举。
- 递推骨架：

```
last = 0
for t in reversed(range(T)):
    delta = r[t] + gamma * next_v[t] * (1 - done[t]) - v[t]
    last  = delta + gamma * lam * (1 - done[t]) * (1 - trunc[t]) * last
    A[t]  = last
R = A + v
```

- 终止（done）：环境真的结束了，没有未来回报，δ 里不自举，递推也在这里切断。
- 截断（时间上限、rollout 片段末尾）：环境没结束，δ 里仍要用最后观测的 V(s_{t+1}) 自举；递推在 episode 边界处切断，不把下一个 episode 的优势传回来。
- 片段末尾的 next_v 用最后一个观测的价值估计，不填 0。
- 验证：λ = 0 时 A_t = δ_t；λ = 1 且无边界时 A_t 等于折扣回报（末尾自举）减 V(s_t)。

**追问链**：
1. λ 怎么影响偏差和方差？→ λ 接近 0 只看一步，方差低，V 不准时有偏；λ 接近 1 接近蒙特卡洛回报，无偏但方差高。PPO 论文 MuJoCo 实验用 γ = 0.99、λ = 0.95。
2. LLM 只在最后一个 token 有奖励时怎么用？→ 每个 token 是一步，奖励放在最后一个有效 token 上，其他步奖励为 0（加了 KL 惩罚时每步还有 −β·KL）；pad 位置要 mask 掉。
3. 截断也当 done 处理有什么后果？→ 把“还没结束”当成“没有未来回报”，价值被系统性低估，时间上限附近的状态学到错误的估值。

**易错点**：
- 用 values[t+1] 当 next_value，episode 边界处拿到了下一个 episode 的第一个状态。
- 优势归一化之后再算 R = A + V，回报目标被改变了。

## code-model-011 手写 BPE 训练：输入词频表 {词: 次数} 和合并次数 n，输出合并规则列表
- level: 2
- tags: [core]

**一句话答案**：把每个词拆成字符序列加词尾标记；每轮按词频统计所有相邻对的出现次数，取最高频的一对合并成新符号，在所有词里替换，记下这条规则，重复 n 轮。

**要点**：
- 初始化：`vocab = {tuple(w) + ('</w>',): c}`，词尾标记区分词中和词尾的同一片段。
- 每轮统计：遍历每个词的相邻对，计数加上该词的词频，不是加 1。
- 替换：从左到右扫描，遇到最佳对就合并并跳过两个位置，否则保留一个。
- 编码新文本时按学到的合并顺序依次应用，顺序就是优先级。
- 例：{low: 5, lower: 2, newest: 6, widest: 3}，平局规则取“计数最高 → 词尾优先 → 字典序最小”，则首轮合并是 (w, e)——(t, </w>) 与 (s, t) 各 11 次，字典序在 (w, e) 之后。要复现经典的前三次 (t, </w>)、(s, t</w>)、(e, st</w>)，换成语料 {lowest: 5, newer: 6, widest: 3} 更直接。平局规则必须写死，否则同一份词频会得到不同的合并序列。
- 没有相邻对可合并时提前结束。

**追问链**：
1. 每轮全量重新统计太慢怎么办？→ 维护对计数和“对 → 出现在哪些词”的倒排索引，合并后只更新受影响的词和相邻对。
2. 字节级 BPE 改了什么？→ 初始符号从字符换成 256 个字节，任何字符串都能编码；通常先用正则预分词，防止跨空格、跨标点合并。
3. 词表大小怎么取舍？→ 越大序列越短、推理越快，但嵌入层和输出层变大，低频 token 训练不充分。

**易错点**：
- 重叠的对计数和替换口径不一致：“aaaa” 里 (a, a) 统计出 3 次，实际只能合并 2 次。
- 替换时用字符串 replace，把 “a b” 误合并进别的符号里。

## code-model-012 手写 GQA 的 KV 头复制：K、V 形状 (B, n_kv, T, d)，复制成 (B, n_q, T, d)
- level: 1
- tags: [core, inference]

**一句话答案**：每个 KV 头服务连续的 rep = n_q/n_kv 个 Q 头：在头维度用 repeat_interleave 把每个 KV 头连续复制 rep 次，Q 头 j 对应 KV 头 j // rep。

**要点**：
- 前提 n_q 能被 n_kv 整除，组大小 `rep = n_q // n_kv`。
- 写法：`k[:, :, None].expand(B, n_kv, rep, T, d).reshape(B, n_q, T, d)`，等价于 `repeat_interleave(k, rep, dim=1)`；expand 不拷贝内存，reshape 时才物化。
- 缓存里只存 n_kv 个头，复制只在算注意力时做，省下的就是 KV Cache 的 n_q/n_kv 倍。
- rep = 1 是 MHA，n_kv = 1 是 MQA。
- 更省的做法不复制：把 Q reshape 成 (B, n_kv, rep, T, d)，和 K 直接广播相乘。

**追问链**：
1. 为什么不用 `repeat`（tile）？→ tile 得到的顺序是 [0, 1, 0, 1]，Q 头和 KV 头的对应关系错了，和预训练权重不匹配。
2. 已有 MHA 权重怎么转成 GQA？→ GQA 论文的做法是把同组的 K、V 头做均值池化，再用约 5% 的预训练计算量继续训练。

**易错点**：
- 在 T 维度上复制。
- 在写入缓存之前复制，KV Cache 没有省下来。

## code-model-013 手写带 loss mask 的交叉熵：输入 logits (B, T, V)、input_ids (B, T)、每个 token 的角色，只在 assistant token 上算 loss
- level: 1
- tags: [core, training]

**一句话答案**：logits 去掉最后一位、labels 去掉第一位做错位；非 assistant 位置的 label 置为 −100；交叉熵只在有效 token 上求和，再除以有效 token 数。

**要点**：
- 错位：位置 t 的 logits 预测 t+1 的 token，`logits[:, :-1]` 对 `labels[:, 1:]`，mask 也按目标 token 对齐取 `[:, 1:]`。
- mask 规则：system、user、工具结果为 0；assistant 的回复、思考和工具调用为 1，结束符也为 1，否则模型学不会停。
- 实现：`labels[mask == 0] = -100`，`cross_entropy(logits.reshape(-1, V), labels.reshape(-1), ignore_index=-100)`，默认就是在有效 token 上平均。
- 公式：loss = −Σ m_t·log p(x_t | x_<t) / Σ m_t。
- mask 从 chat template 渲染时的角色区间得到，不要靠在 token 序列里搜角色字符串。
- 数值稳定：用 log_softmax，不要先 softmax 再取 log。

**追问链**：
1. 梯度累积时按 micro-batch 平均有什么问题？→ 每个 micro-batch 的有效 token 数不同，先各自平均再平均，会改变 token 的权重；要按全局有效 token 数归一化。
2. packing 之后要注意什么？→ 用 attention mask 或 position id 隔开样本，并且按 token 平均时长样本权重更大。
3. 多轮样本只算最后一轮会怎样？→ 前面几轮 assistant 回复浪费了；主流做法是所有 assistant 轮都算。

**易错点**：
- 没做错位，模型在学“复制当前 token”。
- 结束符被 mask 掉，推理时一直往下写。

## code-eng-003 实现带抖动的指数退避重试：retry(fn, max_attempts, base, cap)
- level: 1
- tags: [harness]

**一句话答案**：失败后等 sleep = random(0, min(cap, base·2^attempt)) 再重试（Full Jitter），只重试临时错误，到最大次数或总时限就抛出；有副作用的调用要带幂等键。

**要点**：
- 退避上限 cap 防止等待无限增长；Full Jitter 让并发客户端的重试时间错开，避免同时重试打垮下游。
- 只重试可重试错误：超时、429、5xx、连接错误；参数错误、401、403 直接抛出。
- 服务端给了 Retry-After 时优先用它。
- 同时受总时限（deadline）约束：剩余时间不够下一次等待时直接失败。
- 最后一次失败要抛出原始异常，并带上尝试次数，方便排查。

**追问链**：
1. 为什么加抖动？→ 不加抖动时，同一时刻失败的客户端会在同一时刻重试，形成一波波同步的流量尖峰；加抖动把重试摊开。
2. 多层都重试会怎样？→ 次数相乘放大负载，五层各试三次，底层负载是 243 倍；只在一层重试，并用重试预算限制重试占比。
3. 超时了能直接重试吗？→ 超时不等于没执行。有副作用的调用要带幂等键，下游不支持幂等就先查询是否已执行。

**易错点**：
- 对所有异常都重试，包括参数错误。
- 用 `time.sleep` 阻塞异步事件循环，应该用 `await asyncio.sleep`。

## code-eng-004 实现 LRU 缓存：get(key) 和 put(key, value) 都是 O(1)，容量满时淘汰最久未使用的项
- level: 1
- tags: [harness, inference]

**一句话答案**：哈希表存 key → 双向链表节点，链表按访问时间排序；get 命中就把节点移到表头，put 时更新或插入表头，超出容量就删表尾节点并从哈希表移除。

**要点**：
- 用头尾哨兵节点，插入和删除不用判断空链表。
- get 也算“使用”，要移动节点；put 已存在的 key 时更新值并移动，不增加计数。
- 淘汰时要同时删链表节点和哈希表里的 key，所以节点里要存 key。
- Python 里可以用 `OrderedDict` 的 `move_to_end` 和 `popitem(last=False)`，面试一般要求手写链表。
- 多线程使用时 get 也会改链表，读写都要加锁。

**追问链**：
1. 要支持 TTL 怎么改？→ 节点存过期时间，get 时发现过期就删除并当作未命中；再加定期清理，防止过期项占着容量。
2. 按字节而不是按条数限制容量？→ 记录每项大小和总大小，put 后循环淘汰表尾直到总大小不超过上限；单项超过上限时直接拒绝缓存。
3. LRU 有什么缺点？→ 一次大范围扫描会把热点数据全挤出去；LRU-K、2Q 等方案要求访问多次才进入热区。

**易错点**：
- 节点里没存 key，淘汰时无法从哈希表删除。
- capacity 为 0 时 put 后没有立即淘汰。

## code-eng-005 实现流式工具调用增量的拼接与解析：输入 OpenAI 兼容格式的 chunk 流，输出完整的工具调用列表
- level: 2
- tags: [harness, inference]

**一句话答案**：按 index 累积每个工具调用：id 出现时记下，name、arguments 按片段拼接；流结束（finish_reason 为 tool_calls）后再整体解析 JSON 并校验。

**要点**：
- 按 `index` 聚合，不按 id：后续分片通常只带 index 和 arguments 片段，不再带 id。
- 一次回复可能有多个并行调用，每个 index 一份缓冲；输出时按 index 排序。
- arguments 分片不是合法 JSON，不能逐片解析；需要提前展示时用容错的增量解析器，执行工具一定等完整参数。
- 同一条流里文本 content 和 tool_calls 可能都出现，要分开累积。
- 解析失败或不符合 schema 时，不要崩溃，把错误作为工具结果返回给模型，让它重新生成调用。
- 流在 finish_reason 之前中断，缓冲里的调用视为不完整，不能执行。

**追问链**：
1. 服务端怎么从模型原始文本解析出工具调用？→ 按模型的标记（如 `<tool_call>`）识别；流式时要扣住可能是半个起始标签的后缀，不能先当普通文本吐出去。
2. 参数为空字符串怎么办？→ 当作 `{}` 处理，再按 schema 检查必填字段。
3. 怎么测试这段代码？→ 把一个完整调用随机切成多段、多个调用交错下发，断言拼接结果和原始调用一致；再测截断和非法 JSON。

**易错点**：
- 用 `=` 覆盖 arguments 而不是 `+=` 拼接。
- 每个 chunk 都尝试 `json.loads` 并在失败时报错。

## code-eng-006 实现按工具描述做 top-k 检索选工具：select_tools(query, tools, k)，每个工具有名字、描述和参数 schema
- level: 2
- tags: [harness]

**一句话答案**：离线把每个工具的名字加描述做嵌入并归一化；请求来时嵌入 query，算余弦相似度取 top-k，先做权限过滤，再加上常驻工具和一个工具搜索工具作为漏召回的出口。

**要点**：
- 索引文本用“名字 + 描述 + 参数说明”，不只用名字；工具更新时重建对应向量。
- 相似度：向量归一化后点积即余弦；工具数很多时用向量库或 `argpartition` 取 top-k，不必全排序。
- 常驻工具（高频、通用）不参与检索，始终提供。
- 权限过滤在交给模型之前做，不让模型看到无权使用的工具。
- 可以和 BM25 混合：工具名、专有名词用关键词检索更准。
- 输出顺序尽量稳定，比如按工具名排序，减少前缀缓存失效。

**追问链**：
1. 检索漏召回怎么办？→ 保留一个“搜索工具”的工具，模型发现缺工具时可以自己搜；同时单独统计检索召回率。
2. 怎么评测这一层？→ 构造“请求 → 应该用的工具”标注集，看 recall@k；不要只看端到端成功率，否则分不清是没检索到还是检索到了没选对。
3. 多轮对话时 query 用什么？→ 只用最后一句容易丢上下文，常用最近几轮拼接或让模型改写出当前意图。

**易错点**：
- 先取 top-k 再做权限过滤，过滤后剩下的工具太少。
- 向量没归一化就用点积当余弦相似度。

## code-eng-007 实现简单的语义缓存：lookup(query) 和 store(query, answer)，嵌入相似度超过阈值就命中
- level: 2
- tags: [harness, inference]

**一句话答案**：存 (嵌入, 回答, 租户, 过期时间)；查询时嵌入 query，只在同租户、未过期的条目里找余弦相似度最高的一条，超过阈值才返回；含工具调用或有副作用的回答不写入。

**要点**：
- 缓存键要带租户和权限范围，防止一个用户的回答被另一个用户命中。
- 写入前过滤：不缓存带工具调用、有副作用、未通过校验的回答。
- 每条设 TTL；源数据变化时按标签失效；容量满时按 LRU 淘汰。
- 相似但不同的风险：“取消订单 A”和“取消订单 B”向量很近，可以把实体抽出来做精确匹配的一部分。
- 阈值按误报率定：在真实流量样本上标注“该命中 / 不该命中”，再选阈值，不能按命中率定。
- 条目多时用向量库做近似最近邻，不要线性扫描。

**追问链**：
1. 为什么 Agent 里语义缓存适用面小？→ 每轮上下文都不同，适合的只有无状态、事实稳定的单轮问答；多轮任务更该依赖前缀缓存。
2. 语义缓存和前缀缓存的区别？→ 前缀缓存在模型服务内部复用 prefill 计算，不会答错；语义缓存在网关跳过整次调用，相似不等于相同，会答错。
3. 缓存了一条错误回答怎么办？→ 支持按条删除，并且只缓存通过校验或被用户认可的回答。

**易错点**：
- 只看相似度，不看租户和过期时间。
- 拿最相似的一条就返回，忘了和阈值比较。
