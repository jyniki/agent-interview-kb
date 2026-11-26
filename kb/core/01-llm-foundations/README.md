# LLM 与 Transformer 基础

> 层级：通用层 · 预计 240 分钟 · 题目：[questions.md](questions.md)
> 前置：无。这是通用层的第一章。
> 后续：[11 后训练入门](../11-post-training-basics/README.md)、[inference/01 KV Cache 与显存估算](../../tracks/inference/01-kv-cache-memory/README.md)

## 1 这一章解决什么问题

Agent 岗的面试常从模型本身问起：一段文字怎么变成 token，注意力怎么算，位置信息从哪来，一个 7B 模型的参数和计算量从哪来，模型为什么一个字一个字地吐，温度设成 0 为什么结果还会变。这些问题决定了 Agent 系统的很多工程取舍：工具结果为什么要裁剪，提示词前缀为什么要稳定，长上下文为什么贵，复现线上问题为什么不能只靠固定参数。

面试官追问时，通常不满足于公式本身，会接着问“为什么”：为什么除以 √d_k 而不是 d_k，为什么 RoPE 的内积只依赖相对位置却还是不能外推，为什么每 token 的计算量是 2N，Chinchilla 的“20 token / 参数”从哪来。这一章对每个结论给出推导、出处或可运行的验证。

学完这一章，你应该能：

- 写出 BPE 的训练和编码算法，说清编码为什么要按合并顺序，以及词表大小对序列长度、嵌入参数、低频 token 的影响；
- 从点积的方差推出 √d_k 缩放，写出带因果 mask 的多头注意力，算出 MHA、MQA、GQA 的参数和 KV 大小；
- 推导 RoPE 的内积只依赖相对位置，用“波长”解释它为什么外推失败，以及 PI、NTK、YaRN 各改了哪些频率；
- 说清 Pre-LN 和 Post-LN 的区别、RMSNorm 和 SwiGLU 改了什么；
- 从模型 config 数出参数量并和官方数字对上，估算每 token 的 FLOPs 和训练算力 C ≈ 6ND；
- 说出 Kaplan 和 Chinchilla 结论的差别和原因，用 Chinchilla 的拟合式推出最优分配；
- 写出温度、top-k、top-p 的实现，解释温度 0 为什么也不确定。

KV Cache 的推导和显存估算在 [inference/01](../../tracks/inference/01-kv-cache-memory/README.md)，量化在 [inference/04](../../tracks/inference/04-quantization/README.md)，这里只给结论和链接。

**本章符号**（后文沿用）：

| 符号 | 含义 |
|---|---|
| V | 词表大小 |
| T | 序列长度（token 数） |
| d | 隐藏维度（config 里的 `hidden_size`），也写作 d_model |
| L | 层数（`num_hidden_layers`） |
| n_h, n_kv | 查询头数、KV 头数；MHA 时 n_kv = n_h |
| d_h | 每个头的维度，通常 d_h = d / n_h；注意力公式里也写作 d_k |
| d_ff | FFN 中间层宽度（`intermediate_size`） |
| q_i, k_j, v_j | 位置 i 的查询向量，位置 j 的键、值向量 |
| s_ij, a_ij | 位置 i 对位置 j 的打分和注意力权重 |
| θ_i, λ_i | RoPE 第 i 对维度的旋转频率（弧度 / 位置）和波长 2π/θ_i |
| N, D, C | 参数量、训练 token 数、训练算力（FLOPs） |
| τ | 采样温度（为避免和序列长度 T 混淆，本章用 τ） |

## 2 Tokenizer：BPE 的训练、编码与词表大小

### 2.1 问题：模型只能处理有限个离散符号

模型的输入是一个整数序列，每个整数查嵌入表得到一个 d 维向量。切分粒度有三种选择：

| 方案 | 做法 | 优点 | 代价 |
|---|---|---|---|
| 字符级 | 每个字符一个 token | 词表小，没有未登录词 | 序列很长，注意力开销按 T² 增长 |
| 词级 | 按空格或分词器切词 | 序列短 | 词表巨大，新词、拼写错误、代码标识符无法表示 |
| 子词（BPE 等） | 高频片段合成一个 token，低频词拆成几段 | 长度和覆盖率折中 | 切分结果依赖训练语料，跨语言不均衡 |

[Sennrich 等，2015-08](https://arxiv.org/abs/1508.07909) 把 BPE 压缩算法用到机器翻译。BPE 原本是把最高频的字节对替换成一个没用过的字节；论文改成合并字符或字符序列，生僻词被拆成几个常见片段，不再有“未登录词”。

### 2.2 训练：反复合并最高频的相邻对

论文 3.2 节的算法：

1. 初始符号表 = 语料里出现的所有字符，再加一个词尾符号（论文写作 `·`，常见实现写作 `</w>`），用来在解码时还原词的边界。
2. 把每个词拆成字符序列，按词频加权统计所有相邻符号对的出现次数。为了效率，**不统计跨词的对**。
3. 取出现次数最多的对 (A, B)，把所有出现处替换成新符号 AB，记下这条合并规则。
4. 重复 2、3，直到合并次数达到设定值。

最终词表大小 = 初始符号数 + 合并次数。合并次数是唯一的超参数。

论文的玩具语料是 `low`×5、`lower`×2、`newest`×6、`widest`×3。第一步统计：(e, s) 出现在 newest 和 widest 里，计数 6 + 3 = 9；(s, t) 也是 9；(l, o) 是 5 + 2 = 7。并列时取先出现的，于是第一条规则是 e + s → es。第二步 (es, t) 计数 9，合成 est。第三步 (est, </w>) 合成 `est</w>`。后面依次学到 lo、low、ne、new。下面代码会逐条复现。

### 2.3 编码：按规则学到的顺序合并

训练得到的是一张有序的合并规则表。编码一个新词时：

1. 拆成字符，加词尾符号；
2. 在当前序列的所有相邻对里，找到**规则表里排名最靠前（rank 最小）**的那一对，合并；
3. 重复，直到没有相邻对在规则表里。

GPT-2 的 `encoder.py` 就是这么写的：每轮取 `min(pairs, key=bpe_ranks.get)`。

为什么不能用“从左到右取最长的已知片段”？因为结果会和训练时的切分不一致。以 `nest` 为例：按 rank，(e, s) 排第 1，先合并，接着合成 `est</w>`，结果是 `n` + `est</w>`。最长匹配会先吃掉 `ne`（它也在词表里），剩下的 s、t 再也合不成训练时见过的片段，结果是 4 个 token。同一个词两种切法，模型在训练中见到的是前一种。

字节级 BPE（GPT-2 起常用）把初始符号表换成 256 个 UTF-8 字节，任何字符串都能编码，没有未登录字符。代价是：训练语料里少见的文字会被切到字节级。一个常用汉字是 3 个 UTF-8 字节，如果没有学到对应的合并，就要 3 个 token。

SentencePiece（[Kudo 和 Richardson，2018-08](https://arxiv.org/abs/1808.06226)）直接在原始句子上训练，空格也当普通字符，不依赖预分词，适合中文、日文这类没有空格分词的语言。它支持 BPE 和 Unigram 两种算法。

### 2.4 实现：最小 BPE

下面的代码实现训练和按 rank 编码，用论文的玩具语料核对前几条规则，演示 `nest` 在两种编码方式下的差别，再在一段小语料上看合并次数和总 token 数、最冷门 token 频次的关系。

```python
import collections

def train_bpe(word_freq, num_merges):
    """word_freq: {词: 频次}。返回按学到的先后排好的合并规则列表 [(a, b), ...]。"""
    # 每个词表示成符号元组，末尾加词尾符号 "</w>"（Sennrich 论文里写作 '·'）
    vocab = {tuple(w) + ("</w>",): f for w, f in word_freq.items()}
    merges = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for sym, f in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[a, b] += f                     # 相邻对按词频加权计数，不跨词
        if not pairs:
            break
        best = max(pairs, key=pairs.get)             # 频次并列时取先出现的，和论文 Algorithm 1 一致
        merges.append(best)
        new_vocab = {}
        for sym, f in vocab.items():                 # 把所有 (a, b) 替换成新符号 a+b
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new_vocab[tuple(out)] = f
        vocab = new_vocab
    return merges


def encode_word(word, merges):
    """按合并规则的“学到顺序”（rank）编码：每次合并当前序列里 rank 最小的相邻对。"""
    rank = {p: i for i, p in enumerate(merges)}
    sym = list(word) + ["</w>"]
    while True:
        cands = [(rank[p], i) for i, p in enumerate(zip(sym, sym[1:])) if p in rank]
        if not cands:
            return sym
        _, i = min(cands)
        sym[i:i + 2] = [sym[i] + sym[i + 1]]


# 1) 论文 Algorithm 1 的玩具语料：low×5, lower×2, newest×6, widest×3，合并 10 次
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
merges = train_bpe(corpus, 10)
print(merges)
# 第一步：(e, s) 出现在 newest、widest 里，6 + 3 = 9 次，是最高频相邻对
assert merges[:4] == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o")]
# 词表大小 = 初始字符表大小 + 合并次数（论文 3.2 节）
init = {c for w in corpus for c in w} | {"</w>"}
assert len(init) + len(merges) == len(init | {a + b for a, b in merges})

# 2) 编码训练时没见过的词：lowest 被切成两个学到的片段，没有未登录词
assert encode_word("lowest", merges) == ["low", "est</w>"]
assert encode_word("newer", merges) == ["new", "e", "r", "</w>"]
# 同一串字符在不同边界下切法不同："low" 作为整词是 low</w>，作为 lowest 的前缀是 low
assert encode_word("low", merges) == ["low</w>"]

# 3) 编码必须按 rank，不能“贪心找最长匹配”或随意顺序：两种顺序结果可以不同
def encode_left_to_right(word, merges):
    vocab = {a + b for a, b in merges} | set(word) | {"</w>"}
    s, out = word + "</w>", []
    while s:                                          # 每次取能匹配的最长前缀
        for j in range(len(s), 0, -1):
            if s[:j] in vocab:
                out.append(s[:j]); s = s[j:]; break
    return out
assert encode_word("widest", merges) == ["wi", "d", "est</w>"]
# "nest"：按 rank 先合并 (e, s)，得到 n + est</w>；最长匹配先吃掉 "ne"，剩下的 s、t 再也合不上
assert encode_word("nest", merges) == ["n", "est</w>"]
assert encode_left_to_right("nest", merges) == ["ne", "s", "t", "</w>"]

# 4) 词表大小的取舍：合并次数越多，语料总 token 数越少，但最冷门 token 的出现次数也越少
text = ("the quick brown fox jumps over the lazy dog " * 3 +
        "a quick test of the tokenizer on the lazy fox " * 2).split()
freq = collections.Counter(text)
lengths, min_counts = [], []
for k in [0, 10, 20, 40]:
    m = train_bpe(freq, k)
    toks = collections.Counter()
    for w, f in freq.items():
        for t in encode_word(w, m):
            toks[t] += f
    lengths.append(sum(toks.values()))
    min_counts.append(min(toks.values()))
print("总 token 数", lengths, "最冷门 token 出现次数", min_counts)
assert lengths == sorted(lengths, reverse=True) and lengths[0] > 2 * lengths[-1]
assert min_counts[-1] <= min_counts[0]

# 5) 字节级 BPE 的起点：UTF-8 字节。常用汉字每个 3 字节，没有合并时 1 个汉字 = 3 个 token
s = "注意力机制"
assert len(s.encode("utf-8")) == 3 * len(s)
assert len("attention".encode("utf-8")) == len("attention")
```

### 2.5 词表大小的取舍

词表 V 影响三件事：

**序列长度**。V 越大，高频片段越长，同样的文字切出的 token 越少。上面的小语料里，合并 0、10、20、40 次，总 token 数从 224 降到 61。token 少意味着推理更快、上下文能装更多内容、注意力的 T² 项更小。公开例子：Llama 3 用 128K 词表，由 tiktoken 的 100K token 加 28K 个补充非英语的 token 组成；在英语样本上，压缩率从 Llama 2 的每 token 3.17 个字符提高到 3.94 个（[Llama 3 技术报告](https://arxiv.org/abs/2407.21783)，3.2 节）。按这个比例，同样 8K token 的上下文能多装约 24% 的英文文本（第 9 节代码）。

**嵌入参数**。输入嵌入是 V × d 的表，输出头（把隐藏状态投成 V 个 logits）也是 d × V。不共享时合计 2·V·d。Llama 3 8B 的 V = 128,256、d = 4096，这两块共 1.05B 参数，占总参数 8.03B 的 13%（第 5.7 节代码从 config 数出）。小模型里这个比例更高。输出头每个 token 还要做一次 d × V 的矩阵乘，占前向计算量的一部分。

**低频 token**。V 越大，排在后面的 token 在语料里越少见，对应的嵌入行得到的梯度更新越少。上面的小语料里，最冷门 token 的出现次数从 3 降到 2；真实词表里，最后几千个 token 可能在训练数据里几乎没出现。[Land 和 Bartolo，2024-05](https://arxiv.org/abs/2405.05417) 把这类 tokenizer 里有、训练中几乎或完全没见过的 token 叫作“训练不足的 token”（under-trained tokens），也就是常说的 glitch token。输入里出现它们，模型可能产生幻觉或乱码输出。原因是 tokenizer 和模型分开训练：tokenizer 的语料里高频的片段，不一定在模型的训练数据里高频。

经验上的折中是：V 足够大，让目标语言和代码的压缩率合理；又不能大到嵌入表占掉大部分参数、尾部 token 训练不足。多语言模型通常需要更大的词表，因为每种语言都要分到足够的合并次数。

### 2.6 和 Agent 的关系

- 计费、上下文预算、截断都按 token 算，不按字数算。同一段中文在不同模型上的 token 数可以差很多，预算要用目标模型的 tokenizer 实测。
- 对话模板里的特殊 token（角色分隔、工具调用起止）是模型和推理框架之间的协议。模板拼错，模型会把工具结果当成用户输入，或者不停输出。SFT 的 chat template 和 loss mask 见 [11 后训练入门第 3 节](../11-post-training-basics/README.md#3-sftchat-templateloss-mask-与-packing)。
- 字符串拼接后再分词，和分别分词再拼接，结果可能不同（上面代码里 `low` 作为整词和作为 `lowest` 前缀切法不同）。按 token 截断工具结果再解码，可能在多字节字符中间截断，产生乱码。

## 3 Self-Attention 与 KV 头的演进

出处：[Vaswani 等，Attention Is All You Need，2017-06](https://arxiv.org/abs/1706.03762)。

### 3.1 从定义出发

一层注意力的输入是 T 个隐藏向量 x_1..x_T（每个 d 维）。每个位置用三个投影矩阵得到查询、键、值：

```text
q_i = W_Q x_i      k_j = W_K x_j      v_j = W_V x_j          （每个 d_k 维）
```

位置 i 的输出是所有值向量的加权平均，权重由 q_i 和每个 k_j 的相似度决定：

```text
s_ij = q_i · k_j / √d_k                     打分
a_ij = exp(s_ij) / Σ_j' exp(s_ij')           softmax，Σ_j a_ij = 1
o_i  = Σ_j a_ij · v_j                         加权求和
```

写成矩阵形式就是论文的公式：

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V        Q: (T, d_k)  K: (T, d_k)  V: (T, d_v)
```

`Q Kᵀ` 是 T × T 的打分矩阵，计算量和显存都随 T 平方增长。这是长上下文贵的根本原因。

### 3.2 为什么除以 √d_k：点积方差推导

论文脚注 4 给出的理由：假设 q 和 k 的各分量独立，均值 0，方差 1。则

```text
E[q·k]   = Σ_m E[q_m] E[k_m] = 0
Var[q·k] = Σ_m Var[q_m k_m]                    （各项独立，方差相加）
         = Σ_m E[q_m²] E[k_m²] − (E[q_m] E[k_m])²
         = Σ_m 1·1 − 0 = d_k
```

点积的标准差是 √d_k。d_k = 128 时约 11.3。打分的尺度随维度增长，softmax 的输入差距很大，输出接近 one-hot。

softmax 接近 one-hot 时梯度很小。softmax 的雅可比是 ∂a_i/∂s_j = a_i(δ_ij − a_j)。如果某个 a_i ≈ 1、其余 ≈ 0，每一项 a_i(δ_ij − a_j) 都接近 0：最大的那项是 1·(1 − 1)，其余是 0·(…)。论文的原话是，点积变大会把 softmax 推到“梯度极小”的区域。

除以 √d_k 后，打分的方差回到 1，和 d_k 无关。所以不管头维度多大，softmax 的输入尺度一致。除以 d_k 会把方差压到 1/d_k，打分几乎都接近 0，softmax 接近均匀，注意力失去选择性。

两个常见误解：缩放不是为了“归一化概率”，归一化是 softmax 做的；推导假设的“方差 1”是初始化附近的近似，训练后 q、k 的尺度会变，所以还有 QK-Norm 等做法直接归一化 q、k。

### 3.3 因果 mask

解码器做下一个 token 预测，位置 i 只能看到 j ≤ i。实现上在 softmax 前把 j > i 的打分设成 −∞：

```text
s_ij = q_i · k_j / √d_k    若 j ≤ i
s_ij = −∞                   若 j > i          → exp(−∞) = 0，a_ij = 0
```

这样训练时一次前向就能并行算出所有位置的预测，每个位置都不会偷看答案。推理时它带来一个重要性质：历史位置的 K、V 不会因为后面追加 token 而改变，所以能缓存。完整推导见 [inference/01 第 2.1 节](../../tracks/inference/01-kv-cache-memory/README.md#21-从注意力出发为什么缓存-kv不缓存-q)。

不带 mask、不带位置编码的注意力是置换等变的：打乱输入顺序，输出跟着同样打乱，每个位置的值不变。所以注意力本身不知道顺序，要另外注入位置信息（第 4 节）。

### 3.4 多头注意力

论文把 d 维向量投影成 h 组 d_k = d/h 维的 Q、K、V，每组独立算注意力，把 h 个输出拼接后乘 W_O：

```text
head_r = Attention(X W_Q^r, X W_K^r, X W_V^r)        r = 1..h，每个 (T, d/h)
MultiHead(X) = Concat(head_1, ..., head_h) · W_O      (T, d)
```

论文的解释：多头让模型能同时关注不同表示子空间、不同位置的信息；单头时，加权平均会把这些信息混在一起。每个头一个 softmax，就能有一个头看前一个 token、另一个头看句首、另一个头看匹配的括号。由于每个头的维度变成 d/h，总计算量和单头全维度注意力相近。原论文用 h = 8、d_k = 64。

### 3.5 MHA、MQA、GQA：参数和 KV 对比

解码时每一步都要读出所有历史 token 的 K 和 V，瓶颈常常在显存带宽，而不是算力（推导见 [inference/01 第 2.3 节](../../tracks/inference/01-kv-cache-memory/README.md#23-每步读写多少字节为什么-decode-受带宽限制)）。于是出现了减少 KV 头数的方案：查询仍是 n_h 个头，K、V 只有 n_kv 个头，每 n_h/n_kv 个查询头共享一组 K、V。

每层的参数和每 token 每层缓存的元素数（d_h = d/n_h）：

| 方案 | W_Q | W_K + W_V | W_O | 每层参数 | 每 token 每层 KV 元素 |
|---|---|---|---|---|---|
| MHA（n_kv = n_h） | d² | 2d² | d² | 4d² | 2·n_h·d_h = 2d |
| GQA（1 < n_kv < n_h） | d² | 2d²·n_kv/n_h | d² | (2 + 2n_kv/n_h)·d² | 2·n_kv·d_h |
| MQA（n_kv = 1） | d² | 2d²/n_h | d² | ≈ 2d² | 2·d_h |

Llama 3 8B 是 n_h = 32、n_kv = 8，每层注意力参数是 2.5d²，KV 是 MHA 的 1/4。

| 方案 | 出处 | 做法 | 代价 |
|---|---|---|---|
| MHA | Vaswani 等，2017-06 | 每个查询头有自己的 K、V 头 | KV Cache 最大 |
| MQA | [Shazeer，1911.02150](https://arxiv.org/abs/1911.02150)，2019-11 | 所有查询头共享一组 K、V | 解码快，质量有一定损失 |
| GQA | [Ainslie 等，2305.13245](https://arxiv.org/abs/2305.13245)，2023-05 | 查询头分组，每组共享一组 K、V，组数介于 1 和头数之间 | 质量接近 MHA，速度接近 MQA |
| MLA | [DeepSeek-V2，2405.04434](https://arxiv.org/abs/2405.04434)，2024-05 | 把 K、V 压缩成一个低维潜向量再缓存 | 实现更复杂，和 RoPE 结合要额外处理 |

两个可以核对的数字：GQA 论文提出可以用原预训练算力的 5% 把已有的 MHA 模型继续训练（uptraining）成 MQA 或 GQA；DeepSeek-V2 报告和 DeepSeek 67B 相比 KV Cache 减少 93.3%。Llama 3 的 8B、70B、405B 都用 GQA，KV 头数都是 8。

MQA、GQA 减少的主要是 KV 的显存和读取带宽，注意力打分的计算量不变：每个查询头仍要和全部历史的 K 做点积。MLA 的吸收技巧、GQA 的 KV 头复制和上训练的均值池化，见 [inference/01 第 4 节](../../tracks/inference/01-kv-cache-memory/README.md#4-mqa--gqa--mla-对-kv-的影响)。

### 3.6 实现：缩放、因果 mask 和 GQA

下面的代码验证四件事：点积方差等于 d_k；不缩放时 softmax 饱和，雅可比变小；带因果 mask 的多头注意力里，MHA、GQA、MQA 只差在 K、V 的头数，改最后一个 token 不影响前面位置；不带 mask 和位置编码时注意力置换等变。

```python
import numpy as np

rng = np.random.default_rng(0)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)          # 减最大值防溢出，不改变结果
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# 1) 点积方差：q、k 各分量独立、均值 0、方差 1 时，q·k 的方差是 d_k
for d_k in [16, 128, 1024]:
    q = rng.standard_normal((20000, d_k))
    k = rng.standard_normal((20000, d_k))
    dots = (q * k).sum(-1)                           # 形状 (20000,)
    assert abs(dots.var() / d_k - 1) < 0.05          # 未缩放：方差 ≈ d_k
    assert abs((dots / np.sqrt(d_k)).var() - 1) < 0.05   # 除以 √d_k 后方差 ≈ 1

# 2) softmax 饱和：一个查询对 n 个键打分。不缩放时最大权重接近 1，熵接近 0
def stats(d_k, scale, n=64, trials=200):
    maxw, ent, grad = [], [], []
    for _ in range(trials):
        q = rng.standard_normal(d_k)
        K = rng.standard_normal((n, d_k))
        s = K @ q / (np.sqrt(d_k) if scale else 1.0) # 打分，形状 (n,)
        p = softmax(s)
        maxw.append(p.max())
        ent.append(-(p * np.log(p + 1e-30)).sum())
        J = np.diag(p) - np.outer(p, p)              # softmax 的雅可比 ∂p/∂s，形状 (n, n)
        grad.append(np.abs(J).sum())
    return np.mean(maxw), np.mean(ent), np.mean(grad)

for d_k in [64, 512]:
    m0, e0, g0 = stats(d_k, scale=False)
    m1, e1, g1 = stats(d_k, scale=True)
    print(f"d_k={d_k}: 不缩放 最大权重 {m0:.3f} 熵 {e0:.3f} 雅可比 {g0:.4f}；"
          f"缩放 最大权重 {m1:.3f} 熵 {e1:.3f} 雅可比 {g1:.4f}")
    assert m0 > 0.75 and m1 < 0.3                    # 不缩放几乎是 one-hot
    assert g0 < g1 / 3                               # 梯度（雅可比的大小）小很多
# 缩放后的统计量和 d_k 无关：打分方差都是 1
assert abs(stats(64, True)[1] - stats(512, True)[1]) < 0.1

# 3) 带因果 mask 的多头注意力，以及 MHA / GQA / MQA 只差在 K、V 的头数
def attention(x, Wq, Wk, Wv, Wo, n_h, n_kv):
    T, d = x.shape
    d_h = d // n_h
    Q = (x @ Wq).reshape(T, n_h, d_h).transpose(1, 0, 2)    # (n_h, T, d_h)
    K = (x @ Wk).reshape(T, n_kv, d_h).transpose(1, 0, 2)   # (n_kv, T, d_h)
    V = (x @ Wv).reshape(T, n_kv, d_h).transpose(1, 0, 2)
    K = np.repeat(K, n_h // n_kv, axis=0)                   # 每组 n_h/n_kv 个查询头共用一个 KV 头
    V = np.repeat(V, n_h // n_kv, axis=0)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(d_h)             # (n_h, T, T)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)        # j > i 的位置是未来
    S = np.where(mask, -np.inf, S)
    A = softmax(S)                                          # 每行对 j ≤ i 归一化
    O = (A @ V).transpose(1, 0, 2).reshape(T, d)            # 拼接各头
    return O @ Wo, A

T, d, n_h = 6, 32, 8
d_h = d // n_h
x = rng.standard_normal((T, d))
Wq, Wo = rng.standard_normal((d, d)) / np.sqrt(d), rng.standard_normal((d, d)) / np.sqrt(d)
for n_kv in [8, 2, 1]:                                      # MHA、GQA、MQA
    Wk = rng.standard_normal((d, n_kv * d_h)) / np.sqrt(d)
    Wv = rng.standard_normal((d, n_kv * d_h)) / np.sqrt(d)
    out, A = attention(x, Wq, Wk, Wv, Wo, n_h, n_kv)
    assert np.allclose(A.sum(-1), 1) and np.all(np.triu(A[0], 1) == 0)   # 看不到未来
    # 因果性：改最后一个 token，前面位置的输出一个数都不变
    x2 = x.copy(); x2[-1] += 10
    out2, _ = attention(x2, Wq, Wk, Wv, Wo, n_h, n_kv)
    assert np.allclose(out[:-1], out2[:-1]) and not np.allclose(out[-1], out2[-1])
    # K、V 投影的参数量和每 token 缓存都按 n_kv / n_h 缩小
    assert Wk.size == d * d * n_kv // n_h

# 4) 不带 mask、不带位置编码的注意力对输入顺序是置换等变的：打乱输入，输出跟着同样打乱
def attn_nomask(x):
    Wk = np.eye(d); S = x @ Wq @ (x @ Wk).T / np.sqrt(d)
    return softmax(S) @ x
perm = rng.permutation(T)
assert np.allclose(attn_nomask(x)[perm], attn_nomask(x[perm]))
```

结果：d_k = 512 时，不缩放的最大注意力权重平均 0.93，熵 0.18，雅可比的绝对值和约 0.20；缩放后最大权重约 0.11，熵约 3.68（64 个键的均匀分布熵是 ln 64 ≈ 4.16），雅可比约 1.92，大了近 10 倍。缩放后的统计量在 d_k = 64 和 512 下几乎一样，这就是“方差回到 1、和维度无关”的效果。

## 4 位置编码与长度外推

### 4.1 问题：注意力本身对顺序不敏感

第 3.6 节验证了：不带 mask 和位置编码时，打乱输入，每个 token 得到的输出不变。因果 mask 会带来一些隐式的位置信息，但实践中仍加显式位置编码。

| 方案 | 出处 | 做法 | 外推能力 |
|---|---|---|---|
| 正弦绝对位置 | Vaswani 等，2017 | 每个位置一个固定向量，加到输入上 | 弱 |
| 可学习绝对位置 | 早期 GPT、BERT | 每个位置一个可训练向量 | 超过训练长度就没有对应向量 |
| ALiBi | [Press 等，2108.12409](https://arxiv.org/abs/2108.12409)，2021-08 | 不加位置向量，按距离给注意力打分加线性惩罚 | 论文中 1.3B 模型在 1024 长度上训练，在 2048 长度上困惑度和直接用 2048 训练的正弦模型相当 |
| RoPE | [Su 等，2104.09864](https://arxiv.org/abs/2104.09864)，2021-04 | 按位置把 Q、K 的每两维旋转一个角度 | 训练长度以外会退化，需要扩展方法 |

### 4.2 RoPE：旋转推导

我们想要一个变换 f(x, m)，作用在位置 m 的向量上，使得变换后的内积只依赖相对位置 m − n：

```text
⟨f(q, m), f(k, n)⟩ = g(q, k, m − n)
```

**二维情况**。把二维向量看成复数 q̃ = q_0 + i·q_1。令 f(q, m) = q̃ · e^{imθ}，也就是把向量逆时针转 mθ 弧度。两个复数 a、b 对应的二维向量内积是 Re[a · conj(b)]，于是

```text
⟨f(q, m), f(k, n)⟩ = Re[ q̃ e^{imθ} · conj(k̃ e^{inθ}) ]
                   = Re[ q̃ · conj(k̃) · e^{i(m−n)θ} ]
```

m 和 n 只以 m − n 的形式出现。原因很直接：旋转是正交变换，R(m)ᵀR(n) = R(n − m)，两边各转一次，内积里只剩下相对的那一段转角。旋转不改变向量长度，所以 RoPE 不改变 q、k 的范数。

写成实数矩阵就是

```text
R(mθ) = [ cos mθ   −sin mθ ]
        [ sin mθ    cos mθ ]
```

**高维情况**。把 d_h 维向量按相邻两维分成 d_h/2 对，第 i 对用自己的频率 θ_i 旋转：

```text
θ_i = base^(−2i/d_h)，i = 0, 1, …, d_h/2 − 1          原论文 base = 10000

⟨R(m)q, R(n)k⟩ = Σ_i Re[ q̃_i · conj(k̃_i) · e^{i(m−n)θ_i} ]
```

每一对的内积都只依赖 m − n，求和后仍然只依赖 m − n。RoPE 作用在每一层的 q、k 上，不加到输入嵌入上，也不作用在 v 上。

**频率谱**。第 0 对 θ_0 = 1，每个位置转 1 弧度，波长 λ_0 = 2π ≈ 6.3 个 token。最后一对 θ 接近 1/base，波长很长。Llama 3 用 base = 500,000、d_h = 128，最长波长约 250 万个 token（第 9 节代码）。高频对分辨相邻 token，低频对编码远距离的粗略位置。原论文还证明了这种频率设置下，内积的上界随相对距离增大而衰减（论文称为 long-term decay）。

**实现上的一个坑**：原论文和上面的推导是相邻两维 (2i, 2i+1) 配对。Hugging Face 的 Llama 实现用 `rotate_half`，把前一半和后一半配对（第 i 维和第 i + d_h/2 维）。两种写法数学上等价，但权重的排列不同；Hugging Face 的转换脚本在导入 Meta 原始权重时，对 W_Q、W_K 做了一次 `permute` 重排。自己拼权重、混用两种实现时，结果会悄悄出错。

### 4.3 RoPE 为什么不能直接外推

内积只依赖相对位置，看起来应该能处理任意长度。实际上超过训练长度后效果会明显变差。

用波长来看。训练长度为 L 时，波长 λ_i < L 的维度对，在训练中已经转过完整的若干圈，每个角度都见过；波长 λ_i > L 的维度对，在训练中连一圈都没转满，只见过 [0, L·θ_i) 这一段角度。推理时超过 L，这些低频维度出现训练时从未见过的角度，注意力打分偏离训练分布。PI 论文还指出，直接外推可能产生极大的注意力分数。

YaRN 论文用的就是这个视角：一些维度的波长比预训练最大长度还长，这些维度实际上在编码绝对位置。以 d_h = 128、base = 10000、训练长度 4096 为例，64 对维度里有 18 对转不满一圈（第 4.6 节代码，推导）。

ALiBi 走的是另一条路：不旋转，直接在打分上加 −m·(i − j)，m 是每个头固定的斜率。8 个头时斜率是 1/2、1/4、…、1/256 的几何序列。距离越远惩罚越大，偏向近处的 token。它没有“没见过的角度”这个问题，所以外推更平滑，代价是远距离信息被线性压低。

### 4.4 长度扩展：PI、NTK、YaRN

三种方法都在改“位置 → 角度”的映射，区别在于改哪些频率、改多少。设训练长度为 L，目标长度为 L'，扩展倍数 s = L'/L。

**Position Interpolation（PI）**（[Chen 等，2306.15595](https://arxiv.org/abs/2306.15595)）：把位置下标线性缩小，m → m/s。新长度 [0, L') 的位置全部落回训练过的范围 [0, L)，没有外推。代价是所有频率都被压小 s 倍：最高频那对，相邻 token 的角度差从 1 弧度变成 1/s 弧度，模型分辨近邻 token 的能力被削弱，需要微调恢复。论文把 LLaMA 扩到 32768，在 Pile 上微调 1000 步；论文还推导出插值后注意力分数的上界比外推小约 600 倍。

**NTK-aware**（开源社区提出，YaRN 论文附录 A.2 给出形式）：不动位置，改底数。新底数 b' = b · s^(d_h/(d_h−2))。这个式子的推导条件是：最高频（i = 0）不变，最低频（最后一对）正好缩小 s 倍，和 PI 一样。中间的频率按几何级数过渡，高频改得少，低频改得多。缺点是最优底数难定，有些维度会被轻微外推，所以实际要把 s 设得比目标倍数大。Code Llama 把底数手动调到 1M，也是这个思路。

**YaRN**（[Peng 等，2309.00071](https://arxiv.org/abs/2309.00071)）由两部分组成：

1. **NTK-by-parts 插值**。按每对维度在训练长度内转的圈数 r = L/λ 分三段：r < α（转不满一圈的低频维度）完全按 PI 插值；r > β（转了很多圈的高频维度）不插值；中间用线性斜坡过渡。论文给 Llama 系列的取值是 α = 1、β = 32。
2. **注意力温度**。在 softmax 前把打分除以 t，论文推荐 √(1/t) = 0.1·ln(s) + 1，s = 8 时约 1.208。实现时把 q、k 都乘 √(1/t)，等价于打分乘 1/t，可以直接并进 RoPE 的 cos、sin 表，推理时零开销。

论文称 YaRN 所需 token 比此前方法少 10 倍、训练步数少 2.5 倍。实验里，s = 16 的模型在 64K 长度的数据上微调 400 步；s = 32 的模型从 s = 16 的检查点再训 200 步，同样只用 64K 数据，能外推到 128K。

另一个方向是预训练时就把底数调大。Llama 3 把 RoPE 底数调到 500,000，技术报告说这样能更好地支持长上下文。

### 4.5 三个长度

面试时要分清三个长度：训练时的原生长度、靠外推方法支持的长度、模型真正能有效利用信息的长度。第三个通常最短，见 [03 章的 Lost in the Middle](../03-prompt-context/README.md)。“支持 128K”只说明在这个长度上不崩，不说明中间的信息都能被用上。要用大海捞针或多跳检索类测试实测。

### 4.6 实现：RoPE、PI、NTK、YaRN、ALiBi

下面的代码实现原论文的相邻配对 RoPE，assert 内积只依赖相对位置、和复数闭式一致；再分别实现 PI、NTK-aware 底数变换、YaRN 的分段插值，assert 各自改了哪些频率；最后实现 ALiBi 的斜率。

```python
import numpy as np

rng = np.random.default_rng(0)

def rope_freqs(d_h, base=10000.0):
    """θ_i = base^(-2i/d_h)，i = 0..d_h/2-1。第 i 对维度每前进一个位置转 θ_i 弧度。"""
    return base ** (-np.arange(0, d_h, 2) / d_h)            # 形状 (d_h/2,)

def rope(x, pos, theta):
    """x: (..., d_h)，相邻两维 (x[2i], x[2i+1]) 组成一对，按角度 pos·θ_i 旋转。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    ang = pos * theta
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

d_h = 64
theta = rope_freqs(d_h)
q, k = rng.standard_normal(d_h), rng.standard_normal(d_h)

# 1) 旋转不改长度
assert np.isclose(np.linalg.norm(rope(q, 37, theta)), np.linalg.norm(q))

# 2) 内积只依赖相对位置 m - n：(m, n) 整体平移任意距离，打分不变
def score(m, n):
    return rope(q, m, theta) @ rope(k, n, theta)
for m, n in [(5, 2), (100, 97), (4000, 3997)]:
    assert np.isclose(score(m, n), score(3, 0))
assert not np.isclose(score(5, 2), score(5, 3))             # 相对距离变了，打分就变

# 3) 闭式：<R(m)q, R(n)k> = Σ_i Re[ q̃_i · conj(k̃_i) · e^{i(m-n)θ_i} ]，q̃_i = q[2i] + i·q[2i+1]
qc, kc = q[0::2] + 1j * q[1::2], k[0::2] + 1j * k[1::2]
for m, n in [(7, 3), (3, 7), (50, 0)]:
    assert np.isclose(score(m, n), np.real((qc * np.conj(kc) * np.exp(1j * (m - n) * theta)).sum()))

# 4) 频率谱：第 0 对每个位置转 1 弧度，最后一对几乎不转。波长 λ_i = 2π/θ_i
lam = 2 * np.pi / theta
assert np.isclose(theta[0], 1.0) and lam[-1] > 40_000
# d_h = 128、底数 10000、训练长度 4096 的配置：有多少对维度在训练长度内转不满一圈（推导）
lam128 = 2 * np.pi / rope_freqs(128, 10000.0)
assert (lam128 > 4096).sum() == 18
# Llama 3 8B：底数 500,000，d_h = 128，config 里的训练长度 8192：29 对转不满一圈（推导）
assert (2 * np.pi / rope_freqs(128, 500000.0) > 8192).sum() == 29

# 5) Position Interpolation：位置乘 L/L'，新长度的位置全部落回训练范围 [0, L)
L, L_new = 2048, 8192
s = L_new / L
pos = np.arange(L_new)
assert (pos / s).max() < L
# 代价：相邻 token 在最高频维度上的角度差从 1 弧度缩成 1/s 弧度
assert np.isclose((1 / s) * theta[0], 0.25)

# 6) NTK-aware：改底数 b' = b · s^(d/(d-2))，最高频不变，最低频正好缩小 s 倍（YaRN 附录 A.2）
b_new = 10000.0 * s ** (d_h / (d_h - 2))
th_ntk = rope_freqs(d_h, b_new)
assert np.isclose(th_ntk[0], theta[0])                       # 高频保持
assert np.isclose(th_ntk[-1], theta[-1] / s)                 # 最低频等于线性插值

# 7) YaRN 的 NTK-by-parts：r = L/λ 是训练长度内转了几圈。r < α 完全按 PI 插值，r > β 不动
alpha, beta = 1, 32                                          # 论文给 Llama 系列的取值
r = L / lam
gamma = np.clip((r - alpha) / (beta - alpha), 0, 1)
th_yarn = (1 - gamma) * theta / s + gamma * theta
assert np.isclose(th_yarn[0], theta[0])                      # 高频（转很多圈）不插值
assert np.isclose(th_yarn[-1], theta[-1] / s)                # 低频（转不满一圈）全插值
# 注意力温度：sqrt(1/t) = 0.1·ln(s) + 1，s = 8 时约 1.208（论文式 23）
assert round(0.1 * np.log(8) + 1, 3) == 1.208

# 8) ALiBi：不旋转，打分加 -m·(i - j)。8 个头的斜率是 1/2, 1/4, ..., 1/256
n_heads = 8
slopes = 2.0 ** (-8 * np.arange(1, n_heads + 1) / n_heads)
assert np.allclose(slopes, [1 / 2 ** i for i in range(1, 9)])
i = 10
bias = -slopes[:, None] * (i - np.arange(i + 1))             # (头, j=0..i)
assert np.all(bias[:, -1] == 0) and np.all(np.diff(bias, axis=1) > 0)   # 越近惩罚越小
```

## 5 Transformer 块：归一化、FFN、参数与 FLOPs

### 5.1 一个块的结构：Pre-LN 和 Post-LN

解码器由 L 个相同的块堆成。每个块有两个子层：注意力和 FFN（前馈网络），每个子层外面包一个残差连接。区别在归一化放在哪里：

```text
Post-LN（原始 Transformer）：  h = LN(h + Sublayer(h))
Pre-LN（GPT-2 起的主流）：     h = h + Sublayer(LN(h))         最后一个块之后再加一个 LN
```

原论文写的是 `LayerNorm(x + Sublayer(x))`，即 Post-LN。[Xiong 等，2002.04745](https://arxiv.org/abs/2002.04745) 用平均场理论证明：初始化时，Post-LN 靠近输出层的参数梯度的期望很大，直接用大学习率训练不稳定，所以必须有学习率 warm-up；Pre-LN 初始化时梯度表现良好，论文实验里去掉 warm-up 也能达到可比的效果，训练时间和调参都更少。

直观的原因在残差路径上。Pre-LN 展开后：

```text
h_L = h_0 + Σ_{l=1..L} Sublayer_l(LN(h_{l−1}))
```

输入 h_0 经过一条恒等路径直通到最后一层，梯度也能沿这条路径直接回传。Post-LN 每一层都对残差和做一次归一化，恒等路径被 L 次归一化打断。Pre-LN 的代价是残差流上没有归一化，它的范数随深度增长，深层子层的贡献相对变小。所以最后要再加一个 LN，有些模型还会调整初始化或加额外的归一化。

现在的主流开源模型（Llama、Qwen 等）都用 Pre-LN，归一化层换成 RMSNorm。Hugging Face 的 Llama 实现里，每层有 `input_layernorm`（注意力前）和 `post_attention_layernorm`（FFN 前），两个都作用在子层输入上，名字里的 post 指“注意力之后”，不是 Post-LN。

### 5.2 RMSNorm

LayerNorm 对每个 token 的 d 维向量减均值、除标准差，再乘增益 g、加偏置 b：

```text
LayerNorm(x) = (x − μ) / √(σ² + ε) ⊙ g + b         μ、σ² 是 x 在 d 维上的均值和方差
RMSNorm(x)   = x / √(mean(x²) + ε) ⊙ g              没有减均值，没有偏置
```

[Zhang 和 Sennrich，1910.07467](https://arxiv.org/abs/1910.07467) 的假设是：LayerNorm 的“重新居中”不变性（对整体平移不变）可以去掉，只保留“重新缩放”不变性。去掉减均值后少一次归约，论文报告在不同模型上运行时间减少 7% 到 64%，效果相当。x 的均值为 0 时，两者完全相同。

实现细节：Hugging Face 的 `LlamaRMSNorm` 先把输入转成 float32 算方差，再转回原精度。BF16 下直接平方求均值，精度损失会累积。

### 5.3 SwiGLU

原始 Transformer 的 FFN 是两个矩阵加一个激活：

```text
FFN(x)    = ReLU(x W_1) W_2                         W_1: d × d_ff，W_2: d_ff × d，d_ff = 4d
FFN_SwiGLU(x) = (SiLU(x W_gate) ⊙ x W_up) W_down     SiLU(z) = z · sigmoid(z)
```

SwiGLU 出自 [Shazeer，2002.05202](https://arxiv.org/abs/2002.05202)。gate 分支经过 SiLU 后逐元素乘 up 分支，相当于让网络自己学一个“门”。它有三个矩阵，为了和两矩阵 FFN 的参数量、计算量对齐，论文把 d_ff 缩小到原来的 2/3。论文实验里 d_ff 从 3072 改成 2048，GEGLU 和 SwiGLU 的困惑度最好。

Llama 3 的 d_ff 就是这么来的：4d = 16384，取 2/3 得 10922，乘 `ffn_dim_multiplier` = 1.3 得 14198，向上取整到 `multiple_of` = 1024 的倍数，得 14336（Meta `llama-models` 仓库里的计算方式，第 5.7 节代码复现）。

### 5.4 实现：归一化、SwiGLU、Pre-LN 和 Post-LN

下面的代码验证：LayerNorm 和 RMSNorm 都对缩放不变，只有 LayerNorm 对平移不变，均值为 0 时两者相等；宽度取 2/3·4d 的 SwiGLU 和 4d 的两矩阵 FFN 参数量相同；48 个随机块堆起来，Post-LN 每层输出的范数都被拉回 √d，Pre-LN 残差流的范数随深度增长，子层为 0 时 Pre-LN 原样保留输入。

```python
import numpy as np

rng = np.random.default_rng(0)
d = 64

def layernorm(x, g, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g

def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g      # 不减均值，没有偏置

g = rng.uniform(0.5, 1.5, d)
x = rng.standard_normal(d) * 3 + 2

# 1) 两者都对整体缩放不变；LayerNorm 还对整体平移不变，RMSNorm 不是
assert np.allclose(layernorm(5 * x, g), layernorm(x, g))
assert np.allclose(rmsnorm(5 * x, g), rmsnorm(x, g))
assert np.allclose(layernorm(x + 7, g), layernorm(x, g))
assert not np.allclose(rmsnorm(x + 7, g), rmsnorm(x, g))
# 均值为 0 时两者相同：RMSNorm 去掉的只是“减均值”这一步
x0 = x - x.mean()
assert np.allclose(layernorm(x0, g, eps=0), rmsnorm(x0, g, eps=0))
# 参数：LayerNorm 有增益和偏置 2d 个，RMSNorm（Llama 式实现）只有增益 d 个

# 2) SwiGLU：gate 分支过 SiLU 后逐元素乘 up 分支，再过 down。三个矩阵
def silu(z):
    return z / (1 + np.exp(-z))
d_ff = 172
W_gate, W_up = rng.standard_normal((d, d_ff)) * 0.1, rng.standard_normal((d, d_ff)) * 0.1
W_down = rng.standard_normal((d_ff, d)) * 0.1
y = (silu(x @ W_gate) * (x @ W_up)) @ W_down
assert y.shape == (d,)
# 宽度取 2/3·4d 时，三矩阵 SwiGLU 和 d_ff = 4d 的两矩阵 FFN 参数量相同
assert 3 * d * (2 * 4 * d // 3) <= 2 * d * 4 * d < 3 * d * (2 * 4 * d // 3 + 1)

# 3) Pre-LN 和 Post-LN：同样 48 个随机初始化的残差块
L = 48
Ws = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(L)]
ones = np.ones(d)
def f(h, W):                                                            # 一个简化的子层
    return np.tanh(h @ W)

def post_ln(h):
    norms = []
    for W in Ws:
        h = layernorm(h + f(h, W), ones)                               # 残差之后归一化
        norms.append(np.linalg.norm(h))
    return h, norms

def pre_ln(h):
    norms = []
    for W in Ws:
        h = h + f(layernorm(h, ones), W)                               # 只归一化子层的输入
        norms.append(np.linalg.norm(h))
    return layernorm(h, ones), norms                                    # 最后要再归一化一次

h0 = rng.standard_normal(d)
_, n_post = post_ln(h0)
_, n_pre = pre_ln(h0)
# Post-LN：每块之后范数都被拉回 √d；Pre-LN：残差流上没有归一化，范数随深度增长
assert np.allclose(n_post, np.sqrt(d), rtol=1e-3)
assert n_pre[-1] > 3 * n_pre[0]
# 恒等路径：Pre-LN 里第 l 块的输出 = 输入 + Σ 各块子层输出，输入经残差直通到最后一层。
# 把所有子层置 0 时，Pre-LN 的残差流原样保留输入；Post-LN 每层都做一次 LN，输入被反复改写
Ws_saved = list(Ws)
Ws[:] = [np.zeros((d, d))] * L
h_pre = h0.copy()
for W in Ws:
    h_pre = h_pre + f(layernorm(h_pre, ones), W)
assert np.allclose(h_pre, h0)
assert not np.allclose(post_ln(h0)[0], h0)
Ws[:] = Ws_saved
```

### 5.5 参数量估算：每层约 12d²

Kaplan 等（[2001.08361](https://arxiv.org/abs/2001.08361)）式 2.1 给出的经典估算，前提是 MHA、两矩阵 FFN、d_attn = d、d_ff = 4d，不算偏置和归一化：

```text
注意力：W_Q、W_K、W_V、W_O 各 d × d              4d²
FFN：   W_1 是 d × 4d，W_2 是 4d × d             8d²
每层                                            12d²
非嵌入参数 N ≈ 12 · L · d²
```

嵌入（V × d）和位置嵌入不算进 N。Kaplan 发现不算嵌入时，loss 和 N 的幂律关系更干净。

现代模型要修正两处：

1. **GQA 让注意力变小**：W_K、W_V 只有 d × (n_kv·d_h)。Llama 3 8B 是 n_kv/n_h = 1/4，注意力 = (1 + 0.25 + 0.25 + 1)·d² = 2.5d²。
2. **SwiGLU 的宽度不是 4d**：三个矩阵共 3·d·d_ff。Llama 3 8B 的 d_ff = 14336 = 3.5d，FFN = 10.5d²。

合计 13d²，套 12d² 会低估约 8%。所以面试估算可以先说 12d²，再说明 GQA 和 SwiGLU 怎么改它。嵌入不能随手忽略：Llama 3 8B 的嵌入加输出头 1.05B，占总参数 13%。

### 5.6 FLOPs 估算：每 token 前向约 2N，训练约 6N

**前向 2N**。矩阵乘 y = xW，W 是 m × n，每个输出元素是 m 次乘加，共 m·n 次乘加。每次乘加记 2 次浮点运算（一次乘，一次加），所以每个参数每个 token 贡献 2 FLOPs。线性层合起来是 2N。

**注意力的额外项**。打分 q·kᵀ 和加权 Σ a_j v_j 不涉及参数，不在 N 里。位置 t 的一个头对 t 个键打分是 t·d_h 次乘加，加权求和也是 t·d_h 次乘加。所有头、所有层合计 4·L·d_attn·t FLOPs。Kaplan 的式 2.2 是 C_forward ≈ 2N + 2·n_layer·n_ctx·d_attn，系数是 2 而不是 4，因为它按因果 mask 下平均每个位置只看一半上下文计算（第 9 节代码验证这两个口径一致）。论文指出，d_model > n_ctx/12 时上下文项占比较小，所以训练算力估算里忽略了它。上下文很长时不能忽略：Llama 3 8B 在 8K 位置上，注意力约占每 token 前向的 22%（第 5.7 节代码，推导）。

**输入嵌入和输出头**。输入嵌入是查表，不做乘法；输出头是一次 d × V 的矩阵乘，要算 2·d·V。

**训练 6N**。反向传播要算两份梯度：对激活的梯度（往前传）和对权重的梯度（用于更新），各一次和前向同规模的矩阵乘，约是前向的 2 倍。所以训练每 token 约 2N + 4N = 6N，整个训练

```text
C ≈ 6 · N · D          D 是训练 token 数
```

Kaplan 用 C ≈ 6NBS（B 是批大小，S 是步数，BS 就是 D）。Chinchilla 论文附录 F 按每个操作精确计数，和 6ND 的比值在 0.99 到 1.10 之间，论文结论是差别很小、不影响分析。

**推理和训练的区别**：推理只有前向。Prefill T 个 token 约 2N·T 加注意力项；Decode 每步 2N 加注意力项，但每步要读一遍全部权重和 KV，瓶颈在带宽，见 [inference/01 第 2.2 节](../../tracks/inference/01-kv-cache-memory/README.md#22-每步的计算量有缓存和没缓存)。

### 5.7 实现：从 config 数参数量，和官方数字对上

下面的代码读 Llama 3 8B 和 Qwen2.5-7B 的 config.json 字段，按上面的公式数参数，assert 和 Hugging Face 上 safetensors 的参数总数一个不差；再复现 d_ff = 14336 的来历、12d² 和 13d² 的差别，以及 2N、6ND 的几个数字。Qwen2.5 的注意力有 QKV 偏置（模型卡写明），要算进去。

```python
# 原样摘自 Hugging Face config.json（2026-11 查阅），只保留估算要用的字段。
# Llama 3 8B 取自与官方权重同结构的镜像 unsloth/llama-3-8b（官方仓库需要登录）。
LLAMA3_8B = {"hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
             "num_attention_heads": 32, "num_key_value_heads": 8, "vocab_size": 128256,
             "tie_word_embeddings": False, "attention_bias": False}
QWEN25_7B = {"hidden_size": 3584, "intermediate_size": 18944, "num_hidden_layers": 28,
             "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064,
             "tie_word_embeddings": False, "attention_bias": True}   # 模型卡：Attention QKV bias
# Hugging Face API 里 safetensors 的 BF16 参数总数（2026-11 查阅）
OFFICIAL = {"llama3_8b": 8_030_261_248, "qwen25_7b": 7_615_616_512}


def count(c):
    d, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    nh, nk, ff = c["num_attention_heads"], c["num_key_value_heads"], c["intermediate_size"]
    dh = d // nh
    attn = d * nh * dh + 2 * d * nk * dh + nh * dh * d                 # Wq, Wk, Wv, Wo
    if c["attention_bias"]:
        attn += nh * dh + 2 * nk * dh                                   # q、k、v 的偏置
    mlp = 3 * d * ff                                                    # SwiGLU：gate、up、down
    norms = 2 * d                                                       # 每层两个 RMSNorm 的增益
    body = L * (attn + mlp + norms) + d                                 # 加最后一个 RMSNorm
    emb = V * d * (1 if c["tie_word_embeddings"] else 2)                # 输入嵌入 + 输出头
    return body + emb, body, emb, attn, mlp


for name, c in [("llama3_8b", LLAMA3_8B), ("qwen25_7b", QWEN25_7B)]:
    total, body, emb, attn, mlp = count(c)
    print(f"{name}: 总 {total:,}，非嵌入 {body/1e9:.2f}B，嵌入 {emb/1e9:.2f}B，占 {emb/total:.1%}")
    assert total == OFFICIAL[name]                                      # 一个参数都不差

# 1) Llama 3 8B 的 FFN 宽度从 4d 推出来：取 2/3（三个矩阵代替两个），乘 1.3，向上取整到 1024 的倍数
d = 4096
h = int(2 * 4 * d / 3)
h = int(1.3 * h)
h = 1024 * ((h + 1024 - 1) // 1024)
assert h == LLAMA3_8B["intermediate_size"] == 14336

# 2) 经典估算“每层 12d²”的前提：MHA、两矩阵 FFN、d_ff = 4d → 4d²（注意力）+ 8d²（FFN）
d = 4096
assert 4 * d * d + 2 * d * (4 * d) == 12 * d * d
# 套到 Llama 3 8B：GQA 让注意力只有 2.5d²，SwiGLU 的 3·d·14336 = 10.5d²，合计 13d²
_, body, emb, attn, mlp = count(LLAMA3_8B)
assert attn == 2.5 * d * d and mlp == 10.5 * d * d
assert abs(32 * 12 * d * d / body - 0.92) < 0.01                       # 12d² 低估约 8%
assert abs(32 * 13 * d * d / body - 1) < 0.001
# 嵌入不能忽略：词表 128K 时嵌入加输出头约 1.05B，占总参数 13%
assert round(emb / 1e9, 2) == 1.05

# 3) 每 token 前向 FLOPs ≈ 2N（每个参数一次乘加）+ 注意力打分和加权 4·L·d_attn·T
#    输入嵌入是查表，不做乘法；输出头是一次 d×V 的矩阵乘，要算进去
total, body, emb, _, _ = count(LLAMA3_8B)
L, d_attn = 32, 4096
N_matmul = body + emb // 2                                             # 只算参与矩阵乘的参数
for T in [1, 8192]:
    F = 2 * N_matmul + 4 * L * d_attn * T
    print(f"T={T}: 每 token 前向 {F/1e9:.1f} GFLOPs，注意力占 {4*L*d_attn*T/F:.1%}")
assert round(2 * N_matmul / 1e9, 1) == 15.0                            # 约 2 × 7.5B
# 按这个式子，8K 上下文时注意力的打分和加权约占每 token 前向的 22%（推导）
F8k = 2 * N_matmul + 4 * L * d_attn * 8192
assert 0.22 < 4 * L * d_attn * 8192 / F8k < 0.23

# 4) 训练 C ≈ 6ND：前向 2N，反向约 2 倍。Chinchilla 70B × 1.4T，和 Gopher 的 5.76e23 同量级
C = 6 * 70e9 * 1.4e12
assert round(C / 1e23, 2) == 5.88 and abs(C / 5.76e23 - 1) < 0.03
# Llama 3 405B 用 15.6T token：6ND = 3.79e25，和技术报告给的 3.8e25 FLOPs 预算一致
assert round(6 * 405e9 * 15.6e12 / 1e25, 1) == 3.8
```

两个模型都对上了：Llama 3 8B 是 8,030,261,248，Qwen2.5-7B 是 7,615,616,512。Qwen2.5-7B 数出的非嵌入参数 6.53B，也和模型卡一致。数参数是验证 config 读法的好办法：漏掉偏置、搞错 head_dim、忘了输出头是否共享，都会差出一截。

## 6 解码策略

### 6.1 每一步从分布里选一个 token

模型每一步输出 V 个 logits z_1..z_V，softmax 后是词表上的概率分布。解码策略决定怎么从中选 token。

| 策略 | 做法 | 适合 | 代价 |
|---|---|---|---|
| 贪心 | 每步取概率最大的 token | 抽取、分类、工具参数 | 容易重复，缺少多样性 |
| Beam Search | 同时保留多个候选序列 | 翻译等输出较确定的任务 | 开放生成里容易变得平淡和重复 |
| 温度 | logits 除以 τ 再 softmax。τ 小于 1 分布变尖，大于 1 变平 | 调节随机程度 | τ 太高会出现不通顺的内容 |
| top-k | 只在概率最高的 k 个 token 里采样 | 截掉长尾 | k 固定，不随分布形状变化 |
| top-p（nucleus） | 只在累计概率达到 p 的最小集合里采样 | 开放生成 | 和温度一起调时效果不直观 |

### 6.2 温度

```text
p_i(τ) = exp(z_i / τ) / Σ_j exp(z_j / τ)
```

- τ = 1：原分布。
- τ → 0：最大 logit 和其他 logit 的差被放大到无穷，分布趋于 one-hot，等价于贪心。
- τ → ∞：所有 z_i/τ 趋于 0，分布趋于均匀。

温度不改变 token 的概率排序，所以贪心结果和温度无关。温度 0 在数学上是除以 0，框架都特殊处理：vLLM 的 `SamplingParams` 把温度小于 1e-5 当作贪心；0 到 0.01 之间的温度会被抬到 0.01，防止数值溢出。Hugging Face 的 `TemperatureLogitsWarper` 要求温度严格大于 0，贪心要用 `do_sample=False`。

### 6.3 top-k 和 top-p

**top-k**：把概率排序，只保留前 k 个，重新归一化后采样。[Fan 等，1805.04833](https://arxiv.org/abs/1805.04833) 在故事生成里用 k = 10 的 top-k 采样，比束搜索效果好。

**top-p**（[Holtzman 等，1904.09751](https://arxiv.org/abs/1904.09751)）：把概率从大到小排序，取累计概率第一次达到 p 的最小前缀：

```text
V_p = 最小的集合，使 Σ_{i∈V_p} p_i ≥ p
p'_i = p_i / Σ_{j∈V_p} p_j    若 i ∈ V_p，否则 0
```

论文指出，按最大似然解码（贪心、束搜索）会产生平淡、不连贯或陷入重复的文本；纯采样会从分布“不可靠的长尾”里采到怪词。概率的大部分质量集中在少数 token（核）上，从核里采样，能截掉长尾，又保留多样性。

top-p 的优点是保留的 token 数随分布形状变化。模型很确定时（一个 token 占 0.95），p = 0.9 只留 1 个；模型不确定时（分布平），留很多个。top-k 的 k 是固定的：确定时留了一堆不该留的，不确定时又截掉了合理的候选。

**组合顺序**。Hugging Face 的 `generate` 按温度 → top-k → top-p 的顺序应用这几个处理器。温度在前，意味着升温会让分布变平，同样的 p 会放进更多 token。两个参数一起调时效果不直观，一般固定一个调另一个。

### 6.4 实现：温度、top-k、top-p

下面的代码实现过滤后的采样分布，assert 温度的三个极限、排序不变；用一个已知分布手算 top-p = 0.8 保留 3 个 token；验证 top-p 的保留个数随分布形状变化、升温会放进更多 token；最后用 10 万次采样核对分布。

```python
import numpy as np

rng = np.random.default_rng(0)

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def filter_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """返回采样分布。顺序和常见推理框架一致：先除温度，再 top-k，再 top-p，最后重新归一化。"""
    if temperature == 0:                              # 约定：温度 0 等于贪心
        p = np.zeros_like(logits); p[np.argmax(logits)] = 1.0
        return p
    p = softmax(logits / temperature)
    keep = np.ones_like(p, dtype=bool)
    if top_k is not None:
        keep &= p >= np.sort(p)[-top_k]               # 并列时可能多留几个，示例里没有并列
    if top_p is not None:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        n = int(np.searchsorted(cum, top_p) + 1)      # 累计概率第一次 ≥ p 的最小前缀
        nucleus = np.zeros_like(keep); nucleus[order[:n]] = True
        keep &= nucleus
    p = np.where(keep, p, 0.0)
    return p / p.sum()

logits = np.log(np.array([0.50, 0.20, 0.15, 0.10, 0.04, 0.01]))   # 6 个 token，概率已知

# 1) 温度：T → 0 趋于 one-hot，T = 1 不变，T 大趋于均匀；排序不变，所以贪心结果和 T 无关
assert np.allclose(filter_probs(logits, 1.0), np.exp(logits))
assert filter_probs(logits, 0.1).max() > 0.99
assert np.ptp(filter_probs(logits, 100.0)) < 0.02
ent = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
assert ent(filter_probs(logits, 0.5)) < ent(filter_probs(logits, 1.0)) < ent(filter_probs(logits, 2.0))
assert all(np.argmax(filter_probs(logits, t)) == 0 for t in [0.1, 1, 10])

# 2) top-p = 0.8：0.50 + 0.20 = 0.70 < 0.8，再加 0.15 = 0.85 ≥ 0.8，保留 3 个
p = filter_probs(logits, top_p=0.8)
assert (p > 0).sum() == 3 and np.allclose(p[:3], np.array([0.50, 0.20, 0.15]) / 0.85)
# top-k 的 k 固定，top-p 留下的个数随分布形状变：尖分布只留 1 个，平分布留很多
sharp = np.log(np.array([0.95, 0.02, 0.01, 0.01, 0.005, 0.005]))
flat = np.zeros(6)
assert (filter_probs(sharp, top_p=0.9) > 0).sum() == 1
assert (filter_probs(flat, top_p=0.9) > 0).sum() == 6
# 温度和 top-p 有交互：先升温，分布变平，同样的 p 会放进更多 token
assert (filter_probs(logits, 2.0, top_p=0.8) > 0).sum() > (filter_probs(logits, 1.0, top_p=0.8) > 0).sum()

# 3) 蒙特卡洛核对：按 filter_probs 采样 10 万次，经验频率和分布一致
p = filter_probs(logits, temperature=0.7, top_k=4, top_p=0.9)
draws = rng.choice(len(p), size=100_000, p=p)
freq = np.bincount(draws, minlength=len(p)) / len(draws)
assert np.abs(freq - p).max() < 0.01 and freq[4:].sum() == 0

# 4) 温度 0 也不确定的根源：浮点加法不满足结合律，归约顺序变了结果就变
a, b, c = np.float32(1e8), np.float32(1.0), np.float32(-1e8)
assert (a + b) + c == 0.0 and (a + c) + b == 1.0
# 同一组数按不同块数分段求和（模拟不同批大小下内核选了不同的归约切分）
x = rng.standard_normal(4096).astype(np.float32)
def chunked_sum(v, n_chunks):
    total = np.float32(0)
    for ch in np.array_split(v, n_chunks):
        s = np.float32(0)
        for e in ch:
            s = np.float32(s + e)
        total = np.float32(total + s)
    return total
sums = {float(chunked_sum(x, n)) for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]}
assert len(sums) > 1                                   # 数学上相等，浮点上不等
# 两个 logit 只差 1 个 ulp（float32 在 3.0 附近的最小间隔约 2.4e-7）时，
# 归约顺序带来的一个 ulp 的误差就足以翻转 argmax，贪心选出另一个 token
l1 = np.float32(3.0)
l2 = np.nextafter(l1, np.float32(4.0))
ulp = l2 - l1
assert 2e-7 < ulp < 3e-7
assert np.argmax([l1, l2]) == 1 and np.argmax([l1 + ulp + ulp, l2]) == 0
```

### 6.5 温度设为 0 为什么结果还会变

[Thinking Machines，Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)（2025-09）的分析分三层：

1. **浮点加法不满足结合律**。(a + b) + c 和 (a + c) + b 在浮点下可以不等。上面代码里 (1e8 + 1) − 1e8 = 0，而 (1e8 − 1e8) + 1 = 1。同一组数按不同方式分块求和，结果可以差几个 ulp（最小可表示间隔）。
2. **单次前向本身是确定的**。常见的说法是“GPU 并发加原子操作导致随机”。博客的结论是这个假设不完整：LLM 前向里通常不用原子加，同样的输入、同样的批，逐次运行结果相同。
3. **真正的原因是缺少批不变性**。服务端的批大小随负载变化。矩阵乘、RMSNorm、注意力这三类涉及归约的内核，会按批大小选不同的切分和归约顺序。于是同一条请求的数值结果取决于它和谁拼在一批。当两个 token 的 logit 只差一个 ulp 时，贪心就会选出不同的 token，后面的生成全部分叉。

博客的实验：Qwen3-235B-A22B-Instruct-2507，温度 0，同一个提示生成 1000 次、每次 1000 token。默认内核下有 80 种不同结果，第一次分叉出现在第 103 个 token；改用批不变的内核后，1000 次全部一致。代价是性能：博客报告的端到端测试里，默认 vLLM 26 秒，未优化的确定性版本 55 秒，改进注意力内核后 42 秒。

对 Agent 的启示：不要假设温度 0 就能复现。需要复现时，记录完整轨迹并回放，见 [Trace 与回放](../../tracks/harness/06-tracing-replay/README.md)。

## 7 KV Cache 为什么存在，有多大

**问题**：自回归生成每一步都要对所有历史 token 做注意力。如果每一步都重新算历史 token 的 K 和 V，线性层的计算量会随长度平方增长。

**做法**：因果 mask 下，历史 token 的 K 和 V 不会因为后面追加 token 而改变（第 3.3 节），算一次就存下来。新 token 只算自己的 Q、K、V，再和缓存做注意力。推理因此分成两个阶段：

- **Prefill**：一次性处理整个输入，并行度高，主要受算力限制；
- **Decode**：每步只生成一个 token，要读出全部权重和整个 KV Cache，主要受显存带宽限制。

**大小**：

```text
每 token 字节数 = 2 (K 和 V) × 层数 × KV 头数 × 每头维度 × 每元素字节数
总大小 = 每 token 字节数 × 序列长度 × 并发请求数
```

以 Llama 3 8B 为例：32 层，8 个 KV 头，每头维度 4096 ÷ 32 = 128，用 FP16 存（2 字节）：

```text
2 × 32 × 8 × 128 × 2 = 131,072 字节 = 128 KiB / token
8K token 的一个请求 = 1 GiB
```

同样的结构用 MHA（32 个 KV 头），每 token 是 512 KiB，是 GQA 的 4 倍。这就是 GQA 和 MLA 要省的东西。

KV Cache 省的是重算历史 token 的投影和 MLP，注意力打分本身仍要和全部历史做点积，每步的计算量随长度线性增长。有缓存和没缓存时每步的 FLOPs、读写字节、算术强度、从 config 估算单卡并发、PagedAttention，都在 [inference/01 KV Cache 与显存估算](../../tracks/inference/01-kv-cache-memory/README.md)；KV 量化在 [inference/04 第 7 节](../../tracks/inference/04-quantization/README.md#7-kv-cache-量化)；批处理和 PD 分离在 [inference/02](../../tracks/inference/02-batching-pd/README.md)。

**对 Agent 的启示**：

- Agent 每一步都把工具结果追加进上下文，KV Cache 随步数增长。工具结果裁剪不只是省钱，也是省显存。
- 前缀相同的请求可以复用已算好的 KV Cache。系统提示和工具定义放在最前面并保持稳定，是为了命中前缀缓存，见 [03 章第 6 节](../03-prompt-context/README.md#6-上下文装配顺序与前缀缓存)。

## 8 预训练目标与 Scaling Law

### 8.1 目标：下一个 token 预测

解码器模型最小化交叉熵：

```text
loss = −(1/T) Σ_t log p_θ(x_t | x_<t)
```

因果 mask 让一次前向同时得到所有位置的预测，T 个位置的 loss 一起算。对话数据做 SFT 时，通常只在助手回复上算 loss，系统提示和用户输入被 mask 掉（Llama 3 技术报告 4.1.3 节：在目标 token 上用标准交叉熵，mask 掉 prompt token 的 loss）。细节见 [11 后训练入门第 3 节](../11-post-training-basics/README.md#3-sftchat-templateloss-mask-与-packing)。

### 8.2 Kaplan：loss 随规模呈幂律下降

[Kaplan 等，2001.08361](https://arxiv.org/abs/2001.08361)，2020-01。主要结论（第 1.1 节）：

- loss 和非嵌入参数 N、数据量 D、训练算力 C 各自呈幂律关系，前提是另外两个不成为瓶颈：L(N) = (N_c/N)^α_N，α_N ≈ 0.076；L(D) 的 α_D ≈ 0.095；L(C_min) 的 α_C ≈ 0.050。趋势跨越六个数量级以上。
- 在合理范围内，loss 对宽深比等其他结构超参数依赖很弱。
- 过拟合的程度取决于 N^0.74/D：模型变大 8 倍，数据只需多约 5 倍就能避免损失。
- 大模型更省样本。给定算力时，最优做法是训练很大的模型，并在远没收敛时停止。
- 最优分配：N ∝ C_min^0.73，数据只按 D ∝ C^0.27 缓慢增长。算力乘 10，最优模型变大约 5 倍。

### 8.3 Chinchilla：参数和数据同比例增长

[Hoffmann 等，2203.15556](https://arxiv.org/abs/2203.15556)，2022-03。论文训练了 400 多个 70M 到 16B 的模型，用三种方法估计 N_opt ∝ C^a 和 D_opt ∝ C^b：

| 方法 | a | b |
|---|---|---|
| 1. 每个模型取训练曲线的最低包络 | 0.50 | 0.50 |
| 2. 等算力曲线（IsoFLOP profiles） | 0.49 | 0.51 |
| 3. 拟合参数化 loss | 0.46 | 0.54 |
| Kaplan 等（2020） | 0.73 | 0.27 |

结论：给定算力，参数量和训练 token 数应同比例增长，模型大一倍，数据也要多一倍。当时的大模型（大多约 300B token）普遍训练不足。验证实验：70B 的 Chinchilla 用 1.4T token 训练，和 280B 的 Gopher 用相同算力（5.76 × 10²³ FLOPs）。

方法 3 拟合的 loss 形式是

```text
L(N, D) = E + A / N^α + B / D^β         E = 1.69，A = 406.4，B = 410.7，α = 0.34，β = 0.28
```

E 是数据本身的熵，不可约；第二项是模型容量有限带来的损失；第三项是数据有限、训练没收敛带来的损失。

**从拟合式推出最优分配**（论文式 4）。约束 C = 6ND，代入 D = C/(6N)：

```text
L(N) = E + A·N^(−α) + B·(C/6)^(−β)·N^β
dL/d(ln N) = −α·A·N^(−α) + β·B·(C/6)^(−β)·N^β = 0
→ N_opt = G · (C/6)^(β/(α+β))，  D_opt = G⁻¹ · (C/6)^(α/(α+β))，  G = (αA / (βB))^(1/(α+β))
```

所以 a = β/(α+β)，b = α/(α+β)。用正文里的 α = 0.34、β = 0.28 代入，得 a = 0.45，和 Table 2 的 0.46 差 0.01，Gopher 算力下的最优模型约 32B，而论文说约 40B。[Besiroglu 等，2404.10102](https://arxiv.org/abs/2404.10102) 的复现指出，论文 TeX 源码注释里未取整的值是 α = 0.3392、β = 0.2849，取整对数据项的影响很大。代回未取整的值，a = 0.46、N_opt ≈ 40B，都对上了（第 8.5 节代码）。同一篇复现还认为方法 3 的置信区间窄得不合理，并且和方法 1、2 不一致。这说明拟合常数本身不稳，结论应该记“同比例增长”，不要记具体常数。

### 8.4 两篇结论为什么不同

Chinchilla 论文自己给出的主要原因：Kaplan 对所有模型用固定的训练 token 数和学习率调度。用一个按 130B token 设计的余弦调度，中途取出的 loss 会高估“按更短调度训练”的模型的 loss，从而低估了小模型多训数据的效果。Chinchilla 发现学习率调度的长度应大致和训练 token 数匹配。另外 Kaplan 的模型大多小于 100M 参数，Chinchilla 的大多大于 500M。

[Pearce 和 Song，2406.12907](https://arxiv.org/abs/2406.12907) 给出了另一个解释：Kaplan 只数非嵌入参数，又是在小规模上分析。小模型里嵌入参数占比高，只数非嵌入参数会让系数偏大。他们在 Kaplan 的条件下模拟 Chinchilla 的研究，得到局部系数 0.78，接近 Kaplan 的 0.73；改用总参数，得到 0.51，接近 Chinchilla 的 0.50。

**“每参数 20 token”**来自 1.4T ÷ 70B，是 Chinchilla 这一个实验的比例，不是定律。按拟合式，最优 D/N 随算力变化。实际训练还要考虑推理成本：模型要服务大量请求时，用更多数据训练更小的模型更划算。Llama 3 技术报告说，405B 旗舰模型接近其训练预算（3.8 × 10²⁵ FLOPs）下的计算最优规模，而小模型训练得远超计算最优，在相同推理预算下效果更好。8B 模型按 20 倍只需约 160B token，实际预训练语料约 15T token。

Scaling Law 拟合的是预训练 loss。loss 和下游任务指标的关系不一定平滑，要单独评测。

### 8.5 实现：用 Chinchilla 的拟合式求最优分配

下面的代码用论文式 10 的常数，在 C = 6ND 约束下求闭式最优解，并用数值扫描核对；再用未取整的常数复现 a = 0.46 和 40B；最后验证同样算力下 70B / 1.4T 的拟合 loss 低于 280B / 300B。

```python
import numpy as np

# Chinchilla Approach 3 的拟合（论文式 10）：L(N, D) = E + A/N^α + B/D^β
E, A, B, alpha, beta = 1.69, 406.4, 410.7, 0.34, 0.28

def loss(N, D):
    return E + A / N ** alpha + B / D ** beta

# 1) 在约束 C = 6ND 下求最优：代入 D = C/(6N)，对 log N 求导令其为 0，得到论文式 4
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
a, b = beta / (alpha + beta), alpha / (alpha + beta)
def optimum(C):
    return G * (C / 6) ** a, (C / 6) ** b / G
assert round(a, 2) == 0.45 and round(b, 2) == 0.55   # 和 Table 2 的 0.46 / 0.54 差 0.01，原因见下

# 2) 数值核对：扫描 N，等算力线上 loss 的最小点和闭式一致
C = 5.76e23                                           # Gopher 的训练算力
Ns = np.logspace(9, 12, 20001)
best = Ns[np.argmin(loss(Ns, C / (6 * Ns)))]
N_opt, D_opt = optimum(C)
assert abs(best / N_opt - 1) < 0.01
# 用论文正文里四舍五入后的 α、β，算出约 32B；Besiroglu 等从论文 TeX 源码注释里找到
# 未取整的 α = 0.3392、β = 0.2849，代回去约 40B，和论文 Figure 4 说明的 40B 一致
assert 30e9 < N_opt < 34e9
alpha, beta = 0.3392, 0.2849
G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
a, b = beta / (alpha + beta), alpha / (alpha + beta)
N_opt, D_opt = optimum(C)
assert round(a, 2) == 0.46 and round(b, 2) == 0.54   # 和 Table 2 的 0.46 / 0.54 对上
assert round(N_opt / 1e9) == 40
print(f"Gopher 算力下 Approach 3 最优：N ≈ {N_opt/1e9:.0f}B, D ≈ {D_opt/1e12:.1f}T")

# 3) 同样的算力，280B 训 300B token（Gopher）不如 70B 训 1.4T token（Chinchilla）
C_gopher = 6 * 280e9 * 300e9
C_chin = 6 * 70e9 * 1.4e12
assert abs(C_gopher / C_chin - 1) < 0.15              # 6ND 口径下两者算力接近
assert loss(70e9, 1.4e12) < loss(280e9, 300e9)

# 4) 指数的含义：算力乘 10，最优 N 乘 10^a。Kaplan 的 0.73 下 N 涨 5.4 倍，Chinchilla 的 0.5 下涨 3.2 倍
assert round(10 ** 0.73, 1) == 5.4 and round(10 ** 0.5, 1) == 3.2
# “20 token / 参数”只是 1.4T / 70B，不是从拟合推出的常数：闭式下 D/N 随算力变化
r1 = optimum(1e22)[1] / optimum(1e22)[0]
r2 = optimum(1e25)[1] / optimum(1e25)[0]
assert 1.4e12 / 70e9 == 20 and r2 > r1
```

## 9 数值例子：Llama 3 8B 走一遍

用技术报告 Table 3 和 config.json 里的一组数字，把本章的公式全部代一遍。可以手算，也可以运行下面的代码核对。

| 量 | 值 | 来源 |
|---|---|---|
| 层数 L | 32 | Table 3 |
| d / d_ff | 4096 / 14336 | Table 3 |
| 查询头 / KV 头 | 32 / 8 | Table 3 |
| 词表 V | 128,256（Table 3 写作 128,000；config 里多出的 256 个是特殊 token 位） | config.json |
| RoPE 底数 | 500,000 | Table 3 |

手算步骤：

1. **参数**。每层注意力 2.5d²，FFN 3·4096·14336 = 10.5d²，两个 RMSNorm 2d，合计 13d² + 2d。32 层约 6.98B。嵌入加输出头 2 × 128,256 × 4096 ≈ 1.05B。总数 8,030,261,248。
2. **每 token 前向**。2 ×（6.98B + 0.53B 输出头）≈ 15.0 GFLOPs。位置 8192 上再加注意力 4 × 32 × 4096 × 8192 ≈ 4.3 GFLOPs，合计约 19.3 GFLOPs。
3. **Prefill 8K token**。线性层 8192 × 15.0 GFLOPs ≈ 123 TFLOPs；注意力按因果 mask 平均每个位置看一半，约 18 TFLOPs；合计约 141 TFLOPs。注意力约占 12%。
4. **KV Cache**。2 × 32 × 8 × 128 × 2 字节 = 128 KiB / token，8K 上下文 1 GiB。
5. **Tokenizer**。每 token 3.94 个英文字符，8K token 约 3.2 万个字符；用 Llama 2 的 tokenizer 约 2.6 万个。
6. **注意力缩放**。d_h = 128，不缩放时打分的标准差约 11.3。
7. **RoPE**。最高频一对波长 6.3 个 token，最低频一对约 250 万个 token。
8. **训练算力**。按“20 token / 参数”是 161B token，6ND ≈ 7.7 × 10²¹ FLOPs。实际预训练语料约 15T token，是这个数的近百倍，这正是第 8.4 节说的“为推理成本多训数据”。

```python
import math

# Llama 3 8B（技术报告 Table 3 与 config.json）：32 层，d = 4096，32 个查询头，8 个 KV 头，
# FFN 宽 14336，词表 128,256，RoPE 底数 500,000，输入嵌入和输出头不共享
L, d, n_h, n_kv, d_ff, V, base = 32, 4096, 32, 8, 14336, 128_256, 500_000.0
d_h = d // n_h

# (a) 参数：每层 13d²，非嵌入 6.98B；嵌入 + 输出头 1.05B；总数 8,030,261,248
per_layer = d * d + 2 * d * n_kv * d_h + d * d + 3 * d * d_ff + 2 * d
body = L * per_layer + d
emb = 2 * V * d
assert body + emb == 8_030_261_248
assert per_layer - 2 * d == 13 * d * d

# (b) 一个 token 的前向 FLOPs：2 ×（非嵌入 + 输出头）+ 注意力 4·L·d·T
f_linear = 2 * (body + V * d)
f_attn = lambda T: 4 * L * n_h * d_h * T
assert round(f_linear / 1e9, 1) == 15.0
assert round((f_linear + f_attn(8192)) / 1e9, 1) == 19.3

# (c) Prefill 8K token：线性层 8192 × 15 GFLOPs；因果注意力平均每个位置看 T/2 个键
T = 8192
prefill = T * f_linear + sum(f_attn(t + 1) for t in range(T))
assert round(prefill / 1e12) == 141                     # 约 141 TFLOPs
assert 0.11 < sum(f_attn(t + 1) for t in range(T)) / prefill < 0.13
# 摊到每个 token：注意力平均 ≈ 2·L·d·T，正是 Kaplan 式 2.2 里的上下文项 2·n_layer·n_ctx·d_attn
attn_avg = sum(f_attn(t + 1) for t in range(T)) / T
assert abs(attn_avg / (2 * L * d * T) - 1) < 1e-3

# (d) KV Cache：2 × 32 × 8 × 128 × 2 字节 = 128 KiB / token，8K 上下文 1 GiB
kv = 2 * L * n_kv * d_h * 2
assert kv == 128 * 1024 and kv * 8192 == 2 ** 30

# (e) 8K token 大约是多少英文：Llama 3 tokenizer 每 token 3.94 字符，Llama 2 是 3.17
assert round(8192 * 3.94) == 32_276 and round(8192 * 3.17) == 25_969
assert round(3.94 / 3.17 - 1, 3) == 0.243               # 同样的上下文多装约 24% 文本

# (f) 注意力缩放：d_h = 128，未缩放的打分标准差约 √128 ≈ 11.3（分量方差 1 的假设下）
assert round(math.sqrt(d_h), 1) == 11.3

# (g) RoPE：最高频一对每个位置转 1 弧度（波长 6.3 个 token），最低频一对波长约 250 万个 token
theta = [base ** (-2 * i / d_h) for i in range(d_h // 2)]
lam = [2 * math.pi / t for t in theta]
assert round(lam[0], 1) == 6.3 and 2.4e6 < lam[-1] < 2.6e6

# (h) 训练：按“每参数 20 token”，8B 的计算最优数据量是 160B token，6ND ≈ 7.7e21 FLOPs。
#     Llama 3 的预训练语料约 15T token，技术报告说小模型训练得远超计算最优
N = (body + emb)
assert round(20 * N / 1e9) == 161
assert round(6 * N * 20 * N / 1e21, 1) == 7.7
print(f"前向 {f_linear/1e9:.1f} GFLOPs/token，Prefill 8K {prefill/1e12:.0f} TFLOPs，KV {kv//1024} KiB/token")
```

## 10 工程细节与常见坑

每条写现象、原因、排查和处理。

**1. 按字数估 token，或者用别的模型的 tokenizer 估**

- 现象：上下文预算算好了，线上仍然报超长；同一段中文换了模型，费用差了一截。
- 原因：token 数取决于 tokenizer 的训练语料和词表大小。字节级 BPE 下，没学到合并的汉字一个要 3 个 token；学到了则一个词一个 token。
- 排查：用目标模型自己的 tokenizer 对真实样本统计“字符 / token”比，按语种、按内容类型（代码、JSON、自然语言）分别统计。
- 处理：预算按实测比例并留余量；截断按 token 截，截完检查是否切在多字节字符中间。

**2. 特殊 token 和 chat template 不匹配**

- 现象：模型不停输出、不调用工具，或者把工具结果当成用户的话来回应。
- 原因：角色分隔符、结束符、工具调用标记是训练时的约定。手工拼 prompt 时漏了结束符、用错了模板版本，或者把特殊 token 的字面字符串当普通文本分词，模型看到的就是训练时没见过的序列。
- 排查：打印最终送进模型的 token id，和官方模板（如 `apply_chat_template` 的输出）逐个比对；确认结束符 id 在推理框架的停止条件里。
- 处理：用模型自带的模板生成 prompt，不手工拼；用户输入里的特殊 token 字面量要转义。

**3. 训练不足的 token（glitch token）**

- 现象：某些罕见字符串让模型输出乱码、复读或者完全答非所问。
- 原因：tokenizer 和模型分开训练，词表里有些 token 在模型训练数据里几乎没出现，嵌入没被训练好（第 2.5 节）。
- 排查：Land 和 Bartolo 的思路是看嵌入的统计特征，例如输出头里某些行和“未使用 token”的行很接近。业务里更直接的做法是对出问题的输入逐 token 二分定位。
- 处理：输入侧做规范化或过滤；训练自己的模型时，tokenizer 的语料要和模型的训练数据同分布。

**4. 注意力里用低精度做 softmax**

- 现象：长上下文时输出质量下降，或者出现 NaN；FP16 下比 BF16 更容易出问题。
- 原因：打分很大时 exp 溢出；softmax 的求和在半精度下累积误差大。所以实现都先减最大值，并在 float32 下算 softmax。Hugging Face 的 Llama eager 实现里就是 `softmax(..., dtype=torch.float32)` 再转回原精度。
- 排查：检查自己写的注意力是否减了最大值、softmax 精度是多少；对比 float32 参考实现的输出。
- 处理：softmax 和 RMSNorm 的统计量用 float32；用经过验证的注意力 kernel。

**5. 因果 mask 和填充 mask 写错**

- 现象：训练 loss 低得反常（偷看了答案）；或者同一条 prompt 单独跑和放进 batch 跑结果不同。
- 原因：mask 的方向写反（`triu` 和 `tril`、k = 0 和 k = 1）；左填充时填充位置没被 mask 掉，或者位置下标没有按每条序列的真实起点算；packing 多个文档进一条序列时没有加文档间 mask。Llama 3 技术报告提到，他们用了阻止同一序列内不同文档互相注意的 mask，在超长序列的继续预训练里很重要。
- 排查：第 3.6 节的测试：改最后一个 token，前面位置的输出必须不变；单条和 batch 的 logits 逐元素比。
- 处理：mask 和位置下标写成单元测试，每次改注意力代码都跑。

**6. RoPE 的配对方式和权重排列不一致**

- 现象：从原始权重转换来的模型能跑，但输出是乱的；或者自己实现的 RoPE 和框架的结果对不上。
- 原因：原论文是相邻两维配对，Hugging Face Llama 是前后两半配对（`rotate_half`）。两种写法需要不同的 W_Q、W_K 列排列，转换脚本里做了 `permute`。混用就等于把位置信息加错了维度。
- 排查：拿一个小 q、k，用两种实现分别算 ⟨R(m)q, R(n)k⟩，确认在同一套权重排列下一致。
- 处理：只用一种约定，并在权重转换处写明。

**7. 长度扩展配置和训练时不一致**

- 现象：模型标称 128K，超过某个长度后困惑度陡增或开始胡言乱语；或者短文本上反而变差。
- 原因：`rope_scaling` 的类型、倍数、`original_max_position_embeddings` 和模型实际训练时的设置不一致；或者推理框架不支持某种 scaling，静默忽略了。YaRN 的温度、NTK 的底数都要和微调时相同。
- 排查：读 config 里的 `rope_theta` 和 `rope_scaling`，和模型卡对照；在不同长度上测困惑度和大海捞针，看拐点在哪。
- 处理：按模型卡设置；只在需要长上下文时开启扩展，部分框架的静态缩放会影响短文本。

**8. 参数量或 FLOPs 估算漏项**

- 现象：估出来的参数和模型卡差几个百分点到十几个百分点；MFU（实际算力利用率）算出来大于 100% 或低得离谱。
- 原因：用了 12d² 但模型是 GQA + SwiGLU（Llama 3 8B 差 8%）；忘了嵌入和输出头（占 13%）；忘了 QKV 偏置；输入输出嵌入是否共享看错；长上下文训练时忽略了注意力项（8K 时约占前向的 22%）。
- 排查：照第 5.7 节，从 config 数出参数和官方数字对到个位。
- 处理：参数按 config 精确数；FLOPs 用 6N 加注意力项，并说明 N 是否含嵌入。

**9. 采样参数的边界**

- 现象：温度设成 0.001 结果和贪心不一样；top-p 设成 0 报错或只输出一个 token；不同框架同样的参数输出风格不同。
- 原因：框架对极小温度做了截断（vLLM 把 0 到 0.01 之间的温度抬到 0.01），对温度 0 走贪心分支；温度、top-k、top-p、重复惩罚的应用顺序在不同框架里可能不同；top-p 的“累计到 p”是含还是不含边界那个 token，实现可能不同。
- 排查：读所用框架的采样代码，确认处理器的顺序和边界；对同一组 logits 离线算出过滤后的分布。
- 处理：需要确定输出时直接用贪心模式，不靠极小温度；跨框架迁移时重新调参。

**10. 把温度 0 当作可复现**

- 现象：同一请求温度 0 跑几次，结果在中途开始分叉；离线复现不了线上的失败。
- 原因：服务端批大小随负载变化，归约顺序变了，logit 的最后几位变了，接近并列的 token 被翻转（第 6.5 节）。
- 排查：同一请求在空闲时和高负载时各跑多次，比较第一次分叉的位置。
- 处理：Agent 记录完整轨迹并回放工具结果；确实需要逐位确定时，用批不变的内核，接受性能代价。

## 11 对比与取舍

### 11.1 分词方案

| 方案 | 初始符号 | 训练 | 编码 | 适合 |
|---|---|---|---|---|
| BPE（字符级起点） | 字符 + 词尾符号 | 反复合并最高频相邻对 | 按合并顺序（rank）合并 | 有空格分词的语言 |
| 字节级 BPE | 256 个字节 | 同上，通常先用正则预切分 | 同上 | 通用，任何输入都能编码 |
| SentencePiece | 原始句子里的字符，空格也是字符 | BPE 或 Unigram | 同训练算法 | 中文、日文等不依赖空格的语言 |

### 11.2 注意力结构

| 结构 | 每层注意力参数 | 每 token 每层 KV 元素 | 什么时候选 |
|---|---|---|---|
| MHA | 4d² | 2d | 老模型；新模型基本不选 |
| GQA | (2 + 2n_kv/n_h)·d² | 2·n_kv·d_h | 默认选择，生态和 kernel 支持最好 |
| MQA | ≈ 2d² | 2·d_h | 极端看重 Decode 速度，能接受质量风险 |
| MLA | 见 [inference/01 第 4.3 节](../../tracks/inference/01-kv-cache-memory/README.md#43-mla缓存潜向量用吸收避免展开) | 潜向量 + 解耦 RoPE 键 | 长上下文、大并发，引擎要有 MLA kernel |

### 11.3 位置编码与长度扩展

| 方法 | 改什么 | 高频维度 | 低频维度 | 需要微调 |
|---|---|---|---|---|
| 直接外推 | 不改 | 不变 | 出现训练外的角度 | — |
| PI | 位置 m → m/s | 压小 s 倍，近邻难分辨 | 压小 s 倍 | 需要，论文 1000 步 |
| NTK-aware | 底数 b → b·s^(d_h/(d_h−2)) | 基本不变 | 最低频压小 s 倍 | 不微调时比 PI 好；微调后不如 PI（YaRN 附录 A.2） |
| YaRN | 按 r = L/λ 分段插值 + 注意力温度 | 不变（r > β） | 按 PI 插值（r < α） | 少量，论文 400 步 |
| ALiBi | 打分加 −m·距离 | 无旋转 | 无旋转 | 训练时就要用 |
| 预训练加大底数 | base 调大（Llama 3 用 500,000） | 不变 | 波长更长 | 在预训练里完成 |

### 11.4 归一化与 FFN

| 选项 | 优点 | 代价 | 现状 |
|---|---|---|---|
| Post-LN | 每层输出尺度固定 | 初始化时输出层附近梯度大，需要 warm-up | 原始 Transformer、BERT |
| Pre-LN | 恒等路径直通，训练稳定 | 残差流范数随深度增长，要加最终 LN | 主流 |
| LayerNorm | 平移和缩放都不变 | 多一次归约，多一组偏置 | 较早的模型 |
| RMSNorm | 少一次归约，论文报告提速 7% 到 64% | 不对平移不变 | Llama、Qwen 等 |
| ReLU/GELU FFN（d_ff = 4d） | 简单 | — | 较早的模型 |
| SwiGLU（d_ff ≈ 8d/3） | 同参数量下困惑度更好（Shazeer 2020） | 三个矩阵，宽度要凑整 | 主流 |

### 11.5 解码参数怎么选

| 场景 | 设置 | 理由 |
|---|---|---|
| 工具参数、抽取、分类 | 贪心，配合结构化输出约束 | 要的是最可能的那个答案，格式错误少 |
| 开放生成、写作 | 温度 0.7 左右加 top-p 0.9 左右，按任务调 | 截掉长尾，保留多样性 |
| 多次采样投票、生成多个方案 | 温度较高 | 需要候选之间有差异 |
| 需要逐位复现 | 贪心 + 批不变内核，或者直接回放记录 | 温度 0 本身不保证复现 |

## 12 自测清单

- [ ] 手算论文玩具语料的前三次 BPE 合并，说明编码为什么要按 rank，举出最长匹配切错的例子
- [ ] 说出词表变大对序列长度、嵌入参数、低频 token 各有什么影响，以及 glitch token 的来历
- [ ] 从 q、k 分量独立同分布推出点积方差是 d_k，用 softmax 的雅可比说明不缩放时梯度为什么小
- [ ] 写出带因果 mask 的多头注意力，说出 MHA、GQA、MQA 的每层参数和每 token KV 元素
- [ ] 用复数推导 RoPE 的内积只依赖 m − n，用波长解释它为什么不能直接外推
- [ ] 说出 PI、NTK-aware、YaRN 分别改了哪些频率，写出 NTK 的新底数和 YaRN 的温度公式
- [ ] 解释 Pre-LN 为什么比 Post-LN 好训，RMSNorm 去掉了什么，SwiGLU 的宽度为什么是 2/3
- [ ] 推出每层 12d² 的前提，说出 GQA 和 SwiGLU 怎么改它；从 config 数出 Llama 3 8B 的参数
- [ ] 推出前向 2N、训练 6N，说清注意力项什么时候不能忽略
- [ ] 说出 Kaplan 和 Chinchilla 的指数和结论差别，两个解释原因，“20 token / 参数”的来历
- [ ] 写出温度、top-k、top-p 的实现，说出框架里的应用顺序和边界
- [ ] 解释温度 0 不可复现的三层原因，以及 Agent 里该怎么复现
- [ ] 给定层数、KV 头数、头维度，当场算出 KV Cache 大小

## 资料

论文：

- Sennrich 等，[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)，arXiv 1508.07909，2015-08（3.2 节 BPE 算法与 Algorithm 1）
- Kudo、Richardson，[SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)，arXiv 1808.06226，2018-08
- Land、Bartolo，[Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models](https://arxiv.org/abs/2405.05417)，arXiv 2405.05417，2024-05
- Vaswani 等，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，arXiv 1706.03762，2017-06（3.2.1 节脚注 4 的方差推导，3.2.2 节多头）
- Shazeer，[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)，arXiv 1911.02150，2019-11
- Ainslie 等，[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)，arXiv 2305.13245，2023-05
- DeepSeek-AI，[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)，arXiv 2405.04434，2024-05
- Su 等，[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)，arXiv 2104.09864，2021-04（3.2 节旋转推导，3.3 节长程衰减）
- Press 等，[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409)，arXiv 2108.12409，2021-08（第 3 节斜率）
- Chen 等，[Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)，arXiv 2306.15595，2023-06
- Peng 等，[YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)，arXiv 2309.00071，2023-08（3.2、3.3 节，附录 A.2 NTK-aware 底数推导，第 4 节训练设置）
- Xiong 等，[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)，arXiv 2002.04745，2020-02
- Zhang、Sennrich，[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)，arXiv 1910.07467，2019-10
- Shazeer，[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)，arXiv 2002.05202，2020-02
- Kaplan 等，[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)，arXiv 2001.08361，2020-01（1.1、1.2 节结论，2.1 节式 2.1、2.2 与 Table 1）
- Hoffmann 等，[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，arXiv 2203.15556，2022-03（Table 2、式 4、式 10，附录 F 的 FLOPs 计数）
- Besiroglu 等，[Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102)，arXiv 2404.10102，2024-04
- Pearce、Song，[Reconciling Kaplan and Chinchilla Scaling Laws](https://arxiv.org/abs/2406.12907)，arXiv 2406.12907，2024-06
- Fan 等，[Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)，arXiv 1805.04833，2018-05（top-k 采样）
- Holtzman 等，[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)，arXiv 1904.09751，2019-04
- Grattafiori 等（Meta Llama Team），[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)，arXiv 2407.21783，2024-07（3.2 节 Table 3 与词表，4.1.3 节 SFT loss mask）

博客、配置与源码：

- Thinking Machines Lab（Horace He 等），[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，2025-09-10
- [unsloth/llama-3-8b config.json](https://huggingface.co/unsloth/llama-3-8b)（与 meta-llama/Meta-Llama-3-8B 同结构的公开镜像；官方仓库需登录，两者 Hugging Face API 的参数总数都是 8,030,261,248），2026-11 查阅
- Qwen 团队，[Qwen2.5-7B 模型卡与 config.json](https://huggingface.co/Qwen/Qwen2.5-7B)，2026-11 查阅
- Meta，[llama-models 的 `sku_list.py`](https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py)（`ffn_dim_multiplier`、`multiple_of`）和 [llama3 的 `model.py`](https://github.com/meta-llama/llama3/blob/main/llama/model.py)（FFN 宽度计算），main 分支，2026-11 查阅
- OpenAI，[gpt-2 的 `encoder.py`](https://github.com/openai/gpt-2/blob/master/src/encoder.py)（按 `bpe_ranks` 编码），2026-11 查阅
- Hugging Face transformers 源码，[`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)（`LlamaRMSNorm`、`rotate_half`、float32 softmax）、[`convert_llama_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)（`permute`）、[`generation/utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py)（采样处理器顺序），main 分支，2026-11 查阅
- vLLM 源码，[`sampling_params.py`](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)（温度 0 与最小温度），main 分支，2026-11 查阅
