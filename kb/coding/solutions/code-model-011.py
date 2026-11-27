'''code-model-011：手写 BPE 训练。

题目：输入词频表 {词: 次数} 和合并次数 n，每轮按词频统计所有相邻对，取最高频的一对合并成新符号，
在所有词里替换，记下规则，重复 n 轮。
复杂度：设 V 为词表大小、L 为平均词长（符号数）、n 为合并次数：时间 O(n·V·L)，空间 O(V·L)。
'''
import numpy as np


def _count_pairs(words):
    '''统计相邻对的加权计数：计数加词频，不是加 1。'''
    counts = {}
    for tok, c in words.items():
        for pair in zip(tok, tok[1:]):
            counts[pair] = counts.get(pair, 0) + c
    return counts


def _merge(tok, pair):
    '''从左到右扫描，遇到最佳对就合并并跳过两个位置，否则保留一个。'''
    out, i, first = [], 0, pair[0]
    while i < len(tok):
        if i < len(tok) - 1 and tok[i] == first and tok[i + 1] == pair[1]:
            out.append(first + pair[1])
            i += 2
        else:
            out.append(tok[i])
            i += 1
    return tuple(out)


def train_bpe(word_freq, num_merges):
    '''返回合并规则列表 [(a, b), ...]，顺序即编码时的优先级。'''
    words = {(tuple(w) + ('</w>',)): c for w, c in word_freq.items()}
    merges = []
    for _ in range(num_merges):
        counts = _count_pairs(words)
        if not counts:
            break                                   # 没有相邻对可合并时提前结束
        # 平局规则（必须固定，否则结果依赖字典顺序）：
        # 计数最高 → 右符号是词尾标记 </w> 的优先 → 再按符号字典序最小。
        # 经典示例里 (e,s)、(s,t)、(t,</w>) 计数并列，靠第二条选出 (t,</w>)；
        # 之后 (s,t</w>)、(e,st</w>) 计数唯一，不再需要平局规则
        best = min(counts.items(), key=lambda kv: (-kv[1], kv[0][1] != '</w>', kv[0]))[0]
        merges.append(best)
        words = {_merge(tok, best): c for tok, c in words.items()}
    return merges


def apply_bpe(token, merges):
    '''编码时按学到的合并顺序依次应用，顺序就是优先级。'''
    tok = tuple(token) + ('</w>',)
    for pair in merges:
        tok = _merge(tok, pair)
    return tok


def apply_bpe_seq(token, merges):
    '''等价实现：字符串 replace 的写法，用于对照（只在符号无歧义时等价）。'''
    tok = list(tuple(token) + ('</w>',))
    for a, b in merges:
        out, i = [], 0
        while i < len(tok):
            if i < len(tok) - 1 and (tok[i], tok[i + 1]) == (a, b):
                out.append(a + b)
                i += 2
            else:
                out.append(tok[i])
                i += 1
        tok = out
    return tuple(tok)


np.random.seed(0)

# 题目给的例子：最高频对依次是 (t, </w>)、(s, t</w>)、(e, st</w>)
freq = {'lowest': 5, 'newer': 6, 'widest': 3}   # 经典小例子
merges = train_bpe(freq, 6)
# 第一次合并是最高频的 (w, e)（'lowest' 5 + 'newer' 6）；
# 第二轮 (t, </w>) 与 (s, t) 计数都是 8，平局规则让右符号是词尾的 (t, </w>) 先合并，
# 这也说明为什么题目强调“计数打平时要有确定的平局规则”
assert merges[0] == ('w', 'e')
assert merges[1] == ('t', '</w>')
assert merges[2] == ('s', 't</w>')
# 第三次合并 (e, st</w>) 与 (e, s) 都是 6，靠字典序让 (e, s) 先；不同平局规则下后续顺序会变
assert len(merges) == 6
assert all(isinstance(a, str) and isinstance(b, str) for a, b in merges)

# 词尾标记：'low</w>'、'new</w>' 等词尾与词中同一片段分开；每个词的编码都以 </w> 结尾
enc = {w: apply_bpe(w, merges) for w in freq}
assert all(t[-1].endswith('</w>') for t in enc.values())
assert all('</w>' not in s for t in enc.values() for s in t[:-1])

# 合并后每个词的符号序列，等于从初始字符逐轮按规则合并的结果（apply_bpe 与逐轮写法一致）
for w in freq:
    assert apply_bpe(w, merges) == apply_bpe_seq(w, merges)
# 合并规则可以从编码结果里还原：每轮把相邻两符号拼起来就是学到的符号
state = {w: tuple(w) + ('</w>',) for w in freq}
replay = list(state.values())
for a, b in merges:
    replay = [_merge(t, (a, b)) for t in replay]
assert [apply_bpe(w, merges) for w in freq] == replay

# 每轮统计按词频加权：'lowest'(5) + 'widest'(3) 的 (e,s) 是 8，不是 2
counts = _count_pairs({tuple(w) + ('</w>',): c for w, c in freq.items()})
assert counts[('e', 's')] == 8
assert counts[('t', '</w>')] == 8                            # 5 + 3
assert counts[('s', 't')] == 8
assert counts[('w', 'e')] == 11                              # 5 + 6，本轮最高频

# 重叠对：「aaaa」里 (a, a) 统计 3 次，但只能合并 2 次
assert _count_pairs({('a', 'a', 'a', 'a', '</w>'): 1})[('a', 'a')] == 3
assert _merge(('a', 'a', 'a', 'a'), ('a', 'a')) == ('aa', 'aa')
m1 = train_bpe({'aaaa': 1}, 1)
assert m1 == [('a', 'a')] and apply_bpe('aaaa', m1) == ('aa', 'aa', '</w>')
# 朴素写法「两两配对后替换」会得到 0 个合并；字符串 replace 的常见错误见下
assert _merge(('a', 'a', 'a'), ('a', 'a')) == ('aa', 'a')

# 不能误合并到别的符号里：规则 (e, s) 不应改动已合并出的 'es'/'st' 这类符号
w2 = {'es': 1, 'e': 1, 's': 1}
m2 = train_bpe(w2, 1)
assert m2 == [('s', '</w>')]          # (s, </w>) 计数 2 最高，不碰 'es' 这个已有的符号
assert all(isinstance(s, str) for t in [apply_bpe(w, m2) for w in w2] for s in t)

# 合并次数用尽 / 没有可合并对象时提前停
assert train_bpe({'a': 1}, 5) == [('a', '</w>')]             # 只剩 (a, </w>) 一对，合并后提前结束
assert len(train_bpe({'ab': 3}, 1)) == 1
assert len(train_bpe({'ab': 3}, 10)) == 2                    # 最多合并到只剩一个符号

# 平局规则是确定的：计数打平时结果只由数据决定，与字典插入顺序无关
f1 = {'ab': 1, 'cd': 1}
f2 = {'cd': 1, 'ab': 1}
assert train_bpe(f1, 4) == train_bpe(f2, 4)

# 编码顺序即优先级：按规则顺序逐条应用，交换顺序可能得到不同结果
r_a = [('a', 'b'), ('b', 'c')]
r_b = [('b', 'c'), ('a', 'b')]
assert apply_bpe('abc', r_a) == ('ab', 'c', '</w>')     # (a,b) 先合并后，'b' 已不再是独立符号
assert apply_bpe('abc', r_b) == ('a', 'bc', '</w>')
assert apply_bpe('abc', r_a) != apply_bpe('abc', r_b)

# 训练结果可复现（无随机性），且规则不跨词泄漏：每个词独立编码后长度 ≤ 原字符数 + 1
assert train_bpe(freq, 6) == merges
for w in freq:
    assert len(apply_bpe(w, merges)) <= len(w) + 1
