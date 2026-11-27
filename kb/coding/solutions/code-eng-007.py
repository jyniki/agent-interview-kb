"""code-eng-007 简单的语义缓存：lookup(query) 和 store(query, answer)。

每条缓存存 (嵌入, 回答, 租户, 权限范围, 过期时间, 标签)；查询时只在同租户、同权限范围、
未过期的条目里找余弦相似度最高的一条，超过阈值才返回；含工具调用或有副作用的回答不写入；
容量满时按 LRU 淘汰。另带实体护栏：抽出来的编号 / 引号内文本不同就不命中。

复杂度：
- lookup：O(N · 词表大小)，N 为条目数；条目多时应换向量库做近似最近邻；
- store / delete / invalidate_tag：O(N)（线性找过期项与索引失效），淘汰 O(1)；
- 空间 O(N · 向量维度)。改进方案：HNSW 检索 O(log N)，标签索引 O(命中数)。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_THRESHOLD = 0.85

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-鿿]+")
_ENTITY_RE = re.compile(r"\b[A-Za-z]*-?\d{2,}\b")  # 订单号、单号之类带数字的实体


class Embedder:
    """默认嵌入：小写词 + 中文二元字组，单值词频（q 只需要方向）。

    这是“离线算好、只读”的词袋模型：同一文本永远得到同一个向量。
    想接真实向量模型，实现同样的 embed(text) -> 归一化向量即可。
    """

    @staticmethod
    def tokenize(text: str) -> list[str]:
        # 去掉所有空白：中文有没有空格不应改变语义，否则二元组会被拆断
        tokens: list[str] = []
        for piece in _TOKEN_RE.findall(re.sub(r"\s+", "", text.lower())):
            if piece.isascii():
                tokens.append(piece)
            else:
                tokens.append(piece)
                # 中文单字的信息量太低，补二元组
                for i in range(len(piece) - 1):
                    tokens.append(piece[i : i + 2])
        return tokens

    def embed(self, text: str) -> dict[str, float]:
        counts: dict[str, int] = {}
        for token in self.tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        # 对数饱和：重复出现不会让某一维无限大，余弦仍在 [0, 1]
        vector = {t: 1.0 + math.log(c) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {t: v / norm for t, v in vector.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def _canonical(text: str) -> str:
    """缓存键的规范化形式：去首尾空白并折叠内部空白。"""
    return " ".join(text.split())


def entities(text: str) -> frozenset:
    """抽取“相似但不同”的实体：带数字的编号 + 引号 / 书名号里的内容。

    “取消订单 A”和“取消订单 B”向量几乎一样，靠这里区分。
    """
    found = set(m.lower() for m in _ENTITY_RE.findall(text))
    for pattern in (r'"([^"]+)"', r"'([^']+)'", r"「([^」]+)」", r"《([^》]+)》"):
        found.update(m.strip().lower() for m in re.findall(pattern, text))
    return frozenset(found)


@dataclass
class Answer:
    text: str
    tool_calls: list = field(default_factory=list)   # 有工具调用 → 不缓存
    side_effecting: bool = False                     # 有副作用 → 不缓存
    validated: bool = True                           # 未通过校验 → 不缓存
    tags: tuple = ()


@dataclass
class Entry:
    key: str
    vector: dict
    answer: Answer
    tenant: str
    scope: frozenset
    expire_at: float
    ents: frozenset
    last_used: float


@dataclass
class LookupResult:
    hit: bool
    answer: Optional[Answer] = None
    score: float = 0.0
    reason: str = ""  # "hit" | "miss" | "below_threshold" | "entity_mismatch" | "expired"


class SemanticCache:
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        *,
        capacity: int = 1024,
        ttl: float = 300.0,
        clock=time.monotonic,
        embedder: Optional[Embedder] = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold 应在 [0, 1]")
        if capacity < 0:
            raise ValueError("capacity 不能为负")
        if ttl < 0:
            raise ValueError("ttl 不能为负")
        self.threshold = threshold
        self.capacity = capacity
        self.ttl = ttl
        self._clock = clock
        self._embedder = embedder or Embedder()
        self._lock = threading.RLock()
        self._entries: dict[str, Entry] = {}
        self.bypass_reasons: dict[str, int] = {}  # 统计：为什么没缓存

    # ------------------------------------------------------------ 写入

    def cacheable(self, answer: Answer) -> tuple[bool, str]:
        if answer.tool_calls:
            return False, "has_tool_calls"
        if answer.side_effecting:
            return False, "side_effecting"
        if not answer.validated:
            return False, "unvalidated"
        if not answer.text.strip():
            return False, "empty_answer"
        return True, ""

    def store(self, query: str, answer: Answer, *, tenant: str = "default", scope: frozenset = frozenset()) -> bool:
        """写入缓存。不可缓存的回答直接拒绝，返回 False。"""
        ok, reason = self.cacheable(answer)
        if not ok:
            self.bypass_reasons[reason] = self.bypass_reasons.get(reason, 0) + 1
            return False
        with self._lock:
            vector = self._embedder.embed(query)
            key = _canonical(query)
            now = self._clock()
            self._entries[key] = Entry(
                key=key,
                vector=vector,
                answer=answer,
                tenant=tenant,
                scope=frozenset(scope),
                expire_at=now + self.ttl,
                ents=entities(query),
                last_used=now,
            )
            self._evict(now)
            return True

    # ------------------------------------------------------------ 查询

    def lookup(self, query: str, *, tenant: str = "default", scope: frozenset = frozenset()) -> LookupResult:
        """同租户、同权限范围、未过期、相似度 ≥ 阈值且实体一致才算命中。"""
        with self._lock:
            self._purge_expired()
            vector = self._embedder.embed(query)
            q_ents = entities(query)
            best: Optional[Entry] = None
            best_score = -1.0
            blocked = ""  # 最高分那条为什么不能给
            for entry in self._entries.values():
                # 租户 / 权限先过滤：一个用户的回答绝不能命中另一个用户
                if entry.tenant != tenant:
                    continue
                if not entry.scope.issubset(frozenset(scope)):
                    continue
                score = cosine(vector, entry.vector)
                if score < best_score:
                    continue
                if score < self.threshold:
                    if best is None or score > best_score:
                        blocked = "below_threshold"
                        best_score = score
                    continue
                if q_ents != entry.ents and (q_ents or entry.ents):
                    # 相似但实体不同：取消订单 A / B，绝不能直接复用
                    if best is None or score > best_score:
                        blocked = "entity_mismatch"
                        best_score = score
                    continue
                best = entry
                best_score = score
                blocked = ""

            if best is None:
                return LookupResult(False, None, max(best_score, 0.0), blocked or "miss")
            # 顺序无关：命中后刷新 LRU 时间，但不重置 TTL（过期时间以写入为准）
            best.last_used = self._clock()
            return LookupResult(True, best.answer, best_score, "hit")

    # ------------------------------------------------------------ 维护

    def _purge_expired(self) -> int:
        now = self._clock()
        dead = [k for k, e in self._entries.items() if now >= e.expire_at]
        for key in dead:
            del self._entries[key]
        return len(dead)

    def _evict(self, now: float) -> None:
        while len(self._entries) > self.capacity:
            victim = min(self._entries.values(), key=lambda e: (e.last_used, e.key))
            del self._entries[victim.key]

    def delete(self, query: str, *, tenant: str = "default") -> bool:
        """按条删除：缓存了错误回答时用。"""
        with self._lock:
            key = _canonical(query)
            entry = self._entries.get(key)
            if entry is None or entry.tenant != tenant:
                return False
            del self._entries[key]
            return True

    def invalidate_tag(self, tag: str) -> int:
        """源数据变化时按标签失效。"""
        with self._lock:
            dead = [k for k, e in self._entries.items() if tag in e.answer.tags]
            for key in dead:
                del self._entries[key]
            return len(dead)

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._entries)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------------------------------------------------------------- 自测
# 只在直接运行时执行；被 import 时不跑断言、不产生任何副作用。
# 直接 `python3 <本文件>` 即可运行；CI 的 run_snippets 也是整份执行。
if __name__ == "__main__":
    # ---------------------------------------------------------------- 断言

    def ans(text, **kw) -> Answer:
        return Answer(text, **kw)


    # 1) 完全相同或改写后的问法命中；返回原回答。
    clock = FakeClock()
    cache = SemanticCache(threshold=0.7, capacity=10, ttl=100, clock=clock)
    assert cache.store("北京今天天气怎么样", ans("晴，21 度"))
    hit = cache.lookup("北京今天天气怎么样")
    assert hit.hit and hit.answer.text == "晴，21 度" and hit.score >= 0.999
    hit = cache.lookup("北京 今天 天气 怎么样")  # 标点 / 空白差异
    assert hit.hit, hit.reason

    # 2) 语义相近但不同的问题在词袋下相似度不够，不命中（宁可漏，不可错）。
    miss = cache.lookup("上海明天股票行情如何")
    assert not miss.hit and miss.reason == "below_threshold"
    assert miss.answer is None, "近乎相似但未过阈值也要返回 None"

    # 3) 只看相似度不够：必须和阈值比较（易错点）。
    cache = SemanticCache(threshold=0.99, capacity=10, ttl=100, clock=clock)
    cache.store("北京今天天气怎么样", ans("晴"))
    assert not cache.lookup("北京今天天气如何").hit

    # 4) 租户隔离：另一个租户查同一句话不命中（易错点）。
    cache = SemanticCache(threshold=0.0, capacity=10, ttl=100, clock=clock)
    cache.store("我的账单是多少", ans("100 元"), tenant="alice")
    assert cache.lookup("我的账单是多少", tenant="bob").hit is False
    assert cache.lookup("我的账单是多少", tenant="alice").hit is True

    # 5) 权限范围：缓存写入时 scope 更宽时，只有 scope 覆盖它的调用方才能命中。
    cache = SemanticCache(threshold=0.0, capacity=10, ttl=100, clock=clock)
    cache.store("部门预算", ans("1000 万"), tenant="acme", scope=frozenset({"finance"}))
    assert cache.lookup("部门预算", tenant="acme", scope=frozenset({"public"})).hit is False
    assert cache.lookup("部门预算", tenant="acme", scope=frozenset({"finance"})).hit is True
    assert cache.lookup("部门预算", tenant="acme", scope=frozenset({"finance", "public"})).hit is True

    # 6) 过期：TTL 到了就不命中，返回值不带答案。
    clock = FakeClock()
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=5, clock=clock)
    cache.store("怎么重置密码", ans("点忘记密码"))
    clock.advance(4.999)
    assert cache.lookup("怎么重置密码").hit is True
    clock.advance(0.001)
    assert cache.lookup("怎么重置密码").hit is False
    assert len(cache) == 0, "过期项在查询时被清掉"

    # 7) 命中不续期：TTL 以写入时间为准，避免缓存永不失效。
    clock = FakeClock()
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=10, clock=clock)
    cache.store("发票怎么开", ans("在订单页申请"))
    clock.advance(6)
    assert cache.lookup("发票怎么开").hit
    clock.advance(5)  # 距写入 11s，虽然刚用过也已过期
    assert cache.lookup("发票怎么开").hit is False

    # 8) 不缓存带工具调用 / 有副作用 / 未校验 / 空回答的回答。
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=100, clock=clock)
    assert cache.store("帮我取消订单 12345", ans("已取消", tool_calls=[{"name": "cancel_order"}])) is False
    assert cache.store("帮我转账", ans("已转账", side_effecting=True)) is False
    assert cache.store("瞎编一个问题", ans("瞎编", validated=False)) is False
    assert cache.store("空回答", ans("   ")) is False
    assert cache.bypass_reasons == {"has_tool_calls": 1, "side_effecting": 1, "unvalidated": 1, "empty_answer": 1}
    assert cache.lookup("帮我取消订单 12345").hit is False

    # 9) 实体护栏：“取消订单 A” 与 “取消订单 B” 相似度很高，但不能互相命中。
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=100, clock=clock)
    cache.store("取消订单 A100 的下单", ans("已取消 A100"))
    near = cache.lookup("取消订单 B200 的下单")
    assert near.hit is False and near.reason == "entity_mismatch", near.reason
    assert cache.lookup("取消订单 A100 的下单").hit is True

    # 10) 引号内实体同样生效。
    cache = SemanticCache(threshold=0.3, capacity=10, ttl=100, clock=clock)
    cache.store('把"早上好"翻译成英文', ans("Good morning"))
    assert cache.lookup('把"晚上好"翻译成英文').hit is False
    assert cache.lookup('把"早上好"翻译成英文').hit is True

    # 11) 容量满按 LRU 淘汰：最近被查过的留下，最久没用的先走。
    clock = FakeClock()
    cache = SemanticCache(threshold=0.99, capacity=2, ttl=1000, clock=clock)
    cache.store("问题一", ans("一"))
    clock.advance(1)
    cache.store("问题二", ans("二"))
    clock.advance(1)
    assert cache.lookup("问题一").hit  # 问题一变成最近使用
    clock.advance(1)
    cache.store("问题三", ans("三"))   # 淘汰问题二
    assert cache.keys() == ["问题一", "问题三"], cache.keys()
    assert cache.lookup("问题二").hit is False
    assert cache.lookup("问题一").hit and cache.lookup("问题三").hit

    # 12) capacity = 0：什么也存不下。
    cache = SemanticCache(threshold=0.5, capacity=0, ttl=100, clock=clock)
    cache.store("问题", ans("答案"))
    assert len(cache) == 0 and cache.lookup("问题").hit is False

    # 13) 按条删除：缓存了错误回答可以删掉。
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=100, clock=clock)
    cache.store("年假几天", ans("10 天"), tenant="hr")
    assert cache.delete("年假几天", tenant="other") is False, "不能删别的租户的条目"
    assert cache.delete("年假几天", tenant="hr") is True
    assert cache.lookup("年假几天", tenant="hr").hit is False
    assert cache.delete("不存在", tenant="hr") is False

    # 14) 按标签失效：源数据变化时整类失效。
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=1000, clock=clock)
    cache.store("价格多少", ans("99", tags=("price",)))
    cache.store("库存多少", ans("10", tags=("stock",)))
    assert cache.invalidate_tag("price") == 1
    assert cache.lookup("价格多少").hit is False and cache.lookup("库存多少").hit is True

    # 15) 阈值应按误报率定：在标注样本上选阈值，而不是按命中率。
    samples = [
        ("北京今天天气怎么样", "北京今天天气如何", True),
        ("怎么重置密码", "如何重置密码", True),
        ("退款多久到账", "退款几天到账", True),
        ("取消订单 A100", "取消订单 B200", False),
        ("北京天气", "上海天气", False),
        ("怎么开发票", "怎么开发邮箱", False),
    ]
    cache = SemanticCache(threshold=0.0, capacity=100, ttl=1000, clock=FakeClock())
    for i, (q, _, _) in enumerate(samples):
        cache.store(q, ans(f"a{i}"), tenant="t")
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.99]
    best_threshold, best_fp = None, 1.0
    for th in thresholds:
        cache.threshold = th
        fp = fn = 0
        for q, variant, should_hit in samples:
            hit = cache.lookup(variant, tenant="t").hit
            if hit and not should_hit:
                fp += 1
            if not hit and should_hit:
                fn += 1
        if fp < best_fp:
            best_threshold, best_fp = th, fp
    assert best_threshold is not None and best_fp == 0, best_threshold

    # 16) 同一 query 重复 store 是覆盖（幂等），不会产生两条。
    cache = SemanticCache(threshold=0.5, capacity=10, ttl=1000, clock=clock)
    cache.store("同一个问题", ans("第一版"))
    cache.store("同一个问题", ans("第二版"))
    assert len(cache) == 1 and cache.lookup("同一个问题").answer.text == "第二版"

    # 17) 构造参数校验。
    for kw in ({"threshold": -0.1}, {"threshold": 1.1}, {"capacity": -1}, {"ttl": -1}):
        try:
            SemanticCache(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} 应报错")

    # 18) 流式 / 顺序无关：写入顺序不影响同一 query 的命中结果。
    queries = ["怎么退款", "发货要多久", "能改地址吗"]
    clock = FakeClock()
    cache_a = SemanticCache(threshold=0.5, capacity=10, ttl=100, clock=clock)
    for i, q in enumerate(queries):
        cache_a.store(q, ans(str(i)))
    clock = FakeClock()
    cache_b = SemanticCache(threshold=0.5, capacity=10, ttl=100, clock=clock)
    for i, q in reversed(list(enumerate(queries))):
        cache_b.store(q, ans(str(i)))
    for q in queries:
        ra, rb = cache_a.lookup(q), cache_b.lookup(q)
        assert ra.hit == rb.hit and ra.answer.text == rb.answer.text

    # 19) 并发读写不出错，命中结果一致。
    import threading

    cache = SemanticCache(threshold=0.5, capacity=50, ttl=1000, clock=FakeClock())
    errors: list[BaseException] = []


    def worker(seed: int) -> None:
        try:
            for i in range(200):
                q = f"并发问题 {(seed * 7 + i) % 10}"
                cache.store(q, ans(q))
                cache.lookup(q, tenant="t")
                if i % 50 == 0:
                    cache.delete(q)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)


    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(cache) <= 50

    # 20) 多轮任务不该走语义缓存：把上下文拼进 query 后相似度明显下降，自然不命中。
    cache = SemanticCache(threshold=0.85, capacity=10, ttl=100, clock=FakeClock())
    cache.store("北京天气", ans("晴"))
    assert cache.lookup("北京天气").hit
    assert cache.lookup("刚才你说北京天气那我们是先去公园还是先吃饭").hit is False
