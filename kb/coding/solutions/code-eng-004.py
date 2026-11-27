"""code-eng-004 LRU 缓存：get(key) 和 put(key, value) 都是 O(1)，容量满时淘汰最久未使用的项。

哈希表 key → 双向链表节点（带 key、value、大小、过期时间），链表头是最近使用、尾是最久未使用；
get 命中把节点移到表头，put 更新或插入表头，超出条数 / 字节上限就从表尾淘汰。

扩展（对应追问链）：每项 TTL、按字节限制容量、线程安全。

复杂度：
- get / put / delete 均摊 O(1)（淘汰循环的每一步都删掉一项，总量与写入次数同阶）；
- 空间 O(capacity)。所有读写都在同一把锁内，get 也会改链表。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

_MISSING = object()


class _Node:
    __slots__ = ("key", "value", "size", "expire_at", "prev", "next")

    def __init__(self, key=None, value=None, size=0, expire_at=None):
        self.key = key
        self.value = value
        self.size = size
        self.expire_at = expire_at
        self.prev: Optional["_Node"] = None
        self.next: Optional["_Node"] = None


class LRUCache:
    """容量按条数（capacity）限制，可选按字节（max_bytes）限制，可选统一 TTL。"""

    def __init__(
        self,
        capacity: int,
        *,
        ttl: float | None = None,
        max_bytes: int | None = None,
        sizer: Callable[[Any, Any], int] | None = None,
        clock=time.monotonic,
    ):
        if capacity < 0:
            raise ValueError("capacity 不能为负")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes 不能为负")
        if ttl is not None and ttl < 0:
            raise ValueError("ttl 不能为负")
        self.capacity = capacity
        self.max_bytes = max_bytes
        self.ttl = ttl
        self._sizer = sizer or (lambda key, value: 1)
        self._clock = clock
        self._lock = threading.RLock()
        self._map: dict[Any, _Node] = {}
        self._bytes = 0
        # 头尾哨兵：插入、删除不用判断空链表
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    # ------------------------------------------------------------ 链表操作

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None
        self._bytes -= node.size

    def _push_front(self, node: _Node) -> None:
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node
        self._bytes += node.size

    def _move_to_front(self, node: _Node) -> None:
        if self._head.next is node:
            return
        self._unlink(node)
        self._push_front(node)

    # ------------------------------------------------------------ 过期

    def _expired(self, node: _Node, now: float) -> bool:
        return node.expire_at is not None and now >= node.expire_at

    def _drop(self, node: _Node) -> None:
        self._unlink(node)
        del self._map[node.key]

    def _evict_one(self) -> bool:
        victim = self._tail.prev
        if victim is self._head:
            return False
        self._drop(victim)
        return True

    def _enforce(self) -> None:
        while len(self._map) > self.capacity:
            if not self._evict_one():
                break
        if self.max_bytes is not None:
            while self._bytes > self.max_bytes and self._evict_one():
                pass

    # ------------------------------------------------------------ 对外接口

    def get(self, key, default=None):
        """命中返回 value 并把节点移到表头；未命中或已过期返回 default。"""
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return default
            if self._expired(node, self._clock()):
                self._drop(node)  # 过期项当作未命中，同时从链表和哈希表删除
                return default
            self._move_to_front(node)
            return node.value

    def put(self, key, value) -> bool:
        """写入并移到表头。单项超过字节上限时拒绝写入并返回 False。"""
        size = self._sizer(key, value)
        if self.max_bytes is not None and size > self.max_bytes:
            return False  # 单项比整个缓存上限还大：直接拒绝，不能把别的项全挤掉
        now = self._clock()
        expire_at = None if self.ttl is None else now + self.ttl
        with self._lock:
            node = self._map.get(key)
            if node is not None:
                self._unlink(node)  # 更新：先摘下来重新计大小，节点数不变
                node.value = value
                node.size = size
                node.expire_at = expire_at
                self._push_front(node)
            else:
                node = _Node(key, value, size, expire_at)
                self._map[key] = node
                self._push_front(node)
            self._enforce()
            return True

    def delete(self, key) -> bool:
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return False
            self._drop(node)
            return True

    def purge_expired(self) -> int:
        """主动清理过期项，防止过期数据占着容量。"""
        now = self._clock()
        with self._lock:
            dead = [n for n in self._map.values() if self._expired(n, now)]
            for node in dead:
                self._drop(node)
            return len(dead)

    def __len__(self) -> int:
        with self._lock:
            return len(self._map)

    def __contains__(self, key) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def keys_mru_order(self) -> list:
        """按最近使用到最久未使用列出 key，便于测试。"""
        with self._lock:
            out, node = [], self._head.next
            while node is not self._tail:
                out.append(node.key)
                node = node.next
            return out

    def _check_invariants(self) -> None:
        """链表与哈希表、字节计数必须一致。"""
        forward, node = [], self._head.next
        while node is not self._tail:
            forward.append(node.key)
            assert node.prev.next is node and node.next.prev is node
            node = node.next
        backward, node = [], self._tail.prev
        while node is not self._head:
            backward.append(node.key)
            node = node.prev
        assert forward == backward[::-1]
        assert set(forward) == set(self._map) and len(forward) == len(self._map)
        assert self._bytes == sum(n.size for n in self._map.values())


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

    cache = LRUCache(2)
    cache.put(1, "a")
    cache.put(2, "b")
    assert cache.get(2) == "b" and cache.get(1) == "a"
    cache.put(3, "c")  # get(1) 让 1 成为最近使用，2 被挤掉
    assert cache.get(2) is None and cache.get(1) == "a" and cache.get(3) == "c" and len(cache) == 2
    cache._check_invariants()

    # 满容量后每个 put 都触发一次淘汰，长度始终不超过 capacity。
    cache = LRUCache(3)
    for i in range(100):
        cache.put(i, i * 10)
        assert len(cache) <= 3
    assert cache.keys_mru_order() == [99, 98, 97]
    cache._check_invariants()

    # get 也是“使用”，会刷新顺序（易错点）。
    cache = LRUCache(2)
    cache.put("x", 1)
    cache.put("y", 2)
    cache.get("x")       # x 变成最近使用
    cache.put("z", 3)    # 淘汰 y
    assert cache.get("y") is None and cache.get("x") == 1 and cache.get("z") == 3

    # put 已存在的 key：更新值并移到表头，不增加项数、不触发淘汰。
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.put("a", 100) is True
    assert cache.keys_mru_order() == ["a", "b"] and len(cache) == 2
    cache.put("c", 3)  # 淘汰 b，因为 a 刚被更新过
    assert cache.get("b") is None and cache.get("a") == 100
    cache._check_invariants()

    # capacity = 0：put 后立即淘汰，什么都不留（易错点）。
    zero = LRUCache(0)
    zero.put(1, "x")
    assert len(zero) == 0 and zero.get(1) is None
    zero._check_invariants()

    # capacity = 1：put 新键会挤掉旧键；重复 put 同一个键不淘汰。
    one = LRUCache(1)
    one.put(1, "a")
    one.put(2, "b")
    assert one.get(1) is None and one.get(2) == "b"
    one.put(2, "b2")
    assert one.get(2) == "b2"
    one._check_invariants()

    # 存 None 和存其他假值：用 sentinel 区分“命中但值是 None”和“未命中”。
    cache = LRUCache(2)
    cache.put("none", None)
    cache.put("zero", 0)
    assert cache.get("none", "MISS") is None and "none" in cache
    assert cache.get("missing", "MISS") == "MISS" and "missing" not in cache
    assert cache.get("zero") == 0

    # 删除：delete 命中返回 True，删完链表和哈希表都干净。
    cache = LRUCache(2)
    cache.put("k", 1)
    assert cache.delete("k") is True and cache.delete("k") is False
    assert len(cache) == 0 and cache.get("k") is None
    cache._check_invariants()

    # 字段校验。
    for kw in ({"capacity": -1}, {"capacity": 1, "ttl": -1}, {"capacity": 1, "max_bytes": -1}):
        try:
            LRUCache(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} 应该报错")

    # ---------------- 追问链 1：TTL
    clock = FakeClock()
    cache = LRUCache(3, ttl=10, clock=clock)
    cache.put("a", 1)
    clock.advance(9.999)
    assert cache.get("a") == 1, "未过期应命中"
    clock.advance(0.001)  # 恰好到期：按 now >= expire_at 视为过期
    assert cache.get("a") is None, "到期即未命中"
    assert len(cache) == 0, "过期项被 get 顺手删掉，不占容量"
    cache._check_invariants()

    # 已过期但仍占位的项照常参与淘汰；容量紧张时先挤掉它，新项不受影响。
    clock = FakeClock()
    cache = LRUCache(1, ttl=5, clock=clock)
    cache.put("old", 1)
    clock.advance(6)
    cache.put("new", 2)
    assert cache.get("old") is None and cache.get("new") == 2 and len(cache) == 1
    cache._check_invariants()

    # 主动清理：purge_expired 一次删掉所有过期项并返回个数，已清过的再调返回 0。
    clock = FakeClock()
    cache = LRUCache(3, ttl=5, clock=clock)
    cache.put("a", 1)
    cache.put("b", 2)
    clock.advance(5)
    assert cache.purge_expired() == 2 and len(cache) == 0 and cache.size_bytes == 0
    assert cache.purge_expired() == 0

    # TTL 不因 get 刷新（默认语义：绝对过期）。
    clock = FakeClock()
    cache = LRUCache(1, ttl=5, clock=clock)
    cache.put("k", "v")
    clock.advance(3)
    assert cache.get("k") == "v"
    clock.advance(3)  # 距写入 6s，已过期
    assert cache.get("k") is None

    # 更新已存在的 key 会重置它的 TTL。
    clock = FakeClock()
    cache = LRUCache(2, ttl=5, clock=clock)
    cache.put("k", 1)
    clock.advance(4)
    cache.put("k", 2)  # 重置 TTL
    clock.advance(4)
    assert cache.get("k") == 2

    # ---------------- 追问链 2：按字节限制容量
    nbytes = lambda k, v: len(v)  # 用值长度当大小，默认 sizer 是按条计 1

    cache = LRUCache(100, max_bytes=10, sizer=nbytes)
    assert cache.put("a", "xxxx") is True   # 4 字节
    assert cache.put("b", "yyyyyy") is True  # 6 字节，总量 10
    assert cache.put("c", "z") is True       # 需要淘汰最久未用的 a
    assert cache.size_bytes == 7 and cache.get("a") is None
    assert cache.keys_mru_order() == ["c", "b"]
    cache._check_invariants()

    # 单项超过字节上限：直接拒绝，且不能把已有项挤出去。
    cache = LRUCache(100, max_bytes=10, sizer=nbytes)
    cache.put("keep", "12345")
    assert cache.put("huge", "x" * 11) is False
    assert cache.get("keep") == "12345" and cache.size_bytes == 5 and len(cache) == 1

    # 一次淘汰不够就继续淘汰，直到总字节数不超上限。
    cache = LRUCache(10, max_bytes=10, sizer=nbytes)
    for key in "abcd":
        cache.put(key, "xx")
    assert cache.size_bytes == 8
    cache.put("e", "xxx")  # 11 > 10，淘汰 a（8-2+3=9 就够，只淘汰一项）
    assert cache.size_bytes == 9 and cache.get("a") is None and cache.keys_mru_order() == ["e", "d", "c", "b"]
    cache.put("f", "xxxxx")  # 14 > 10，需要连淘汰 b、c，剩 10 刚好不超
    assert cache.size_bytes == 10 and cache.keys_mru_order() == ["f", "e", "d"]
    cache._check_invariants()

    # 覆盖写更小的值：总字节数减少，不漏算。
    cache = LRUCache(2, max_bytes=100, sizer=nbytes)
    cache.put("k", "abcdef")
    cache.put("k", "ab")
    assert cache.size_bytes == 2
    cache.put("j", "zz")
    assert cache.size_bytes == 4

    # 字节上限为 0：任何非空项都写不进去，大小为 0 的项可以。
    cache = LRUCache(5, max_bytes=0, sizer=nbytes)
    assert cache.put("a", "") is True
    assert cache.put("b", "x") is False
    assert len(cache) == 1

    # 条数与字节数同时限制：两个条件都要满足。
    cache = LRUCache(2, max_bytes=100, sizer=nbytes)
    cache.put(1, "aaa")
    cache.put(2, "bbb")
    cache.put(3, "ccc")
    assert len(cache) == 2 and cache.get(1) is None
    cache._check_invariants()

    # 自定义 sizer：按对象估算大小。
    cache = LRUCache(10, max_bytes=4, sizer=lambda k, v: len(v))  # 2 + 3 > 4
    cache.put("a", [1, 2])
    cache.put("b", [3, 4, 5])
    assert cache.get("a") is None and cache.get("b") == [3, 4, 5] and cache.size_bytes == 3

    # ---------------- 追问链 3：LRU 的缺点 —— 一次顺序扫描冲掉热点
    cache = LRUCache(3)
    cache.put("hot", "热点")
    for i in range(10):
        cache.get("hot")  # 热点被反复访问
    for i in range(10):
        cache.put(f"scan-{i}", i)  # 顺序扫描
    assert cache.get("hot") is None, "LRU 抗不住大范围扫描，热点会被挤出去"

    # ---------------- 并发：get 也改链表，读写同锁
    import random as _random
    import threading

    cache = LRUCache(3)
    errors: list[BaseException] = []


    def hammer(seed: int) -> None:
        rng = _random.Random(seed)
        try:
            for _ in range(2000):
                key = rng.randrange(8)
                if rng.random() < 0.5:
                    cache.get(key)
                else:
                    cache.put(key, key * 2)
                if rng.random() < 0.01:
                    cache.delete(key)
                assert len(cache) <= 3
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)


    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(cache) <= 3
    cache._check_invariants()
