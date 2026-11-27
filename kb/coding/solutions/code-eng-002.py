"""code-eng-002 令牌桶限流器。

桶容量 C、补充速率 r；每次请求按距上次访问的时间惰性补充 min(C, r·Δt) 个令牌，
够就扣减放行，不够就拒绝或等到超时。按 token 数限流时扣减量是请求的预估 token 数。

复杂度：
- try_acquire / acquire 每次调用 O(1) 时间、O(C 的等待轮数) 次 clock 调用；
- 空间 O(1)。锁只保护“补充 + 扣减”这段临界区。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """惰性补充的令牌桶。clock 可注入，便于测试且不 sleep 真实时间。"""

    def __init__(self, capacity: float, rate: float, *, clock=time.monotonic, initial: float | None = None):
        if capacity < 0:
            raise ValueError("capacity 不能为负")
        if rate < 0:
            raise ValueError("rate 不能为负")
        self.capacity = float(capacity)
        self.rate = float(rate)
        self._clock = clock
        self._lock = threading.Lock()
        start = self.capacity if initial is None else min(float(initial), self.capacity)
        self._tokens = max(0.0, start)
        self._last = clock()

    @property
    def tokens(self) -> float:
        """当前可用令牌数（会先按时间补充，但不扣减）。"""
        with self._lock:
            self._refill_locked()
            return self._tokens

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now
        # elapsed <= 0（时钟回拨）时不补充，也不改写 _last：
        # 回拨期间经过的时间不能兑现成令牌，回到原时间点后也不会重复补充。

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """够就扣减并返回 True，不够立即返回 False（不等待）。"""
        self._check(tokens)
        with self._lock:
            self._refill_locked()
            if self._tokens + 1e-12 >= tokens:
                self._tokens = max(0.0, self._tokens - tokens)
                return True
            return False

    def time_until(self, tokens: float = 1.0) -> float:
        """还差多少秒才攒够 tokens；超过容量则返回 inf。"""
        self._check(tokens)
        if tokens > self.capacity:
            return float("inf")
        with self._lock:
            self._refill_locked()
            deficit = tokens - self._tokens
        if deficit <= 0:
            return 0.0
        if self.rate == 0:
            return float("inf")
        return deficit / self.rate

    def acquire(self, tokens: float = 1.0, timeout: float | None = 0.0, *, sleep=time.sleep) -> None:
        """等到有足够令牌后扣减。超时抛 TimeoutError；timeout 为 None 表示一直等。"""
        self._check(tokens)
        if timeout is not None and timeout < 0:
            raise ValueError("timeout 不能为负")
        deadline = None if timeout is None else self._clock() + timeout
        while True:
            if self.try_acquire(tokens):
                return
            wait = self.time_until(tokens)
            if wait == float("inf"):
                raise TimeoutError(f"请求 {tokens} 个令牌超过桶容量 {self.capacity}，永远不会成功")
            if deadline is not None:
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError(f"等待 {timeout}s 仍未拿到 {tokens} 个令牌")
                wait = min(wait, remaining)
            sleep(max(wait, 0.0))

    def _check(self, tokens: float) -> None:
        if not isinstance(tokens, (int, float)) or isinstance(tokens, bool):
            raise TypeError("tokens 必须是数值")
        if tokens < 0:
            raise ValueError("tokens 不能为负")


class FakeClock:
    """可控时钟：advance 推进时间，同时被 TokenBucket 当作 monotonic 使用。"""

    def __init__(self, start: float = 1000.0):
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt

    def sleep(self, dt: float) -> None:
        self.now += dt


# ---------------------------------------------------------------- 自测
# 只在直接运行时执行；被 import 时不跑断言、不产生任何副作用。
# 直接 `python3 <本文件>` 即可运行；CI 的 run_snippets 也是整份执行。
if __name__ == "__main__":
    # ---------------------------------------------------------------- 断言

    # 1) 空桶按速率补充：容量 5、每秒 10 个，0.2s 补 2 个。
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, rate=10, clock=clock, initial=0)
    assert bucket.try_acquire(1) is False
    clock.advance(0.1)
    assert bucket.try_acquire(1.0) is True, "0.1s × 10/s = 1 个令牌"
    assert bucket.try_acquire(1.0) is False, "刚扣完，不存在浮点误差导致的二次放行"

    # 2) 亚秒精度：不能用整数秒截断时间差。0.09s 只补 0.9 个令牌。
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, rate=10, clock=clock, initial=0)
    clock.advance(0.09)
    assert bucket.try_acquire(1.0) is False
    clock.advance(0.011)  # 累计 0.101s → 1.01 个
    assert bucket.try_acquire(1.0) is True

    # 3) 分数令牌会累积：每次 0.05s 补 0.5 个，两次之后刚好够 1 个。
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, rate=10, clock=clock, initial=0)
    clock.advance(0.05)
    assert bucket.try_acquire(1.0) is False
    clock.advance(0.05)
    assert bucket.try_acquire(1.0) is True

    # 4) 补充不超过容量：空闲 100 秒也只有 C 个令牌，突发上限为 C。
    clock = FakeClock()
    bucket = TokenBucket(capacity=3, rate=10, clock=clock, initial=0)
    clock.advance(100)
    assert bucket.tokens == 3
    assert bucket.try_acquire(3) is True
    assert bucket.try_acquire(1) is False

    # 5) 容量满时拒绝：初始化默认装满。
    bucket = TokenBucket(capacity=2, rate=1, clock=FakeClock())
    assert bucket.try_acquire(2) is True
    assert bucket.try_acquire(0.001) is False

    # 6) 按 token 数扣减（LLM 场景）：一次请求按预估 token 数扣。
    clock = FakeClock()
    bucket = TokenBucket(capacity=1000, rate=100, clock=clock)
    assert bucket.try_acquire(800) is True
    assert bucket.try_acquire(300) is False, "剩 200，不够 300"
    clock.advance(1.0)  # 补 100 → 300
    assert bucket.try_acquire(300) is True

    # 7) capacity=0：永远拿不到，等待会超时而不是死循环。
    clock = FakeClock()
    bucket = TokenBucket(capacity=0, rate=100, clock=clock)
    assert bucket.try_acquire(0) is True
    assert bucket.try_acquire(1) is False
    try:
        bucket.acquire(1, timeout=5, sleep=clock.sleep)
    except TimeoutError:
        pass
    else:
        raise AssertionError("capacity=0 应当超时")
    assert clock.now == 1000.0, "无限等待会被 time_until 直接判定为不可能，不能空转"

    # 8) 请求量超过容量：无论如何都等不到，应立刻抛 TimeoutError 而不是睡满 timeout。
    clock = FakeClock()
    bucket = TokenBucket(capacity=5, rate=1000, clock=clock)
    try:
        bucket.acquire(6, timeout=999, sleep=clock.sleep)
    except TimeoutError:
        pass
    else:
        raise AssertionError("超过容量的请求不可能成功")
    assert clock.now == 1000.0, "不可能成功的请求不应消耗等待时间"

    # 9) acquire 会等到令牌：容量 1、速率 1/s，等 2 秒够 2 个。
    clock = FakeClock()
    bucket = TokenBucket(capacity=1, rate=1, clock=clock, initial=0)
    bucket.acquire(1, timeout=None, sleep=clock.sleep)
    assert abs(clock.now - 1001.0) < 1e-9
    assert bucket.tokens < 1e-9

    # 10) acquire 超时：timeout 短于攒够令牌所需时间 → TimeoutError，推进的时间不超过 timeout。
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, rate=1, clock=clock, initial=0)
    try:
        bucket.acquire(5, timeout=2, sleep=clock.sleep)
    except TimeoutError:
        pass
    else:
        raise AssertionError("2s 内攒不够 5 个令牌")
    assert clock.now - 1000.0 <= 2 + 1e-9

    # 11) 时钟回拨：不能凭空补令牌，也不能在时间恢复后重复补。
    clock = FakeClock(start=1000.0)
    bucket = TokenBucket(capacity=10, rate=10, clock=clock, initial=5)
    clock.now = 900.0  # 时间倒退 100 秒
    assert bucket.tokens == 5, "回拨不能补也不能扣"
    clock.now = 1000.0  # 回到原值
    assert bucket.tokens == 5, "回拨期间的时间不能兑现成令牌"
    clock.advance(0.5)  # 正常前进 0.5 秒 → +5
    assert abs(bucket.tokens - 10) < 1e-9

    # 12) 并发：容量 10、速率为 0，16 个线程同时抢，恰好 10 个成功。
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, rate=0, clock=clock, initial=10)
    barrier = threading.Barrier(16)
    wins = []
    lock = threading.Lock()


    def worker():
        barrier.wait()
        ok = bucket.try_acquire(1)
        with lock:
            wins.append(ok)


    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(wins) == 10, f"应恰好 10 次成功，实际 {sum(wins)}"
    assert bucket.tokens == 0

    # 13) 并发下补充与扣减在同一把锁内：并发推进时钟不会重复补充。
    clock = FakeClock()
    bucket = TokenBucket(capacity=100, rate=10, clock=clock, initial=0)
    clock.advance(1.0)  # 补 10
    barrier = threading.Barrier(8)
    seen = []


    def peek():
        barrier.wait()
        with lock:
            seen.append(bucket.tokens)


    threads = [threading.Thread(target=peek) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [10] * 8, f"惰性补充只应发生一次，实际 {seen}"

    # 14) 参数校验。
    for bad in (-1, -0.5):
        try:
            TokenBucket(capacity=1, rate=1, clock=FakeClock()).acquire(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("负令牌数应报错")
    try:
        TokenBucket(capacity=-1, rate=1, clock=FakeClock())
    except ValueError:
        pass
    else:
        raise AssertionError("负容量应报错")
    try:
        TokenBucket(capacity=1, rate=1, clock=FakeClock()).time_until(0)
    except Exception:
        raise AssertionError("time_until(0) 不应报错")
    assert TokenBucket(capacity=1, rate=1, clock=FakeClock()).time_until(1) == 0.0

    # 15) time_until 精确：速率 10/s，剩 0.25 个，需要 0.025 秒。
    clock = FakeClock()
    bucket = TokenBucket(capacity=10, rate=10, clock=clock, initial=0.75)
    assert abs(bucket.time_until(1) - 0.025) < 1e-9
    assert bucket.time_until(11) == float("inf")
