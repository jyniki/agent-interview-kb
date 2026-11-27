"""code-eng-003 带抖动的指数退避重试：retry(fn, max_attempts, base, cap)。

失败后等待 Full Jitter 的 sleep = uniform(0, min(cap, base·2^(attempt-1))) 再重试；
只重试临时错误（超时、连接错误、408/429/5xx），服务端给了 Retry-After 时优先用它；
同时受总时限 deadline 约束，次数用完抛出原始异常并标注尝试次数。

复杂度：时间 O(尝试次数 × (fn 耗时 + 退避等待))，等待总量 ≤ deadline；空间 O(1)。

运行：文件末尾的自测在 `__main__` 保护块里，直接 python3 运行即执行全部断言；被 import 时没有副作用。
"""

from __future__ import annotations

import random
import time

DEFAULT_RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# 模块级固定种子的默认 RNG：同一进程内可复现；生产里每次构造自己的 os.urandom 种子即可
_DEFAULT_RNG = random.Random(0)
DEFAULT_RETRY_TYPES = (TimeoutError, ConnectionError)


class RetryableError(Exception):
    """带 HTTP 状态码的可重试异常示例；真实客户端里换成自己的异常类型即可。"""

    def __init__(self, message: str, status: int, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def error_status(exc: BaseException) -> int | None:
    """从异常里取 HTTP 状态码：status / status_code / code 三选一，常见客户端各有各的叫法。"""
    for attr in ("status", "status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """从异常里取 Retry-After（秒）。只认数字形式；HTTP-date 形式由客户端先换算成秒。"""
    raw = getattr(exc, "retry_after", None)
    if raw is None:
        for holder in (getattr(exc, "headers", None), getattr(getattr(exc, "response", None), "headers", None)):
            if holder is None:
                continue
            try:
                raw = holder.get("Retry-After", holder.get("retry-after"))
            except AttributeError:
                raw = None
            if raw is not None:
                break
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def is_retryable(exc: BaseException, retry_on=None) -> bool:
    if retry_on is not None:
        return isinstance(exc, retry_on)
    if isinstance(exc, DEFAULT_RETRY_TYPES):
        return True
    status = error_status(exc)
    return status is not None and status in DEFAULT_RETRY_STATUS


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Full Jitter：uniform(0, min(cap, base·2^(attempt-1)))。attempt 从 1 开始。"""
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return rng.uniform(0, ceiling)


def retry(
    fn,
    max_attempts: int = 3,
    base: float = 0.5,
    cap: float = 8.0,
    *,
    deadline: float | None = None,
    retry_on=None,
    sleep=time.sleep,
    rng: random.Random | None = None,
    clock=time.monotonic,
    idempotency_key: str | None = None,
    on_retry=None,
):
    """重试 fn()。

    达到最大次数或总时限后抛出最后一次的原始异常，并在异常上写入 attempts。
    idempotency_key 非空时每次尝试都调用 fn(idempotency_key=...)，保证重试可去重。
    """
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")
    if base < 0 or cap < 0:
        raise ValueError("base 和 cap 不能为负")
    rng = rng if rng is not None else _DEFAULT_RNG
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        if deadline is not None and clock() >= deadline and attempt > 1:
            break  # 剩余时间已经不够发起下一次尝试
        try:
            if idempotency_key is None:
                return fn()
            return fn(idempotency_key=idempotency_key)
        except BaseException as exc:  # noqa: BLE001 —— 判断之后再决定抛还是重试
            if not isinstance(exc, Exception):
                raise  # KeyboardInterrupt / SystemExit 之类不重试
            last_exc = exc
            if not is_retryable(exc, retry_on) or attempt >= max_attempts:
                exc.attempts = attempt
                raise
            delay = retry_after_seconds(exc)
            if delay is None:
                delay = backoff_delay(attempt, base, cap, rng)
            if deadline is not None and clock() + delay > deadline:
                exc.attempts = attempt
                exc.retry_exhausted = "deadline"
                raise
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)

    # 只有 deadline 在两次尝试之间耗尽才会走到这里
    assert last_exc is not None
    last_exc.attempts = attempt - 1
    last_exc.retry_exhausted = "deadline"
    raise last_exc


# ---------------------------------------------------------------- 自测
# 只在直接运行时执行；被 import 时不跑断言、不产生任何副作用。
# 直接 `python3 <本文件>` 即可运行；CI 的 run_snippets 也是整份执行。
if __name__ == "__main__":
    # ---------------------------------------------------------------- 断言

    class FakeClock:
        def __init__(self, start: float = 1000.0):
            self.now = float(start)

        def __call__(self) -> float:
            return self.now

        def advance(self, dt: float) -> None:
            self.now += dt


    class Recorder:
        """记录退避等待并推进假时钟，不 sleep 真实时间。"""

        def __init__(self, clock: FakeClock):
            self.clock = clock
            self.delays: list[float] = []

        def __call__(self, dt: float) -> None:
            assert dt >= 0, "退避时间不能为负"
            self.delays.append(dt)
            self.clock.advance(dt)


    class MaxRng(random.Random):
        """uniform(0, x) 恒取上界，用来验证天花板 min(cap, base·2^(n-1))。"""

        def uniform(self, a, b):
            return b


    class ZeroRng(random.Random):
        def uniform(self, a, b):
            return a


    def flaky(fail_times, exc_factory, value="ok"):
        state = {"n": 0}
        seen_keys = []

        def fn(**kwargs):
            state["n"] += 1
            if "idempotency_key" in kwargs:
                seen_keys.append(kwargs["idempotency_key"])
            if state["n"] <= fail_times:
                raise exc_factory(state["n"])
            return value

        fn.state = state
        fn.seen_keys = seen_keys
        return fn


    # 1) 第 3 次成功：只尝试 3 次、只等待 2 次，返回原值。
    clock, sleeper = FakeClock(), Recorder(FakeClock())
    sleeper.clock = clock
    fn = flaky(2, lambda n: TimeoutError("timeout"))
    assert retry(fn, max_attempts=5, base=0.1, cap=10, sleep=sleeper, rng=ZeroRng(), clock=clock) == "ok"
    assert fn.state["n"] == 3 and len(sleeper.delays) == 2

    # 2) 次数上限：一直失败时恰好调用 max_attempts 次，等待 max_attempts-1 次。
    for attempts in (1, 3, 7):
        clock = FakeClock()
        sleeper = Recorder(clock)
        fn = flaky(10_000, lambda n: ConnectionError("reset"))
        try:
            retry(fn, max_attempts=attempts, base=0.01, cap=0.05, sleep=sleeper, rng=ZeroRng(), clock=clock)
        except ConnectionError as exc:
            assert exc.attempts == attempts, f"异常上要带尝试次数，实际 {exc.attempts}"
        else:
            raise AssertionError("次数用完必须抛出")
        assert fn.state["n"] == attempts
        assert len(sleeper.delays) == attempts - 1, "最后一次失败后不再等待"

    # 3) 抛出的是原始异常对象，不是包装过的。
    clock, sleeper = FakeClock(), Recorder(FakeClock())
    sleeper.clock = clock
    boom = TimeoutError("original")
    fn = flaky(10, lambda n: boom)
    try:
        retry(fn, max_attempts=2, base=0.1, cap=1, sleep=sleeper, rng=ZeroRng(), clock=clock)
    except TimeoutError as exc:
        assert exc is boom
    else:
        raise AssertionError("必须抛出")

    # 4) 不可重试错误：参数错误 / 401 / 403 / ValueError 直接抛，一次都不等。
    for exc in (ValueError("bad arg"), RetryableError("unauthorized", 401), RetryableError("forbidden", 403), RetryableError("not found", 404)):
        clock = FakeClock()
        sleeper = Recorder(clock)
        fn = flaky(10, lambda n, e=exc: e)
        try:
            retry(fn, max_attempts=5, base=1, cap=8, sleep=sleeper, rng=ZeroRng(), clock=clock)
        except Exception as raised:
            assert raised is exc and getattr(raised, "attempts") == 1
        else:
            raise AssertionError(f"{exc!r} 不应被重试")
        assert fn.state["n"] == 1 and sleeper.delays == [], "不可重试错误不能产生等待"

    # 5) 可重试错误：超时、连接错误、429、500、503 都会重试。
    for exc in (TimeoutError("t"), ConnectionError("c"), RetryableError("rate", 429), RetryableError("boom", 500), RetryableError("unavailable", 503), RetryableError("gateway", 502)):
        clock = FakeClock()
        sleeper = Recorder(clock)
        fn = flaky(1, lambda n, e=exc: e, value="finally")
        assert retry(fn, max_attempts=2, base=0.01, cap=1, sleep=sleeper, rng=ZeroRng(), clock=clock) == "finally"
        assert fn.state["n"] == 2, f"{exc!r} 应当重试"

    # 6) 状态码挂在 status_code 上也要认。
    class StatusCodeError(Exception):
        def __init__(self, status_code):
            super().__init__(f"http {status_code}")
            self.status_code = status_code


    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(1, lambda n: StatusCodeError(503), value="ok")
    assert retry(fn, max_attempts=2, base=0.01, cap=1, sleep=sleeper, rng=ZeroRng(), clock=clock) == "ok"

    # 7) Full Jitter 的天花板：base=0.5, cap=3 → 0.5, 1, 2, 3, 3。
    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(99, lambda n: TimeoutError("t"))
    try:
        retry(fn, max_attempts=6, base=0.5, cap=3, sleep=sleeper, rng=MaxRng(), clock=clock)
    except TimeoutError:
        pass
    assert sleeper.delays == [0.5, 1.0, 2.0, 3.0, 3.0], f"指数退避 + 封顶，实际 {sleeper.delays}"

    # 8) 抖动落在 [0, 天花板] 内，且固定种子可复现；抖动确实让等待错开。
    clock, sleeper = FakeClock(), Recorder(FakeClock())
    sleeper.clock = clock
    fn = flaky(99, lambda n: TimeoutError("t"))
    try:
        retry(fn, max_attempts=6, base=1, cap=8, sleep=sleeper, rng=random.Random(42), clock=clock)
    except TimeoutError:
        pass
    delays_a = list(sleeper.delays)
    clock, sleeper = FakeClock(), Recorder(FakeClock())
    sleeper.clock = clock
    fn = flaky(99, lambda n: TimeoutError("t"))
    try:
        retry(fn, max_attempts=6, base=1, cap=8, sleep=sleeper, rng=random.Random(42), clock=clock)
    except TimeoutError:
        pass
    assert delays_a == sleeper.delays, "固定种子必须可复现"
    assert all(0 <= d <= ceiling for d, ceiling in zip(delays_a, [1, 2, 4, 8, 8])), delays_a
    assert len(set(delays_a)) > 1, "相同种子下多次抖动不应完全一样"
    assert all(d <= 8 for d in delays_a)

    # 9) Retry-After 优先于退避：服务端说 2 秒就等 2 秒，不走 jitter。
    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(1, lambda n: RetryableError("slow down", 429, retry_after=2.0), value="ok")
    assert retry(fn, max_attempts=3, base=10, cap=10, sleep=sleeper, rng=MaxRng(), clock=clock) == "ok"
    assert sleeper.delays == [2.0], f"应使用 Retry-After，实际 {sleeper.delays}"

    # 10) Retry-After 放在 headers 里（字符串）也能读。
    class HeaderError(Exception):
        def __init__(self):
            super().__init__("429")
            self.status = 429
            self.headers = {"Retry-After": "1.5"}


    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(1, lambda n: HeaderError(), value="ok")
    assert retry(fn, max_attempts=2, base=0.01, cap=1, sleep=sleeper, rng=ZeroRng(), clock=clock) == "ok"
    assert sleeper.delays == [1.5]

    # 11) 坏掉的 Retry-After 不影响流程：非数字、负数都回落到退避。
    for raw in ("Wed, 21 Oct 2015 07:28:00 GMT", -1, None):
        clock = FakeClock()
        sleeper = Recorder(clock)
        fn = flaky(1, lambda n, r=raw: RetryableError("x", 503, retry_after=r), value="ok")
        assert retry(fn, max_attempts=2, base=0.2, cap=1, sleep=sleeper, rng=MaxRng(), clock=clock) == "ok"
        assert sleeper.delays == [0.2], f"Retry-After={raw!r} 应回落退避，实际 {sleeper.delays}"

    # 12) deadline：下一次等待会超出总时限时直接失败，不多睡、不多试。
    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(99, lambda n: TimeoutError("t"))
    try:
        retry(fn, max_attempts=10, base=1, cap=10, deadline=1000.5, sleep=sleeper, rng=MaxRng(), clock=clock)
    except TimeoutError as exc:
        assert exc.attempts == 1 and exc.retry_exhausted == "deadline"
    else:
        raise AssertionError("超过 deadline 必须失败")
    assert fn.state["n"] == 1, "第一次等待 1s 超过剩余 0.5s，不应再尝试"
    assert sleeper.delays == [] and clock.now == 1000.0

    # 13) deadline 内能完成就照常重试：总等待不超过剩余时间。
    clock = FakeClock(start=0)
    sleeper = Recorder(clock)
    fn = flaky(2, lambda n: TimeoutError("t"), value="ok")
    assert retry(fn, max_attempts=5, base=0.1, cap=1, deadline=10, sleep=sleeper, rng=MaxRng(), clock=clock) == "ok"
    assert fn.state["n"] == 3 and abs(clock.now - 0.3) < 1e-9 and sum(sleeper.delays) <= 10

    # 14) fn 自己耗时间把 deadline 耗光：到点后不再发起新的尝试。
    clock = FakeClock(start=0)
    sleeper = Recorder(clock)
    calls = {"n": 0}


    def slow_and_fail():
        calls["n"] += 1
        clock.advance(3)  # 每次调用消耗 3 秒，退避本身是 0
        raise TimeoutError("slow")


    try:
        retry(slow_and_fail, max_attempts=10, base=0, cap=0, deadline=6, sleep=sleeper, rng=ZeroRng(), clock=clock)
    except TimeoutError as exc:
        assert exc.attempts == 2 and exc.retry_exhausted == "deadline"
    else:
        raise AssertionError("deadline 耗尽应失败")
    assert calls["n"] == 2, "第二次调用结束时已到 deadline，第三次不能再发起"
    assert clock.now == 6.0 and sum(sleeper.delays) == 0

    # 15) 幂等键：每次尝试拿到同一个键；不传时 fn 不接受额外参数。
    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(2, lambda n: TimeoutError("t"), value="ok")
    assert retry(fn, max_attempts=3, base=0.01, cap=1, sleep=sleeper, rng=ZeroRng(), clock=clock, idempotency_key="req-abc") == "ok"
    assert fn.seen_keys == ["req-abc"] * 3

    # 16) 自定义 retry_on：默认不重试的异常可以按需重试，反之亦然。
    clock = FakeClock()
    sleeper = Recorder(clock)
    fn = flaky(1, lambda n: KeyError("k"), value="ok")
    assert retry(fn, max_attempts=2, base=0.01, cap=1, retry_on=(KeyError,), sleep=sleeper, rng=ZeroRng(), clock=clock) == "ok"

    # 17) on_retry 回调按顺序报告 (attempt, delay, exc)。
    clock = FakeClock()
    sleeper = Recorder(clock)
    reports = []
    fn = flaky(2, lambda n: TimeoutError("t"), value="ok")
    retry(fn, max_attempts=5, base=1, cap=8, sleep=sleeper, rng=MaxRng(), clock=clock, on_retry=lambda a, d, e: reports.append((a, d, type(e).__name__)))
    assert reports == [(1, 1.0, "TimeoutError"), (2, 2.0, "TimeoutError")]

    # 18) 参数校验。
    try:
        retry(lambda: 1, max_attempts=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_attempts=0 应报错")
    try:
        retry(lambda: 1, base=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("负 base 应报错")

    # 19) 重试次数上限的链路放大：三层各 retry(3) 时底层最多被调用 27 次。
    def layer_retry(inner):
        return lambda: retry(inner, max_attempts=3, base=0, cap=0, sleep=lambda _: None, rng=ZeroRng())


    counter = {"n": 0}


    def leaf():
        counter["n"] += 1
        raise TimeoutError("down")


    try:
        layer_retry(layer_retry(layer_retry(leaf)))()
    except TimeoutError:
        pass
    assert counter["n"] == 27, f"3 层 × 3 次 = 27，实际 {counter['n']}"

    # 20) KeyboardInterrupt 等 BaseException 不重试、不吞掉。
    clock = FakeClock()
    sleeper = Recorder(clock)
    hits = {"n": 0}


    def interrupt():
        hits["n"] += 1
        raise KeyboardInterrupt


    try:
        retry(interrupt, max_attempts=5, sleep=sleeper, clock=clock)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("不应重试 KeyboardInterrupt")
    assert hits["n"] == 1 and sleeper.delays == []
