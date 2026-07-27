"""同步生成器 → 异步迭代 桥接的测试。

最关键的一条是 `test_generator_body_runs_in_one_thread`：整个生成器必须跑在同一线程里，
否则 `core.budget` 的运行预算（contextvars）会在换线程时丢失，超限拦截随之失效。
这正是这里不用 `starlette.concurrency.iterate_in_threadpool` 的原因。
"""

import asyncio
import contextvars
import threading

import pytest

from api.streaming import aiter_in_thread

_probe: contextvars.ContextVar[str | None] = contextvars.ContextVar("probe", default=None)


def _collect(make_iter, **kwargs):
    async def main():
        return [item async for item in aiter_in_thread(make_iter, **kwargs)]

    return asyncio.run(main())


def test_yields_all_items_in_order():
    assert _collect(lambda: iter([1, 2, 3])) == [1, 2, 3]


def test_empty_iterator():
    assert _collect(lambda: iter([])) == []


def test_exception_propagates_to_consumer():
    def gen():
        yield 1
        raise ValueError("生成器内部炸了")

    async def main():
        out = []
        async for item in aiter_in_thread(gen):
            out.append(item)
        return out

    with pytest.raises(ValueError, match="生成器内部炸了"):
        asyncio.run(main())


def test_generator_body_runs_in_one_thread():
    """回归：contextvars 在整个生成器生命周期内必须保持可见。"""
    threads: list[str] = []

    def gen():
        _probe.set("set-inside-generator")
        for i in range(6):
            threads.append(threading.current_thread().name)
            yield {"i": i, "seen": _probe.get()}

    items = _collect(gen)

    assert [it["i"] for it in items] == list(range(6))
    assert all(it["seen"] == "set-inside-generator" for it in items), (
        "contextvar 在迭代过程中丢失，说明生成器被换线程执行了"
    )
    assert len(set(threads)) == 1, f"生成器跨了多个线程：{set(threads)}"


def test_does_not_run_on_the_event_loop_thread():
    main_thread = threading.current_thread().name
    where: list[str] = []

    def gen():
        where.append(threading.current_thread().name)
        yield 1

    _collect(gen)
    assert where and where[0] != main_thread


def test_early_consumer_exit_does_not_hang():
    """消费侧提前退出时不能把工作线程永久卡在有界队列上。"""
    produced = []

    def gen():
        for i in range(1000):
            produced.append(i)
            yield i

    async def main():
        async for item in aiter_in_thread(gen, queue_size=2):
            if item == 3:
                break
        return True

    assert asyncio.run(main()) is True
    # 背压生效：不该把 1000 个元素全生产出来
    assert len(produced) < 1000


def test_backpressure_respects_queue_size():
    """队列满时生产侧要被挡住，而不是把所有元素堆进内存。"""
    produced = 0
    gate = threading.Event()

    def gen():
        nonlocal produced
        for i in range(50):
            produced += 1
            yield i
        gate.set()

    async def main():
        agen = aiter_in_thread(gen, queue_size=2)
        first = await agen.__anext__()
        # 只取了一个元素，生产侧最多领先 queue_size + 1 个
        snapshot = produced
        await agen.aclose()
        return first, snapshot

    first, snapshot = asyncio.run(main())
    assert first == 0
    assert snapshot <= 5, f"背压失效，已生产 {snapshot} 个元素"
