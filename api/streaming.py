"""同步生成器 → 异步迭代 的桥接。

为什么不用 `starlette.concurrency.iterate_in_threadpool`：它每取一个元素就向线程池
提交一次任务，相邻元素可能落在不同线程上。而本项目的运行预算（`core.budget`）是靠
contextvars 传递的 —— `with run_budget():` 在生成器体内设置的变量只对设置它的那个
线程可见，一旦换线程，预算计数就归零、超限拦截随之失效。

所以这里让整个生成器跑在**同一个**后台线程里，用一个有界队列把事件送回事件循环。
队列有界意味着客户端读得慢时会对 Agent 产生背压，而不是把事件无限堆在内存里。
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

_DONE = object()


async def aiter_in_thread(
    make_iter: Callable[[], Iterator[Any]],
    queue_size: int = 64,
) -> AsyncIterator[Any]:
    """在单个后台线程里消费 `make_iter()`，把产出的元素异步 yield 出来。

    生成器内部抛出的异常会在消费侧原样重抛，因此调用方可以照常 try/except。
    消费侧提前退出（例如客户端断开）时，后台线程会在下一次投递时发现取消标志并收尾。
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    cancelled = threading.Event()

    def emit(item: Any) -> bool:
        """从工作线程投递一个元素；返回 False 表示消费侧已经不在了。"""
        try:
            asyncio.run_coroutine_threadsafe(queue.put(item), loop).result()
            return True
        except Exception:
            return False

    def worker() -> None:
        try:
            for item in make_iter():
                if cancelled.is_set() or not emit(item):
                    return
        except BaseException as exc:
            # 预算耗尽等异常也要送到消费侧，由它决定怎么呈现
            if not cancelled.is_set():
                emit(exc)
            return
        emit(_DONE)

    thread = threading.Thread(target=worker, name="agent-stream", daemon=True)
    thread.start()
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancelled.set()
        # 工作线程可能正卡在有界队列的 put 上，腾出空间让它得以看到取消标志
        while not queue.empty():
            queue.get_nowait()
