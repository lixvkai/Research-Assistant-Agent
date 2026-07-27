"""并行执行小工具 — 保证 contextvars（运行预算）能传播到工作线程。"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_in_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 4,
) -> list[R]:
    """对 items 并行执行 fn，按输入顺序返回结果。

    单个元素时直接同步执行，避免无谓的线程开销。
    `contextvars.copy_context()` 确保运行预算在子线程内仍然生效。
    """
    items = list(items)
    if not items:
        return []
    if len(items) == 1 or max_workers <= 1:
        return [fn(i) for i in items]

    # 每个任务需要独立的 Context 快照：同一个 Context 对象不能被并发 run()。
    # 快照必须在父线程创建，否则捕获不到父线程的预算。
    contexts = [contextvars.copy_context() for _ in items]

    def _wrapped(pair: tuple) -> R:
        ctx, item = pair
        return ctx.run(fn, item)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(_wrapped, zip(contexts, items)))
