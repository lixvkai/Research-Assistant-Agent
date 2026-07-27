"""运行预算 — 跨嵌套 Agent 的 LLM 调用数与时限控制。

问题背景：`multi_agent_collaborate` 作为工具挂在外层 ReAct 循环里，一次调用会展开为
「规划 LLM + 4 个专家各自的 ReAct 循环 + 综合 + 反思」，轻易产生几十次 LLM 调用，
而各层互不知情。本模块用 contextvar 提供一个贯穿嵌套层级的预算对象，
在 `core.llm.chat` 这一唯一出口统一计数与拦截。

用法：
    with run_budget():          # 在一次用户请求的最外层开启
        ...                     # 期间所有 chat() 调用共享同一预算

注意：`contextvars` 不会自动传播到新线程，向线程池提交任务时请用
`contextvars.copy_context().run(fn)`（见 `core.parallel.run_in_parallel`）。
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from dataclasses import dataclass, field

from config.settings import MAX_LLM_CALLS_PER_RUN, RUN_DEADLINE_SECONDS


class BudgetExceeded(RuntimeError):
    """预算耗尽（LLM 调用数超限或超过总时限）。"""


@dataclass
class RunBudget:
    max_llm_calls: int = MAX_LLM_CALLS_PER_RUN
    deadline_seconds: float = RUN_DEADLINE_SECONDS
    llm_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_llm_calls - self.llm_calls)

    def consume_llm_call(self) -> None:
        """登记一次 LLM 调用；超限则抛 BudgetExceeded。"""
        if self.deadline_seconds > 0 and self.elapsed > self.deadline_seconds:
            raise BudgetExceeded(
                f"已超过本次请求时限 {self.deadline_seconds:.0f}s（已用 {self.elapsed:.0f}s）"
            )
        if self.max_llm_calls > 0 and self.llm_calls >= self.max_llm_calls:
            raise BudgetExceeded(
                f"已达本次请求的 LLM 调用上限 {self.max_llm_calls} 次"
            )
        self.llm_calls += 1


_current: contextvars.ContextVar[RunBudget | None] = contextvars.ContextVar(
    "run_budget", default=None
)


def get_budget() -> RunBudget | None:
    return _current.get()


@contextlib.contextmanager
def run_budget(budget: RunBudget | None = None):
    """开启一次运行预算作用域；嵌套调用时复用外层预算。"""
    existing = _current.get()
    if existing is not None:
        yield existing
        return
    token = _current.set(budget or RunBudget())
    try:
        yield _current.get()
    finally:
        _current.reset(token)


def consume_llm_call() -> None:
    """由 `core.llm.chat` 调用：若当前有预算作用域则计数。"""
    budget = _current.get()
    if budget is not None:
        budget.consume_llm_call()
