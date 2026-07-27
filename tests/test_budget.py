"""运行预算测试：LLM 调用上限 / 时限 / 嵌套复用 / 线程传播。"""

import time

import pytest

from core.budget import BudgetExceeded, RunBudget, consume_llm_call, get_budget, run_budget
from core.parallel import run_in_parallel


def test_no_budget_scope_is_noop():
    assert get_budget() is None
    consume_llm_call()  # 不应抛异常


def test_call_limit_enforced():
    with run_budget(RunBudget(max_llm_calls=2, deadline_seconds=0)):
        consume_llm_call()
        consume_llm_call()
        with pytest.raises(BudgetExceeded, match="调用上限"):
            consume_llm_call()


def test_deadline_enforced():
    budget = RunBudget(max_llm_calls=100, deadline_seconds=0.01)
    with run_budget(budget):
        time.sleep(0.02)
        with pytest.raises(BudgetExceeded, match="时限"):
            consume_llm_call()


def test_nested_scope_shares_same_budget():
    """嵌套 Agent（如 multi_agent_collaborate）必须复用外层预算，而不是各自重置。"""
    with run_budget(RunBudget(max_llm_calls=3, deadline_seconds=0)) as outer:
        consume_llm_call()
        with run_budget() as inner:
            assert inner is outer
            consume_llm_call()
        assert outer.llm_calls == 2
        assert outer.remaining_calls == 1


def test_budget_propagates_into_worker_threads():
    """contextvars 默认不跨线程，run_in_parallel 必须显式复制上下文。"""
    with run_budget(RunBudget(max_llm_calls=10, deadline_seconds=0)) as budget:
        found = run_in_parallel(lambda _: get_budget() is budget, [1, 2, 3], max_workers=3)
    assert all(found)


def test_run_in_parallel_preserves_order():
    assert run_in_parallel(lambda x: x * 2, [1, 2, 3, 4], max_workers=3) == [2, 4, 6, 8]


def test_run_in_parallel_empty():
    assert run_in_parallel(lambda x: x, []) == []
