"""Orchestrator 执行层测试：拓扑分层 / 并行 / 依赖上下文 / 环兜底。"""

import agents.orchestrator as orch_mod
from agents.orchestrator import Orchestrator
from core.schemas import Plan, SubTask


def _plan(*specs) -> Plan:
    return Plan(
        plan_summary="p",
        subtasks=[SubTask(expert=e, task=t, depends_on=list(d)) for e, t, d in specs],
    )


def test_topo_layers_groups_independent_tasks():
    plan = _plan(
        ("literature", "t0", []),
        ("data_analysis", "t1", []),
        ("writing", "t2", [0, 1]),
    )
    layers = Orchestrator._topo_layers(plan.subtasks)
    assert layers[0] == [0, 1]      # 无依赖 → 同层并行
    assert layers[1] == [2]


def test_topo_layers_handles_reverse_declared_dependency():
    """LLM 可能给出「子任务0 依赖 子任务1」这种逆序依赖，必须按依赖而非下标执行。"""
    plan = _plan(
        ("writing", "needs 1", [1]),
        ("literature", "independent", []),
    )
    layers = Orchestrator._topo_layers(plan.subtasks)
    assert layers[0] == [1]
    assert layers[1] == [0]


def test_topo_layers_ignores_invalid_deps():
    plan = _plan(("literature", "t0", [5, 0, -1]))
    assert Orchestrator._topo_layers(plan.subtasks) == [[0]]


def test_topo_layers_cycle_falls_back_without_dropping_tasks():
    plan = _plan(
        ("literature", "a", [1]),
        ("writing", "b", [0]),
    )
    layers = Orchestrator._topo_layers(plan.subtasks)
    assert sorted(i for layer in layers for i in layer) == [0, 1]


def test_execute_node_passes_upstream_context(monkeypatch):
    calls = []

    class FakeExpert:
        def __init__(self, name):
            self.name = name

        def run(self, task, context=""):
            calls.append((task, context))
            return f"{self.name}:{task}"

    orch = Orchestrator()
    monkeypatch.setattr(orch, "_get_expert", lambda name: FakeExpert(name))

    plan = _plan(
        ("literature", "找论文", []),
        ("writing", "写综述", [0]),
    )
    out = orch._execute_node({"plan": plan})["results"]

    assert out[0] == "literature:找论文"
    assert out[1] == "writing:写综述"
    downstream_context = dict(calls)["写综述"]
    assert "literature:找论文" in downstream_context


def test_execute_node_survives_expert_failure(monkeypatch):
    class BoomExpert:
        def run(self, task, context=""):
            raise RuntimeError("专家挂了")

    orch = Orchestrator()
    monkeypatch.setattr(orch, "_get_expert", lambda name: BoomExpert())

    results = orch._execute_node({"plan": _plan(("literature", "t", []))})["results"]
    assert "执行出错" in results[0]


def test_execute_node_runs_same_layer_in_parallel(monkeypatch):
    """同层子任务应并行：串行执行会让总耗时接近 N×单任务耗时。"""
    import time

    class SlowExpert:
        def run(self, task, context=""):
            time.sleep(0.15)
            return task

    orch = Orchestrator()
    monkeypatch.setattr(orch, "_get_expert", lambda name: SlowExpert())
    monkeypatch.setattr(orch_mod, "ORCHESTRATOR_MAX_WORKERS", 3)

    plan = _plan(
        ("literature", "a", []),
        ("data_analysis", "b", []),
        ("review", "c", []),
    )
    started = time.monotonic()
    results = orch._execute_node({"plan": plan})["results"]
    elapsed = time.monotonic() - started

    assert len(results) == 3
    assert elapsed < 0.4, f"同层未并行，耗时 {elapsed:.2f}s"
