"""Orchestrator 规划解析与兜底测试（不触发 LLM）。"""

from types import SimpleNamespace

import pytest

from agents.orchestrator import Orchestrator
from core.schemas import Reflection


@pytest.fixture
def orch():
    return Orchestrator()


def _resp(content: str):
    """最小化的 LLM 响应桩。"""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_parse_good_json(orch):
    raw = '{"plan_summary":"s","subtasks":[{"expert":"writing","task":"t","depends_on":[0]}]}'
    plan = orch._parse_plan(raw, "orig")
    assert plan.plan_summary == "s"
    assert plan.subtasks[0].expert.value == "writing"
    assert plan.subtasks[0].depends_on == [0]


def test_parse_fenced_json(orch):
    fenced = '思路如下：\n```json\n{"plan_summary":"f","subtasks":[{"expert":"review","task":"t"}]}\n```'
    plan = orch._parse_plan(fenced, "orig")
    assert plan.plan_summary == "f"
    assert plan.subtasks[0].expert.value == "review"


def test_parse_invalid_expert_falls_back(orch):
    plan = orch._parse_plan('{"subtasks":[{"expert":"nope","task":"t"}]}', "ORIG")
    assert plan.subtasks[0].expert.value == "literature"
    assert plan.subtasks[0].task == "ORIG"


def test_parse_non_json_falls_back(orch):
    plan = orch._parse_plan("抱歉我无法规划", "ORIG")
    assert plan.subtasks[0].expert.value == "literature"
    assert plan.subtasks[0].task == "ORIG"


def test_parse_empty_subtasks_falls_back(orch):
    plan = orch._parse_plan('{"plan_summary":"e","subtasks":[]}', "ORIG")
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].task == "ORIG"


def test_extract_json_none_when_absent(orch):
    assert orch._extract_json("no json here") is None


# ── 质量门由 ReviewAgent 负责（不再有第二套审查逻辑） ──────────────

def test_reflect_delegates_to_review_agent(orch, monkeypatch):
    """reflect 节点必须走 ReviewAgent，而不是自己另调一次 LLM。"""
    from agents.specialists import ReviewAgent

    seen = {}

    def fake_review_draft(self, task, draft):
        seen["self"] = self
        seen["task"], seen["draft"] = task, draft
        return Reflection(sufficient=False, critique="不足")

    monkeypatch.setattr(ReviewAgent, "review_draft", fake_review_draft)
    monkeypatch.setattr(orch, "_get_mcp_server", lambda: None)

    verdict = orch._reflect("原任务", "综合稿")

    assert isinstance(seen["self"], ReviewAgent)
    assert (seen["task"], seen["draft"]) == ("原任务", "综合稿")
    assert verdict.sufficient is False


def test_review_draft_parses_verdict(monkeypatch):
    import agents.specialists as sp

    monkeypatch.setattr(
        sp, "chat",
        lambda **kw: _resp('前言 {"sufficient": false, "critique": "缺少实验"} 后记'),
    )
    verdict = sp.ReviewAgent().review_draft("t", "d")
    assert verdict.sufficient is False
    assert verdict.critique == "缺少实验"


def test_review_draft_defaults_to_pass_on_garbage(monkeypatch):
    """审查环节自身不能成为死循环来源：解析不出来就放行。"""
    import agents.specialists as sp

    monkeypatch.setattr(sp, "chat", lambda **kw: _resp("模型今天不想输出 JSON"))
    assert sp.ReviewAgent().review_draft("t", "d").sufficient is True


def test_review_draft_survives_llm_failure(monkeypatch):
    import agents.specialists as sp

    def boom(**kw):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr(sp, "chat", boom)
    assert sp.ReviewAgent().review_draft("t", "d").sufficient is True
