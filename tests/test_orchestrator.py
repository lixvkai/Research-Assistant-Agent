"""Orchestrator 规划解析与兜底测试（不触发 LLM）。"""

import pytest

from agents.orchestrator import Orchestrator


@pytest.fixture
def orch():
    return Orchestrator()


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
