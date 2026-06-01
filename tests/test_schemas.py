"""schemas 测试：事件 dump 契约 + Plan/ToolSpec 校验。"""

import pytest
from pydantic import ValidationError

from core.schemas import (
    Action,
    AnswerToken,
    Plan,
    ReflectionEvent,
    StepStart,
    ToolInfo,
    ToolSpec,
)


def test_event_dump_contract():
    assert StepStart(step=1, max_steps=10).model_dump() == {
        "type": "step_start", "step": 1, "max_steps": 10,
    }
    assert Action(tool="calc", args={"x": 1}, step=2).model_dump() == {
        "type": "action", "tool": "calc", "args": {"x": 1}, "step": 2,
    }
    assert AnswerToken(token="a", partial="ab").model_dump() == {
        "type": "answer_token", "token": "a", "partial": "ab",
    }
    assert ReflectionEvent(sufficient=False, critique="bad", step=3).model_dump() == {
        "type": "reflection", "sufficient": False, "critique": "bad", "step": 3,
    }


def test_plan_valid():
    plan = Plan.model_validate({
        "plan_summary": "x",
        "subtasks": [{"expert": "writing", "task": "t", "depends_on": [0]}],
    })
    assert plan.subtasks[0].expert.value == "writing"
    assert plan.subtasks[0].depends_on == [0]


def test_plan_invalid_expert():
    with pytest.raises(ValidationError):
        Plan.model_validate({"subtasks": [{"expert": "nope", "task": "t"}]})


def test_toolspec_requires_fields():
    with pytest.raises(ValidationError):
        ToolSpec(name="f", description="d")  # 缺 parameters/func


def test_toolinfo_dump():
    info = ToolInfo(name="f", description="d", category="c", version="1.0")
    assert info.model_dump() == {
        "name": "f", "description": "d", "category": "c", "version": "1.0",
    }
