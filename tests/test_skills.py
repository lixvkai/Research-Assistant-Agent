"""Skills 测试：执行引擎成功率统计 + get_skill 不自增 + 持久化。"""

import os

import pytest

import skills.skill_manager as sm
from skills.skill_executor import execute_skill


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "SKILLS_FILE", str(tmp_path / "skills.json"))
    return sm.SkillManager()


def test_get_skill_no_increment(mgr):
    s = mgr.get_skill("文献综述")
    assert s is not None
    assert s.usage_count == 0


def test_skill_backward_compatible_defaults():
    s = sm.Skill(name="t", description="d", steps=["a"], tools_needed=[], category="general")
    assert s.success_count == 0 and s.failure_count == 0
    assert s.success_rate == 0.0


def test_execute_success(mgr, fake_agent_cls):
    agent = fake_agent_cls([
        {"type": "step_start", "step": 1, "max_steps": 10},
        {"type": "answer", "content": "综述结果"},
    ])
    out = execute_skill("文献综述", "RAG 综述", manager=mgr, agent=agent)
    assert out == "综述结果"
    s = mgr.get_skill("文献综述")
    assert (s.usage_count, s.success_count, s.failure_count) == (1, 1, 0)
    assert s.success_rate == 1.0


def test_execute_failure(mgr, fake_agent_cls):
    agent = fake_agent_cls([{"type": "error", "content": "炸了"}])
    execute_skill("文献综述", "x", manager=mgr, agent=agent)
    s = mgr.get_skill("文献综述")
    assert s.failure_count == 1
    assert s.success_rate == 0.0


def test_execute_unknown_skill(mgr, fake_agent_cls):
    agent = fake_agent_cls([{"type": "answer", "content": "x"}])
    assert "未找到技能" in execute_skill("不存在", "x", manager=mgr, agent=agent)


def test_success_rate_persisted(mgr, fake_agent_cls):
    execute_skill("文献综述", "a", manager=mgr,
                  agent=fake_agent_cls([{"type": "answer", "content": "ok"}]))
    execute_skill("文献综述", "b", manager=mgr,
                  agent=fake_agent_cls([{"type": "error", "content": "x"}]))
    # 新建 manager 从磁盘回读
    mgr2 = sm.SkillManager()
    s = mgr2.get_skill("文献综述")
    assert s.usage_count == 2 and s.success_count == 1
    assert s.success_rate == 0.5
