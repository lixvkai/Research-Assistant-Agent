"""ReActAgent（LangGraph）测试：工具流程 / 反思循环 / 步数封顶 / 多轮记忆。"""

import core.react_agent as ra


def test_tool_then_answer(monkeypatch, mkresp, mktool):
    calls = {"n": 0}

    def fake_chat(messages, tools=None, temperature=0.7, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return mkresp("先算一下", tool_calls=[mktool("calc", {"x": 2})])
        return mkresp("答案是 4")

    monkeypatch.setattr(ra, "chat", fake_chat)
    agent = ra.ReActAgent(enable_reflection=False)
    agent.register_tool("calc", "计算", {"type": "object"}, lambda x: x + x)

    events = [e["type"] for e in agent.run_iter("算 2+2")]
    assert events == ["step_start", "thought", "action", "observation", "step_start", "answer"]


def test_expert_no_reflection_single_call(monkeypatch, mkresp):
    n = {"c": 0}

    def fake_chat(messages, tools=None, temperature=0.7, **kw):
        n["c"] += 1
        return mkresp("直接回答")

    monkeypatch.setattr(ra, "chat", fake_chat)
    agent = ra.ReActAgent(enable_reflection=False)
    events = [e["type"] for e in agent.run_iter("hi")]
    assert "reflection" not in events
    assert events[-1] == "answer"
    assert n["c"] == 1


def test_reflection_revises_once(monkeypatch, mkresp):
    def fake_chat(messages, tools=None, temperature=0.7, **kw):
        sys = messages[0].get("content", "")
        if "审查" in sys:
            # 第一次不通过，第二次通过
            fake_chat.reflect += 1
            if fake_chat.reflect == 1:
                return mkresp('{"sufficient": false, "critique": "补充细节"}')
            return mkresp('{"sufficient": true, "critique": ""}')
        fake_chat.agent += 1
        return mkresp("初版答案" if fake_chat.agent == 1 else "修订版答案，更详细")

    fake_chat.reflect = 0
    fake_chat.agent = 0
    monkeypatch.setattr(ra, "chat", fake_chat)

    agent = ra.ReActAgent(enable_reflection=True)
    events = list(agent.run_iter("问题"))
    types_ = [e["type"] for e in events]
    assert "reflection" in types_
    final = [e for e in events if e["type"] == "answer"][0]["content"]
    assert final == "修订版答案，更详细"


def test_step_cap_graceful(monkeypatch, mkresp, mktool):
    def fake_chat(messages, tools=None, temperature=0.7, **kw):
        return mkresp("继续", tool_calls=[mktool("loop", {})])

    monkeypatch.setattr(ra, "chat", fake_chat)
    agent = ra.ReActAgent(enable_reflection=False)
    agent.register_tool("loop", "死循环", {"type": "object"}, lambda: "again")

    events = list(agent.run_iter("无限循环"))
    step_starts = sum(1 for e in events if e["type"] == "step_start")
    assert step_starts <= ra.MAX_REACT_STEPS
    assert events[-1]["type"] == "answer"
    assert "最大推理步数" in events[-1]["content"]


def test_multiturn_history(monkeypatch, mkresp):
    def fake_chat(messages, tools=None, temperature=0.7, **kw):
        return mkresp("回答")

    monkeypatch.setattr(ra, "chat", fake_chat)
    agent = ra.ReActAgent(enable_reflection=False)
    list(agent.run_iter("第一轮"))
    list(agent.run_iter("第二轮"))
    roles = [m["role"] for m in agent.conversation_history]
    assert roles.count("user") == 2
    assert roles.count("assistant") == 2

    agent.reset()
    assert len(agent.conversation_history) == 1  # 仅剩 system
