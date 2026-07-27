"""上下文窗口裁剪 + 会话隔离 / 历史回灌测试。"""

import core.react_agent as ra
from core.react_agent import trim_messages


def test_trim_keeps_recent():
    msgs = [{"role": "user", "content": str(i)} for i in range(10)]
    kept, overflow = trim_messages(msgs, limit=4, trigger=4)
    assert len(kept) == 4
    assert len(overflow) == 6
    assert kept[0]["content"] == "6"


def test_trim_no_overflow():
    msgs = [{"role": "user", "content": "a"}]
    kept, overflow = trim_messages(msgs, limit=5, trigger=5)
    assert kept == msgs
    assert overflow == []


def test_trim_never_starts_with_orphan_tool_message():
    """切点落在 tool 消息上时必须向后推进，否则请求非法。"""
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "r"},
        {"role": "tool", "tool_call_id": "2", "content": "r2"},
        {"role": "assistant", "content": "done"},
    ]
    kept, overflow = trim_messages(msgs, limit=3, trigger=3)
    assert kept[0]["role"] != "tool"
    assert kept[0]["content"] == "done"
    assert len(overflow) == 4


# ── 高低水位（滞回）─────────────────────────────────────────────

def test_trim_waits_for_trigger():
    """超过低水位但没到触发线时不应裁剪 —— 这正是缓冲区的意义。"""
    msgs = [{"role": "user", "content": str(i)} for i in range(30)]
    kept, overflow = trim_messages(msgs, limit=25, trigger=40)
    assert overflow == []
    assert len(kept) == 30


def test_trim_cuts_down_to_limit_not_trigger():
    """一旦触发，要裁到低水位而不是停在触发线，否则下一轮立刻又越界。"""
    msgs = [{"role": "user", "content": str(i)} for i in range(41)]
    kept, overflow = trim_messages(msgs, limit=25, trigger=40)
    assert len(kept) == 25
    assert len(overflow) == 16


def test_trim_amortizes_across_turns():
    """压缩应被摊薄到多轮：裁剪后要能再撑几轮才触发下一次。"""
    msgs = [{"role": "user", "content": str(i)} for i in range(41)]
    kept, _ = trim_messages(msgs, limit=25, trigger=40)

    compressions = 0
    for _ in range(3):  # 再来 3 轮，每轮 4 条消息
        kept.extend({"role": "user", "content": "x"} for _ in range(4))
        kept, overflow = trim_messages(kept, limit=25, trigger=40)
        if overflow:
            compressions += 1
    assert compressions == 0, "25→40 的缓冲应能覆盖至少 3 轮普通对话"


def test_trigger_below_limit_degrades_safely():
    """配置反了也不能每轮空转：触发线低于低水位时退化为触发即裁到低水位。"""
    msgs = [{"role": "user", "content": str(i)} for i in range(30)]
    kept, overflow = trim_messages(msgs, limit=25, trigger=10)
    assert len(kept) == 25
    assert len(overflow) == 5


def test_overflow_goes_to_memory(monkeypatch, mkresp):
    """溢出消息应交给短期记忆压缩，而不是被静默丢弃。"""
    absorbed = {"msgs": []}

    class FakeMemory:
        def get_context_for_prompt(self, q="", session_id=None):
            return ""

        def absorb_overflow(self, messages, session_id=None):
            absorbed["msgs"].extend(messages)

        def record_interaction(self, *a, **kw):
            pass

        def reset_session(self, session_id=None):
            pass

    monkeypatch.setattr(ra, "MAX_CONTEXT_MESSAGES", 2)
    monkeypatch.setattr(ra, "CONTEXT_TRIM_TRIGGER", 2)
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, **kw: mkresp("ok"))

    agent = ra.ReActAgent(enable_reflection=False, memory_manager=FakeMemory())
    for i in range(4):
        list(agent.run_iter(f"第{i}轮"))

    assert absorbed["msgs"], "超出窗口的消息应被短期记忆吸收"


def test_summary_is_visible_in_the_same_turn(monkeypatch, mkresp):
    """回归：压缩当轮就该看到新摘要。

    memory_context 是进图前取的，而压缩发生在图内的 prepare。若不刷新，
    本轮被移出窗口的消息既不在 messages 里也不在摘要里，会出现一轮的空窗。
    """
    sent = []

    def fake_chat(messages, tools=None, **kw):
        sent.append(messages)
        return mkresp("ok")

    class FakeMemory:
        def __init__(self):
            self.summary = ""

        def get_context_for_prompt(self, q="", session_id=None):
            return f"[对话历史摘要]\n{self.summary}" if self.summary else ""

        def absorb_overflow(self, messages, session_id=None):
            self.summary = "早期对话讲的是蛋白质折叠"

        def record_interaction(self, *a, **kw):
            pass

        def reset_session(self, session_id=None):
            pass

    monkeypatch.setattr(ra, "MAX_CONTEXT_MESSAGES", 2)
    monkeypatch.setattr(ra, "CONTEXT_TRIM_TRIGGER", 2)
    monkeypatch.setattr(ra, "chat", fake_chat)

    agent = ra.ReActAgent(enable_reflection=False, memory_manager=FakeMemory())
    list(agent.run_iter("第一轮", session_id="G"))
    sent.clear()
    list(agent.run_iter("第二轮", session_id="G"))  # 这一轮触发压缩

    joined = " ".join(m.get("content") or "" for m in sent[0])
    assert "蛋白质折叠" in joined, "压缩当轮就应能看到刚生成的摘要"


def test_memory_context_is_not_persisted_into_history(monkeypatch, mkresp):
    """回归：记忆上下文只在调用 LLM 时临时拼装，不能写进会话历史。

    修复前它被拼进 user message 并落库，导致第 N 轮堆叠 N 段过期的记忆快照。
    """
    sent = []

    def fake_chat(messages, tools=None, **kw):
        sent.append(messages)
        return mkresp("ok")

    class FakeMemory:
        def get_context_for_prompt(self, q="", session_id=None):
            return "[用户偏好]\n- 用户偏好中文"

        def absorb_overflow(self, messages, session_id=None):
            pass

        def record_interaction(self, *a, **kw):
            pass

        def reset_session(self, session_id=None):
            pass

    monkeypatch.setattr(ra, "chat", fake_chat)
    agent = ra.ReActAgent(enable_reflection=False, memory_manager=FakeMemory())

    list(agent.run_iter("第一个问题", session_id="M"))
    list(agent.run_iter("第二个问题", session_id="M"))

    # LLM 能看到记忆上下文
    assert any("用户偏好中文" in (m.get("content") or "")
               for m in sent[-1])
    # 但历史里只有原始问题，没有记忆上下文的痕迹
    stored = agent.get_messages("M")
    assert [m["content"] for m in stored if m["role"] == "user"] == ["第一个问题", "第二个问题"]
    assert not any("用户偏好" in (m.get("content") or "") for m in stored)


def test_sessions_are_isolated(monkeypatch, mkresp):
    monkeypatch.setattr(ra, "chat", lambda messages, tools=None, **kw: mkresp("ok"))
    agent = ra.ReActAgent(enable_reflection=False)

    list(agent.run_iter("会话A的问题", session_id="A"))
    list(agent.run_iter("会话B的问题", session_id="B"))

    a_text = " ".join(m.get("content") or "" for m in agent.get_messages("A"))
    b_text = " ".join(m.get("content") or "" for m in agent.get_messages("B"))
    assert "会话A的问题" in a_text and "会话B的问题" not in a_text
    assert "会话B的问题" in b_text and "会话A的问题" not in b_text


def test_load_history_rehydrates_session(monkeypatch, mkresp):
    """切换回历史会话后，Agent 应能看到之前的对话内容。"""
    seen = {}

    def fake_chat(messages, tools=None, **kw):
        seen["messages"] = messages
        return mkresp("ok")

    monkeypatch.setattr(ra, "chat", fake_chat)
    agent = ra.ReActAgent(enable_reflection=False)
    agent.load_history(
        [
            {"role": "user", "content": "我在研究蛋白质折叠"},
            {"role": "assistant", "content": "好的"},
        ],
        session_id="S1",
    )

    list(agent.run_iter("继续", session_id="S1"))
    joined = " ".join(m.get("content") or "" for m in seen["messages"])
    assert "蛋白质折叠" in joined


def test_load_history_skips_empty_and_non_dialog():
    agent = ra.ReActAgent(enable_reflection=False)
    agent.load_history(
        [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "kept"},
        ],
        session_id="S2",
    )
    msgs = agent.get_messages("S2")
    assert len(msgs) == 1
    assert msgs[0]["content"] == "kept"


def test_tool_registry_dedups_schemas():
    """重复绑定同名工具不应产生重复的 function schema。"""
    agent = ra.ReActAgent(enable_reflection=False)
    for _ in range(3):
        agent.register_tool("calc", "计算", {"type": "object"}, lambda: 1)
    schemas = agent.tool_registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "calc"
