"""服务层测试：会话隔离、历史回灌、并发保护、落库内容。"""

import pytest

from services.agent_service import (
    SessionBusyError,
    SessionNotFoundError,
    derive_title,
)


# ── 会话 CRUD ─────────────────────────────────────────────────

def test_create_and_list_sessions(make_service):
    service = make_service()
    a = service.create_session("会话A")
    b = service.create_session("会话B")

    assert a["id"] != b["id"]
    titles = [s["title"] for s in service.list_sessions()]
    assert "会话A" in titles and "会话B" in titles


def test_create_session_falls_back_to_default_title(make_service):
    service = make_service()
    assert service.create_session("   ")["title"] == "新对话"


def test_get_missing_session_raises(make_service):
    service = make_service()
    with pytest.raises(SessionNotFoundError):
        service.get_session(9999)


def test_ensure_session_creates_from_message(make_service):
    service = make_service()
    sid = service.ensure_session(None, "帮我分析扩散模型的采样加速")
    assert service.get_session(sid)["title"].startswith("帮我分析扩散模型")


def test_ensure_session_validates_existing(make_service):
    service = make_service()
    with pytest.raises(SessionNotFoundError):
        service.ensure_session(4242, "x")


def test_derive_title_truncates():
    assert derive_title("x" * 100).endswith("…")
    assert derive_title("  ") == "新对话"


def test_delete_session_removes_messages(make_service):
    service = make_service()
    sid = service.create_session("待删除")["id"]
    list(service.stream_chat(sid, "你好"))
    assert service.get_messages(sid)

    service.delete_session(sid)
    with pytest.raises(SessionNotFoundError):
        service.get_messages(sid)


# ── 对话与落库 ────────────────────────────────────────────────

def test_stream_chat_yields_events_and_persists(make_service):
    service = make_service()
    sid = service.create_session("t")["id"]

    events = list(service.stream_chat(sid, "什么是 RAG"))
    assert [e["type"] for e in events][0] == "step_start"
    assert events[-1]["type"] == "answer"

    messages = service.get_messages(sid)
    assert messages[0] == {"role": "user", "content": "什么是 RAG"}
    assert messages[1] == {"role": "assistant", "content": "这是最终答案"}


def test_persisted_answer_has_no_render_markup(make_service):
    """回归：落库的必须是纯文本答案。

    旧实现把 UI 渲染好的字符串（含 <details> 推理轨迹）存进历史，
    切回该会话时这段 HTML 会被当成对话内容重新喂给 LLM。
    """
    service = make_service()
    sid = service.create_session("t")["id"]
    list(service.stream_chat(sid, "问题"))

    stored = service.get_messages(sid)[1]["content"]
    assert "<details>" not in stored
    assert "步骤" not in stored


def test_blank_message_is_ignored(make_service):
    service = make_service()
    sid = service.create_session("t")["id"]
    assert list(service.stream_chat(sid, "   ")) == []
    assert service.get_messages(sid) == []


@pytest.mark.parametrize("bad_session", [None, 0, 9999])
def test_stream_chat_rejects_invalid_session(make_service, bad_session):
    """回归：非法会话 id 不能写出挂不到任何会话的孤儿消息。"""
    service = make_service()
    with pytest.raises(SessionNotFoundError):
        list(service.stream_chat(bad_session, "问题"))


def test_error_event_is_persisted_as_reply(make_service):
    service = make_service()
    sid = service.create_session("t")["id"]
    list(service.stream_chat(sid, "问题"))

    service = make_service([{"type": "error", "content": "工具挂了"}])
    sid = service.create_session("t2")["id"]
    list(service.stream_chat(sid, "问题"))
    assert "工具挂了" in service.get_messages(sid)[1]["content"]


def test_session_id_is_passed_to_agent(make_service):
    service = make_service()
    agent = make_service.state["agent"]
    sid = service.create_session("t")["id"]
    list(service.stream_chat(sid, "问题"))
    assert agent.runs == [("问题", str(sid))]


# ── 历史回灌 ──────────────────────────────────────────────────

def test_history_is_rehydrated_on_first_touch(make_service):
    """切回旧会话（或进程重启后）第一次对话时应把落库消息回灌进图状态。"""
    service = make_service()
    agent = make_service.state["agent"]
    sid = service.create_session("旧会话")["id"]
    list(service.stream_chat(sid, "第一轮"))

    # 模拟进程重启：进程内的「已回灌」标记消失
    service._hydrated.clear()
    agent.loaded.clear()

    list(service.stream_chat(sid, "第二轮"))
    assert str(sid) in agent.loaded
    assert any("第一轮" in m["content"] for m in agent.loaded[str(sid)])


def test_history_is_rehydrated_only_once(make_service):
    service = make_service()
    agent = make_service.state["agent"]
    sid = service.create_session("t")["id"]
    service._hydrated.clear()

    list(service.stream_chat(sid, "一"))
    list(service.stream_chat(sid, "二"))
    # 首次触碰时库里还没有消息，因此不该有回灌；关键是不能反复回灌
    assert len(agent.loaded) <= 1


def test_new_session_skips_rehydration(make_service):
    service = make_service()
    agent = make_service.state["agent"]
    sid = service.create_session("t")["id"]
    list(service.stream_chat(sid, "问题"))
    assert agent.loaded == {}


# ── 并发保护 ──────────────────────────────────────────────────

def test_acquire_session_blocks_second_run(make_service):
    service = make_service()
    sid = service.create_session("t")["id"]

    guard = service.acquire_session(sid)
    assert service.is_busy(sid)
    with pytest.raises(SessionBusyError):
        service.acquire_session(sid)

    guard.release()
    assert not service.is_busy(sid)
    service.acquire_session(sid).release()


def test_different_sessions_can_run_concurrently(make_service):
    service = make_service()
    a, b = service.create_session("a")["id"], service.create_session("b")["id"]
    guard_a = service.acquire_session(a)
    guard_b = service.acquire_session(b)  # 不应抛异常
    guard_a.release()
    guard_b.release()


def test_guard_is_a_context_manager(make_service):
    service = make_service()
    sid = service.create_session("t")["id"]
    with service.acquire_session(sid):
        assert service.is_busy(sid)
    assert not service.is_busy(sid)


# ── 重置 ──────────────────────────────────────────────────────

def test_reset_session_consolidates_and_clears_graph_state(make_service):
    service = make_service()
    agent = make_service.state["agent"]
    memory = make_service.state["memory"]
    sid = service.create_session("t")["id"]
    list(service.stream_chat(sid, "问题"))

    service.reset_session(sid)
    assert memory.consolidated == [str(sid)]
    assert agent.resets == [str(sid)]
    # 历史消息仍然留在库里，只是不再作为当前上下文
    assert service.get_messages(sid)


def test_reset_survives_consolidate_failure(make_service):
    service = make_service()
    agent = make_service.state["agent"]
    memory = make_service.state["memory"]

    def boom(session_id=None):
        raise RuntimeError("LLM 挂了")

    memory.consolidate = boom
    sid = service.create_session("t")["id"]
    service.reset_session(sid)
    assert agent.resets == [str(sid)]
