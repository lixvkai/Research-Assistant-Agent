"""Gradio 事件处理函数测试。

重点是**返回值个数**必须和 `create_demo()` 里声明的 outputs 对得上 ——
会话 id 从模块级全局变量改成 `gr.State` 后，每个 handler 都多了一个进出参数，
这类接线错误只会在运行时暴露，所以用测试钉住。
"""

import pytest

pytest.importorskip("gradio")

import app as app_module


@pytest.fixture
def wired(make_service, monkeypatch):
    """把 app 模块里的服务单例换成桩件。"""
    service = make_service()
    monkeypatch.setattr(app_module, "get_agent_service", lambda: service)
    return service


# ── 发送消息 ──────────────────────────────────────────────────

def test_user_submit_returns_message_history_and_session(wired):
    msg, history, session_id = app_module.user_submit("你好", [], None)
    assert msg == ""
    assert history[-1] == {"role": "user", "content": "你好"}
    assert isinstance(session_id, int), "空会话时应自动新建并回传会话 id"


def test_user_submit_reuses_existing_session(wired):
    sid = wired.create_session("t")["id"]
    _, _, session_id = app_module.user_submit("你好", [], sid)
    assert session_id == sid


def test_user_submit_ignores_blank_message(wired):
    assert app_module.user_submit("   ", [], 7) == ("", [], 7)


def test_bot_respond_renders_answer_with_collapsed_trace(wired):
    sid = wired.create_session("t")["id"]
    frames = list(app_module.bot_respond([{"role": "user", "content": "问题"}], sid))

    assert frames, "应至少产出一帧"
    final = frames[-1][-1]["content"]
    assert "这是最终答案" in final
    assert "<details>" in final, "推理轨迹应折叠在答案下方"


def test_bot_respond_shows_intermediate_steps(wired):
    sid = wired.create_session("t")["id"]
    frames = list(app_module.bot_respond([{"role": "user", "content": "问题"}], sid))
    rendered = "".join(f[-1]["content"] for f in frames)
    assert "search_arxiv" in rendered, "工具调用应在流式过程中显示出来"


def test_bot_respond_requires_trailing_user_message(wired):
    frames = list(app_module.bot_respond([{"role": "assistant", "content": "x"}], 1))
    assert frames == [[{"role": "assistant", "content": "x"}]]


def test_bot_respond_reports_busy_session(wired):
    sid = wired.create_session("t")["id"]
    guard = wired.acquire_session(sid)
    try:
        frames = list(app_module.bot_respond([{"role": "user", "content": "x"}], sid))
    finally:
        guard.release()
    assert "还在处理" in frames[-1][-1]["content"]


def test_bot_respond_releases_session_lock(wired):
    sid = wired.create_session("t")["id"]
    list(app_module.bot_respond([{"role": "user", "content": "问题"}], sid))
    assert not wired.is_busy(sid)


def test_bot_respond_releases_lock_on_failure(wired, monkeypatch):
    sid = wired.create_session("t")["id"]

    def boom(*a, **kw):
        raise RuntimeError("炸了")

    monkeypatch.setattr(wired._agent, "run_iter", boom)
    frames = list(app_module.bot_respond([{"role": "user", "content": "x"}], sid))
    assert "发生错误" in frames[-1][-1]["content"]
    assert not wired.is_busy(sid), "异常路径也必须释放会话锁"


# ── 新建 / 切换 / 删除会话 ────────────────────────────────────

def test_handle_reset_clears_chat_and_session(wired):
    sid = wired.create_session("t")["id"]
    chat, msg, session_id = app_module.handle_reset(sid)
    assert chat == []
    assert msg == ""
    assert session_id is None


def test_handle_reset_without_session_is_noop(wired):
    assert app_module.handle_reset(None) == ([], "", None)


def test_load_session_returns_messages_and_session(wired):
    sid = wired.create_session("t")["id"]
    list(wired.stream_chat(sid, "之前的问题"))

    messages, _html, session_id = app_module.load_session(f"{sid}|1700000000")
    assert session_id == sid
    assert messages[0]["content"] == "之前的问题"


def test_load_session_handles_missing_session(wired):
    messages, _html, session_id = app_module.load_session("99999|1")
    assert messages == []
    assert session_id is None


def test_load_session_handles_garbage_input(wired):
    messages, _html, session_id = app_module.load_session("")
    assert messages == []
    assert session_id is None


def test_delete_active_session_clears_ui(wired):
    sid = wired.create_session("t")["id"]
    _html, chat, session_id = app_module.delete_session(
        f"{sid}|1", sid, [{"role": "user", "content": "x"}]
    )
    assert chat == []
    assert session_id is None


def test_delete_other_session_keeps_ui(wired):
    other = wired.create_session("其他")["id"]
    current = wired.create_session("当前")["id"]
    history = [{"role": "user", "content": "x"}]

    _html, chat, session_id = app_module.delete_session(f"{other}|1", current, history)
    assert chat == history
    assert session_id == current
