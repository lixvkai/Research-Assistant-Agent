"""FastAPI 接口测试：会话 CRUD、SSE 流式对话、并发保护。"""

import json

import pytest

pytest.importorskip("fastapi")


# ── 辅助：解析 SSE 响应 ────────────────────────────────────────

def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 [(event, data), ...]。"""
    frames = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None:
            frames.append((event, data))
    return frames


# ── 元信息 ────────────────────────────────────────────────────

def test_health(api_client):
    client, _ = api_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_openapi_schema_is_served(api_client):
    client, _ = api_client()
    assert client.get("/openapi.json").status_code == 200


# ── 会话 ──────────────────────────────────────────────────────

def test_create_session(api_client):
    client, _ = api_client()
    resp = client.post("/api/sessions", json={"title": "我的会话"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "我的会话"
    assert isinstance(body["id"], int)


def test_create_session_without_body(api_client):
    client, _ = api_client()
    resp = client.post("/api/sessions")
    assert resp.status_code == 201
    assert resp.json()["title"] == "新对话"


def test_list_sessions(api_client):
    client, _ = api_client()
    client.post("/api/sessions", json={"title": "一"})
    client.post("/api/sessions", json={"title": "二"})
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert {s["title"] for s in resp.json()} == {"一", "二"}


def test_get_session_with_messages(api_client):
    client, _ = api_client()
    sid = client.post("/api/sessions").json()["id"]
    client.post("/api/chat", json={"session_id": sid, "message": "你好"})

    body = client.get(f"/api/sessions/{sid}").json()
    assert body["id"] == sid
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]


def test_missing_session_returns_404(api_client):
    client, _ = api_client()
    assert client.get("/api/sessions/98765").status_code == 404
    assert client.get("/api/sessions/98765/messages").status_code == 404
    assert client.delete("/api/sessions/98765").status_code == 404
    assert client.post("/api/sessions/98765/reset").status_code == 404


def test_delete_session(api_client):
    client, _ = api_client()
    sid = client.post("/api/sessions").json()["id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_reset_session(api_client):
    client, service = api_client()
    sid = client.post("/api/sessions").json()["id"]
    client.post("/api/chat", json={"session_id": sid, "message": "你好"})

    assert client.post(f"/api/sessions/{sid}/reset").status_code == 204
    # 历史消息保留，只是不再作为 Agent 的当前上下文
    assert client.get(f"/api/sessions/{sid}/messages").json()


def test_list_sessions_rejects_bad_limit(api_client):
    client, _ = api_client()
    assert client.get("/api/sessions", params={"limit": 0}).status_code == 422
    assert client.get("/api/sessions", params={"limit": 999}).status_code == 422


# ── SSE 流式对话 ──────────────────────────────────────────────

def test_chat_stream_event_sequence(api_client):
    client, _ = api_client()
    sid = client.post("/api/sessions").json()["id"]

    resp = client.post("/api/chat/stream", json={"session_id": sid, "message": "什么是 RAG"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    frames = parse_sse(resp.text)
    names = [name for name, _ in frames]
    assert names[0] == "session", "首个事件应告知客户端会话 id"
    assert names[-1] == "done", "流结束必须有 done 事件"
    assert names[1:-1] == ["step_start", "thought", "action", "observation", "answer"]

    assert frames[0][1]["session_id"] == sid
    answer = next(data for name, data in frames if name == "answer")
    assert answer["content"] == "这是最终答案"


def test_chat_stream_creates_session_when_omitted(api_client):
    client, _ = api_client()
    resp = client.post("/api/chat/stream", json={"message": "帮我搜索扩散模型的论文"})
    assert resp.status_code == 200

    frames = parse_sse(resp.text)
    sid = frames[0][1]["session_id"]
    assert client.get(f"/api/sessions/{sid}").json()["title"].startswith("帮我搜索扩散模型")


def test_chat_stream_unknown_session_returns_404(api_client):
    client, _ = api_client()
    resp = client.post("/api/chat/stream", json={"session_id": 5555, "message": "hi"})
    assert resp.status_code == 404


def test_chat_stream_rejects_blank_message(api_client):
    client, _ = api_client()
    assert client.post("/api/chat/stream", json={"message": ""}).status_code == 422
    assert client.post("/api/chat/stream", json={"message": "   "}).status_code == 422


def test_chat_stream_reports_agent_error(api_client):
    """Agent 抛异常时也要以 error + done 收尾，而不是把连接晾着。"""
    def exploding():
        yield {"type": "step_start", "step": 1, "max_steps": 10}
        raise RuntimeError("模型不可用")

    client, service = api_client()
    service._agent.run_iter = lambda *a, **kw: exploding()

    resp = client.post("/api/chat/stream", json={"message": "hi"})
    frames = parse_sse(resp.text)
    names = [n for n, _ in frames]
    assert "error" in names
    assert names[-1] == "done"
    assert "模型不可用" in next(d for n, d in frames if n == "error")["content"]


def test_chat_stream_conflicts_when_session_busy(api_client):
    client, service = api_client()
    sid = client.post("/api/sessions").json()["id"]

    guard = service.acquire_session(sid)
    try:
        resp = client.post("/api/chat/stream", json={"session_id": sid, "message": "hi"})
        assert resp.status_code == 409
    finally:
        guard.release()


def test_session_lock_is_released_after_stream(api_client):
    client, service = api_client()
    sid = client.post("/api/sessions").json()["id"]
    client.post("/api/chat/stream", json={"session_id": sid, "message": "hi"})
    assert not service.is_busy(sid)
    # 释放干净后应能继续对话
    assert client.post("/api/chat/stream", json={"session_id": sid, "message": "再来"}).status_code == 200


# ── 非流式对话 ────────────────────────────────────────────────

def test_chat_returns_final_answer(api_client):
    client, _ = api_client()
    resp = client.post("/api/chat", json={"message": "什么是 RAG"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "这是最终答案"
    assert body["steps"] == 1
    assert isinstance(body["session_id"], int)


def test_chat_releases_lock_on_failure(api_client):
    client, service = api_client()
    sid = client.post("/api/sessions").json()["id"]

    def boom(*a, **kw):
        raise RuntimeError("炸了")

    service._agent.run_iter = boom
    with pytest.raises(RuntimeError):
        client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert not service.is_busy(sid), "异常路径也必须释放会话锁"
