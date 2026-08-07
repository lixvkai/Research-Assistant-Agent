"""pytest 共享夹具 — 提供桩化 LLM 响应，使全部测试离线可跑。"""

import json
import os
import types

import pytest

# 测试套件必须完全离线；开发机 .env 即使开启 Langfuse 也不能让单测上报数据。
os.environ["LANGFUSE_ENABLED"] = "false"


def _make_response(content, tool_calls=None):
    """构造一个仿 OpenAI 的响应对象：resp.choices[0].message。"""
    dump = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    msg = types.SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda d=dump: dict(d),
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _tool_call(name, arguments):
    return {
        "id": f"call_{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


@pytest.fixture
def mkresp():
    return _make_response


@pytest.fixture
def mktool():
    return _tool_call


class FakeAgent:
    """供 skill 执行引擎测试使用的最小 Agent 桩。"""

    def __init__(self, events):
        self._events = events

    def reset(self):
        pass

    def run_iter(self, prompt):
        for e in self._events:
            yield e


@pytest.fixture
def fake_agent_cls():
    return FakeAgent


# ── 服务层 / API 测试用的桩件 ──────────────────────────────────

DEFAULT_EVENTS = [
    {"type": "step_start", "step": 1, "max_steps": 10},
    {"type": "thought", "content": "先查一下文献", "step": 1},
    {"type": "action", "tool": "search_arxiv", "args": {"query": "rag"}, "step": 1},
    {"type": "observation", "result": "找到 3 篇论文", "step": 1},
    {"type": "answer", "content": "这是最终答案"},
]


class StubAgent:
    """记录调用的 ReActAgent 替身：不触发任何 LLM 请求。"""

    def __init__(self, events=None):
        self.events = list(events if events is not None else DEFAULT_EVENTS)
        self.runs: list[tuple[str, str | None]] = []
        self.loaded: dict[str | None, list] = {}
        self.resets: list[str | None] = []

    def run_iter(self, user_input, session_id=None):
        self.runs.append((user_input, session_id))
        for event in self.events:
            yield dict(event)

    def load_history(self, messages, session_id=None):
        self.loaded[session_id] = list(messages)

    def reset(self, session_id=None):
        self.resets.append(session_id)


class StubMemory:
    """只需要 consolidate 的 MemoryManager 替身。"""

    def __init__(self):
        self.consolidated: list[str | None] = []

    def consolidate(self, session_id=None):
        self.consolidated.append(session_id)
        return 0


@pytest.fixture
def make_service(tmp_path):
    """构造一个用临时 SQLite + 桩 Agent 的 AgentService。"""
    from memory.chat_history import ChatHistoryStore
    from services.agent_service import AgentService

    created = {}

    def _make(events=None, tools=None):
        agent = StubAgent(events)
        memory = StubMemory()
        service = AgentService(
            agent=agent,
            history_store=ChatHistoryStore(db_path=str(tmp_path / "chat.db")),
            memory=memory,
            tools=tools if tools is not None else [],
        )
        created.update(service=service, agent=agent, memory=memory)
        return service

    _make.state = created
    return _make


@pytest.fixture
def api_client(make_service, monkeypatch):
    """TestClient + 被替换成桩件的 AgentService 依赖。"""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from config import settings
    from services import get_agent_service

    # 关掉预热，否则 lifespan 会去加载真实 Agent 与全部工具
    monkeypatch.setattr(settings, "API_WARMUP", False)

    from api.main import create_app

    def _make(events=None):
        service = make_service(events)
        app = create_app()
        app.dependency_overrides[get_agent_service] = lambda: service
        return TestClient(app), service

    return _make
