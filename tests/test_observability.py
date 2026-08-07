"""Langfuse 适配层单测：全部使用假客户端，不访问网络。"""

from __future__ import annotations

import contextlib
import types

import pytest

from config import settings
from core import observability


class FakeObservation:
    def __init__(self):
        self.trace_id = "a" * 32
        self.id = "b" * 16
        self.updates = []
        self.ended = False

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self

    def end(self):
        self.ended = True


class FakeClient:
    def __init__(self):
        self.observation = FakeObservation()
        self.start_kwargs = None

    def start_observation(self, **kwargs):
        self.start_kwargs = kwargs
        return self.observation

    @contextlib.contextmanager
    def start_as_current_observation(self, **kwargs):
        self.start_kwargs = kwargs
        yield self.observation

    def get_current_trace_id(self):
        return None

    def get_current_observation_id(self):
        return None


def test_mask_payload_redacts_credentials_and_local_username(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_CAPTURE_CONTENT", True)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "sk-secret-example")
    payload = {
        "authorization": "Bearer top-secret-token",
        "nested": "key=sk-secret-example C:\\Users\\alice\\paper.pdf",
    }

    masked = observability.mask_payload(payload)

    assert masked["authorization"] == "***"
    assert "sk-secret-example" not in masked["nested"]
    assert "alice" not in masked["nested"]

    # Langfuse SDK v4 calls the hook with a keyword-only ``data`` argument.
    assert observability.mask_payload(data={"token": "secret"}) == {"token": "***"}


def test_capture_can_omit_content(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_CAPTURE_CONTENT", False)

    assert observability.capture_value("private text") == {
        "content_captured": False,
        "characters": 12,
    }


def test_start_and_finish_root_agent_trace(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(settings, "LANGFUSE_CAPTURE_CONTENT", True)

    trace = observability.start_agent_trace(
        session_id="42", input="hello", metadata={"entrypoint": "test"}
    )
    assert trace is not None
    assert trace.scope.session_id == "42"
    assert client.start_kwargs["as_type"] == "agent"

    observability.finish_agent_trace(trace, output="world")
    assert client.observation.ended is True
    assert client.observation.updates[-1]["output"] == "world"


def test_child_operation_uses_explicit_root_parent(monkeypatch):
    client = FakeClient()
    scope = observability.TraceScope("a" * 32, "b" * 16, "42")
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: client)

    with observability.bind_trace_scope(scope):
        with observability.observe_operation("lookup", as_type="tool", input={"q": "x"}) as obs:
            obs.update(output="ok")

    assert client.start_kwargs["trace_context"] == {
        "trace_id": scope.trace_id,
        "parent_span_id": scope.parent_observation_id,
    }
    assert client.start_kwargs["as_type"] == "tool"


def test_disabled_operation_is_noop(monkeypatch):
    monkeypatch.setattr(observability, "get_langfuse_client", lambda: None)
    scope = observability.TraceScope("a" * 32, "b" * 16, "42")

    with observability.bind_trace_scope(scope):
        with observability.observe_operation("noop") as obs:
            assert obs is None


def test_standard_openai_fallback_never_receives_langfuse_kwargs(monkeypatch):
    from core import llm

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(choices=[])

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_client_is_traced", False)
    monkeypatch.setattr(llm, "StandardOpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "is_observability_enabled", lambda: True)
    monkeypatch.setattr(
        llm, "get_langfuse_client", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    llm.chat([{"role": "user", "content": "hello"}])

    assert "name" not in captured
    assert "trace_id" not in captured
    assert "parent_observation_id" not in captured
