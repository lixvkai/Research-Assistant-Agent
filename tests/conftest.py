"""pytest 共享夹具 — 提供桩化 LLM 响应，使全部测试离线可跑。"""

import json
import types

import pytest


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
