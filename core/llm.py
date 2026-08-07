"""DeepSeek LLM client — wraps the OpenAI-compatible API。

客户端统一配置请求超时与自动重试：OpenAI SDK 对超时 / 限流(429) / 5xx / 连接错误
等瞬时故障会按指数退避自动重试 `max_retries` 次，避免单次网络抖动导致整轮失败。
"""

import threading
from typing import Any

from openai import OpenAI as StandardOpenAI

from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from core.budget import consume_llm_call
from core.observability import (
    get_langfuse_client,
    is_observability_enabled,
    openai_trace_kwargs,
)

_client: Any | None = None
_client_is_traced = False
_client_lock = threading.Lock()


def get_client() -> Any:
    """进程级共享客户端（线程安全）——避免每次请求重建连接池。"""
    global _client, _client_is_traced
    if _client is None:
        with _client_lock:
            if _client is None:
                client_class = StandardOpenAI
                if is_observability_enabled():
                    try:
                        # Initialize Langfuse before constructing its OpenAI wrapper.
                        get_langfuse_client()
                        from langfuse.openai import OpenAI as LangfuseOpenAI

                        client_class = LangfuseOpenAI
                        _client_is_traced = True
                    except Exception:
                        # Tracing is fail-open; the standard DeepSeek client still works.
                        client_class = StandardOpenAI
                        _client_is_traced = False
                _client = client_class(
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    timeout=LLM_TIMEOUT,
                    max_retries=LLM_MAX_RETRIES,
                )
    return _client


def chat(
    messages: list[dict],
    model: str = DEEPSEEK_MODEL,
    temperature: float = 0.7,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
    observation_name: str = "deepseek-chat",
    metadata: dict[str, Any] | None = None,
):
    """Send a chat completion request and return the raw response."""
    consume_llm_call()
    client = get_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    if _client_is_traced:
        kwargs.update(openai_trace_kwargs(observation_name))
        if metadata:
            kwargs["metadata"] = metadata
    return client.chat.completions.create(**kwargs)
