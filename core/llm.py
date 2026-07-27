"""DeepSeek LLM client — wraps the OpenAI-compatible API。

客户端统一配置请求超时与自动重试：OpenAI SDK 对超时 / 限流(429) / 5xx / 连接错误
等瞬时故障会按指数退避自动重试 `max_retries` 次，避免单次网络抖动导致整轮失败。
"""

import threading

from openai import OpenAI

from config.settings import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT,
)
from core.budget import consume_llm_call

_client: OpenAI | None = None
_client_lock = threading.Lock()


def get_client() -> OpenAI:
    """进程级共享客户端（线程安全）——避免每次请求重建连接池。"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI(
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
):
    """Send a chat completion request and return the raw response."""
    consume_llm_call()
    client = get_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)
