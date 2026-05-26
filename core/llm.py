"""DeepSeek LLM client — wraps the OpenAI-compatible API."""

from openai import OpenAI
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def chat(
    messages: list[dict],
    model: str = DEEPSEEK_MODEL,
    temperature: float = 0.7,
    tools: list[dict] | None = None,
    tool_choice: str = "auto",
):
    """Send a chat completion request and return the raw response."""
    client = get_client()
    kwargs = dict(model=model, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def chat_stream(
    messages: list[dict],
    model: str = DEEPSEEK_MODEL,
    temperature: float = 0.7,
):
    """Stream a chat completion, yielding text chunks."""
    client = get_client()
    stream = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
