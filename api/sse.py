"""Server-Sent Events 编码。

事件格式与 OpenAI / Anthropic 的流式接口一致：`event:` 行给出事件名（这里直接沿用
ReAct 的事件类型，如 step_start / action / observation / answer），`data:` 行是一行 JSON。
`json.dumps` 会转义换行，所以单条 data 永远不会被换行截断。
"""

from __future__ import annotations

import json
from typing import Any

# 关掉各级缓冲，否则事件会被攒成一批再发，流式效果就没了
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_comment(text: str = "keep-alive") -> str:
    """SSE 注释行 —— 用于心跳，客户端会忽略它。"""
    return f": {text}\n\n"
