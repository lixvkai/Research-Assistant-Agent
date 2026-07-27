"""HTTP 层的请求 / 响应模型。

与 `core.schemas` 的分工：`core.schemas` 是 Agent 内部的契约（ReAct 事件、工具规范），
这里只描述对外的 HTTP 形状，避免把内部模型直接暴露成 API 契约。
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── 会话 ──────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SessionOut(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str


class SessionDetailOut(SessionOut):
    messages: list[MessageOut] = Field(default_factory=list)


# ── 对话 ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    # 为空表示新建会话，会话 id 会通过 SSE 的第一个 `session` 事件返回
    session_id: int | None = None


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    steps: int = 0


# ── 元信息 ────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str = "ok"
    model: str
    tools: int


class ToolOut(BaseModel):
    name: str
    description: str
    category: str
    version: str = "1.0"


class SkillOut(BaseModel):
    name: str
    description: str
    category: str
    usage_count: int = 0
    success_rate: float = 0.0


# ── 知识库 ────────────────────────────────────────────────────

class KBStatsOut(BaseModel):
    files: int
    chunks: int
    collection: str = ""


class KBFileOut(BaseModel):
    name: str
    size: int


class KBIngestOut(BaseModel):
    ok: bool
    filename: str
    chunks: int | None = None
    error: str | None = None
