"""对话端点 —— SSE 流式推送 ReAct 轨迹，另提供一个非流式的便捷接口。

SSE 事件序列：
    session      {"session_id": 12}                首个事件，客户端据此记住会话
    step_start   {"step": 1, "max_steps": 10}
    thought      {"content": "..."}
    action       {"tool": "search_arxiv", "args": {...}}
    observation  {"result": "..."}
    reflection   {"sufficient": false, "critique": "..."}
    answer       {"content": "最终答案"}
    error        {"content": "..."}                异常时出现，之后仍会有 done
    done         {"session_id": 12}                流正常结束的标志
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from api.schemas import ChatRequest, ChatResponse
from api.sse import SSE_HEADERS, sse_event
from api.streaming import aiter_in_thread
from services import (
    AgentService,
    SessionBusyError,
    SessionNotFoundError,
    get_agent_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


async def _prepare(service: AgentService, payload: ChatRequest) -> int:
    """确定目标会话并校验其可用性。"""
    message = payload.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message 不能为空",
        )
    try:
        return await run_in_threadpool(service.ensure_session, payload.session_id, message)
    except SessionNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> StreamingResponse:
    """流式对话。同一会话若已有推理在跑，返回 409。"""
    session_id = await _prepare(service, payload)
    message = payload.message.strip()
    try:
        guard = service.acquire_session(session_id)
    except SessionBusyError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    async def event_stream():
        try:
            yield sse_event("session", {"session_id": session_id})
            stream = aiter_in_thread(lambda: service.stream_chat(session_id, message))
            async for event in stream:
                yield sse_event(event.get("type", "message"), event)
            yield sse_event("done", {"session_id": session_id})
        except Exception as e:
            # 客户端断连走的是 GeneratorExit（BaseException），不会落到这里
            logger.exception("会话 %s 流式对话失败", session_id)
            yield sse_event("error", {"type": "error", "content": str(e)})
            yield sse_event("done", {"session_id": session_id})
        finally:
            guard.release()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    """非流式对话 —— 供脚本、批量评测等不需要中间轨迹的调用方使用。"""
    session_id = await _prepare(service, payload)
    message = payload.message.strip()
    try:
        guard = service.acquire_session(session_id)
    except SessionBusyError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e)) from e

    def _run() -> tuple[str, int]:
        answer, steps = "", 0
        for event in service.stream_chat(session_id, message):
            etype = event.get("type")
            if etype == "step_start":
                steps += 1
            elif etype == "answer":
                answer = event.get("content") or ""
        return answer, steps

    try:
        answer, steps = await run_in_threadpool(_run)
    finally:
        guard.release()
    return ChatResponse(session_id=session_id, answer=answer, steps=steps)
