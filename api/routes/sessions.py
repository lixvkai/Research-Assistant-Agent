"""会话 CRUD。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from starlette.concurrency import run_in_threadpool

from api.schemas import MessageOut, SessionCreate, SessionDetailOut, SessionOut
from services import AgentService, SessionNotFoundError, get_agent_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _not_found(exc: SessionNotFoundError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate | None = None,
    service: AgentService = Depends(get_agent_service),
) -> SessionOut:
    title = payload.title if payload else None
    info = await run_in_threadpool(service.create_session, title)
    return SessionOut(**info)


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    limit: int = Query(default=30, ge=1, le=200),
    service: AgentService = Depends(get_agent_service),
) -> list[SessionOut]:
    rows = await run_in_threadpool(service.list_sessions, limit)
    return [SessionOut(**r) for r in rows]


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_session(
    session_id: int,
    service: AgentService = Depends(get_agent_service),
) -> SessionDetailOut:
    try:
        info = await run_in_threadpool(service.get_session, session_id)
        messages = await run_in_threadpool(service.get_messages, session_id)
    except SessionNotFoundError as e:
        raise _not_found(e) from e
    return SessionDetailOut(**info, messages=[MessageOut(**m) for m in messages])


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def get_messages(
    session_id: int,
    service: AgentService = Depends(get_agent_service),
) -> list[MessageOut]:
    try:
        messages = await run_in_threadpool(service.get_messages, session_id)
    except SessionNotFoundError as e:
        raise _not_found(e) from e
    return [MessageOut(**m) for m in messages]


@router.post(
    "/{session_id}/reset",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
async def reset_session(
    session_id: int,
    service: AgentService = Depends(get_agent_service),
) -> None:
    """结束一段会话：把内容固化进长期记忆，并清空进程内的对话状态。

    历史消息本身保留在库里，只是 Agent 不再把它们当作当前上下文。
    """
    try:
        await run_in_threadpool(service.get_session, session_id)
    except SessionNotFoundError as e:
        raise _not_found(e) from e
    await run_in_threadpool(service.reset_session, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
)
async def delete_session(
    session_id: int,
    service: AgentService = Depends(get_agent_service),
) -> None:
    try:
        await run_in_threadpool(service.delete_session, session_id)
    except SessionNotFoundError as e:
        raise _not_found(e) from e
    return Response(status_code=status.HTTP_204_NO_CONTENT)
