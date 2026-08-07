"""FastAPI 应用 —— 把 Agent 能力暴露为 HTTP 服务。

定位：业务逻辑全在 `services/`，这一层只做协议转换（HTTP ⇄ 服务层调用）。
Gradio UI 与本服务是同一服务层的两个客户端，因此两者行为天然一致。

启动：
    uvicorn api.main:app --reload --port 8000
    # 或 python -m api.main
文档：http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, knowledge, meta, sessions
from config import settings
from core.observability import shutdown_observability

logger = logging.getLogger(__name__)

DESCRIPTION = """
基于 LangGraph 的科研助手 Agent HTTP 接口。

- `POST /api/chat/stream` —— SSE 流式返回 ReAct 推理轨迹（思考 / 工具调用 / 观察 / 反思 / 答案）
- `POST /api/chat` —— 非流式，只要最终答案
- `/api/sessions/*` —— 多会话管理，会话之间的对话状态与短期记忆互相隔离
- `/api/knowledge/*` —— 论文知识库的导入与管理
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if settings.API_WARMUP:
        # 提前把 Agent 和工具装好，否则第一个请求要等好几秒才出第一个事件
        try:
            from services import get_agent_service

            service = get_agent_service()
            logger.info("Agent 预热完成，已加载 %d 个工具", len(service.list_tools()))
        except Exception as e:
            logger.warning("Agent 预热失败（首个请求时会重试）：%s", e)
    try:
        yield
    finally:
        shutdown_observability()


def create_app() -> FastAPI:
    app = FastAPI(
        title="科研助手 Agent API",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )

    if settings.API_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.API_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(meta.router)
    app.include_router(sessions.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(knowledge.router, prefix="/api")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )
