"""元信息端点 —— 健康检查、工具清单、技能清单。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from api.schemas import HealthOut, SkillOut, ToolOut
from config.settings import DEEPSEEK_MODEL
from services import AgentService, get_agent_service

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthOut)
async def health(service: AgentService = Depends(get_agent_service)) -> HealthOut:
    tools = await run_in_threadpool(service.list_tools)
    return HealthOut(model=DEEPSEEK_MODEL, tools=len(tools))


@router.get("/api/tools", response_model=list[ToolOut])
async def list_tools(service: AgentService = Depends(get_agent_service)) -> list[ToolOut]:
    tools = await run_in_threadpool(service.list_tools)
    return [ToolOut(**t) for t in tools]


@router.get("/api/skills", response_model=list[SkillOut])
async def list_skills(service: AgentService = Depends(get_agent_service)) -> list[SkillOut]:
    skills = await run_in_threadpool(service.list_skills)
    return [SkillOut(**s) for s in skills]
