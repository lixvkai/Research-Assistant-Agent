"""Multi-Agent 工具 — 让主 Agent 可以调度多专家协作。"""

from agents.orchestrator import Orchestrator

_orchestrator: Orchestrator | None = None


def _get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def multi_agent_collaborate(task: str) -> str:
    """将复杂科研任务分配给多个专家 Agent 协作完成。"""
    orch = _get_orchestrator()
    return orch.execute(task)


TOOL_DEFINITION = {
    "name": "multi_agent_collaborate",
    "description": "将复杂的科研任务交给多个专家 Agent 协作处理。适用于需要多角度分析的任务，如文献综述、研究方案设计、论文撰写等。",
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "需要多专家协作的复杂任务描述",
            }
        },
        "required": ["task"],
    },
    "func": multi_agent_collaborate,
}
