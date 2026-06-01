"""
Skill 执行引擎 — 把"自然语言步骤"的技能真正驱动起来。

做法：将技能的步骤、所需工具与具体任务拼成一份带流程约束的提示，交给一个绑定了
全部工具的 ReActAgent 执行；执行结束后按是否产出有效答案判定成功，回写成功率统计。

注意：本模块在函数内部懒加载 ReActAgent / MCP，避免与 core.mcp 形成模块级循环导入。
"""

import logging

from skills.skill_manager import SkillManager

logger = logging.getLogger(__name__)

_manager: SkillManager | None = None


def _get_manager() -> SkillManager:
    global _manager
    if _manager is None:
        _manager = SkillManager()
    return _manager


SKILL_SYSTEM_PROMPT = """你是科研助手 Agent。现在要按照一个预定义「技能」的标准流程来完成用户任务。

请严格参考给定的步骤顺序推进，按需调用合适的工具；某些步骤若不适用当前任务可灵活跳过，
但要保证最终产出完整、准确、有条理。使用中文回答。"""


def _build_prompt(skill, task: str) -> str:
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(skill.steps))
    tools = "、".join(skill.tools_needed) if skill.tools_needed else "（无特定要求）"
    return (
        f"【技能】{skill.name}\n"
        f"【技能说明】{skill.description}\n"
        f"【建议工具】{tools}\n"
        f"【标准步骤】\n{steps}\n\n"
        f"【本次任务】\n{task}\n\n"
        f"请按上述流程完成本次任务。"
    )


def _build_agent():
    """构造一个绑定全部工具的执行 Agent（懒加载，避免循环导入）。"""
    from core.react_agent import ReActAgent
    from core.mcp import create_default_mcp_server

    agent = ReActAgent(system_prompt=SKILL_SYSTEM_PROMPT, enable_reflection=False)
    create_default_mcp_server().bind_to_agent(agent)
    return agent


def execute_skill(name: str, task: str, manager: SkillManager | None = None, agent=None) -> str:
    """按技能流程执行任务，回写成功率统计，返回最终结果。"""
    mgr = manager or _get_manager()
    skill = mgr.get_skill(name)
    if skill is None:
        return f"未找到技能 '{name}'。可先用 list_skills 查看可用技能。"

    prompt = _build_prompt(skill, task)
    if agent is None:
        agent = _build_agent()

    answer = ""
    had_error = False
    try:
        agent.reset()
        for event in agent.run_iter(prompt):
            etype = event["type"]
            if etype == "answer":
                answer = event["content"]
            elif etype == "error":
                had_error = True
                answer = answer or f"技能执行出错：{event['content']}"
    except Exception as e:
        logger.exception("技能 '%s' 执行异常", name)
        had_error = True
        answer = f"技能执行出错：{e}"

    success = bool(answer) and not had_error and answer != "未能生成回答。"
    mgr.record_execution(name, success)
    logger.info("技能 '%s' 执行%s（成功率 %.0f%%）",
                name, "成功" if success else "失败", skill.success_rate * 100)
    return answer or "技能执行未产出结果。"
