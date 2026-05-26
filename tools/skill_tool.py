"""技能工具 — 让 Agent 可以查看和使用预定义技能。"""

import json
from skills.skill_manager import SkillManager, Skill

_manager: SkillManager | None = None


def _get_manager() -> SkillManager:
    global _manager
    if _manager is None:
        _manager = SkillManager()
    return _manager


def list_skills(category: str = "") -> str:
    """列出所有可用技能。"""
    mgr = _get_manager()
    skills = mgr.list_skills(category if category else None)
    if not skills:
        return "暂无可用技能。"
    lines = []
    for s in skills:
        lines.append(f"[{s.category}] {s.name} — {s.description} (使用{s.usage_count}次)")
    return "可用技能：\n" + "\n".join(lines)


def get_skill_detail(name: str) -> str:
    """获取技能的详细执行步骤。"""
    mgr = _get_manager()
    skill = mgr.get_skill(name)
    if not skill:
        return f"未找到技能 '{name}'。"
    steps = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(skill.steps))
    tools = ", ".join(skill.tools_needed)
    return (
        f"技能：{skill.name}\n"
        f"描述：{skill.description}\n"
        f"分类：{skill.category}\n"
        f"需要工具：{tools}\n"
        f"执行步骤：\n{steps}"
    )


def create_skill(name: str, description: str, steps: str, tools_needed: str, category: str = "general") -> str:
    """创建新技能。"""
    mgr = _get_manager()
    skill = Skill(
        name=name,
        description=description,
        steps=[s.strip() for s in steps.split(";") if s.strip()],
        tools_needed=[t.strip() for t in tools_needed.split(",") if t.strip()],
        category=category,
    )
    mgr.add_skill(skill)
    return f"成功创建技能 '{name}'，共 {len(skill.steps)} 个步骤。"


TOOL_DEFINITIONS = [
    {
        "name": "list_skills",
        "description": "列出所有可用的科研技能。可按分类筛选（literature/analysis/writing/general）。",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "可选分类筛选", "default": ""},
            },
            "required": [],
        },
        "func": list_skills,
    },
    {
        "name": "get_skill_detail",
        "description": "获取某个技能的详细执行步骤和所需工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
            },
            "required": ["name"],
        },
        "func": get_skill_detail,
    },
    {
        "name": "create_skill",
        "description": "创建一个新的可复用技能。当发现常见的工作流模式时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "技能名称"},
                "description": {"type": "string", "description": "技能描述"},
                "steps": {"type": "string", "description": "执行步骤，用分号分隔"},
                "tools_needed": {"type": "string", "description": "所需工具，用逗号分隔"},
                "category": {"type": "string", "description": "分类：literature/analysis/writing/general", "default": "general"},
            },
            "required": ["name", "description", "steps", "tools_needed"],
        },
        "func": create_skill,
    },
]
