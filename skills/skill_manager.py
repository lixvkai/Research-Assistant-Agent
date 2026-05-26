"""
Skills 技能系统

将常见的科研工作流固化为可复用的 Skill，
Agent 可以按需检索和执行已有技能，也可以从交互中学习新技能。
"""

import json
import os
from dataclasses import dataclass, asdict
from config.settings import DEEPSEEK_MODEL


SKILLS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "skills.json")


@dataclass
class Skill:
    """一个可复用的技能。"""
    name: str
    description: str
    steps: list[str]          # 执行步骤描述
    tools_needed: list[str]   # 需要用到的工具名
    category: str             # "literature" | "analysis" | "writing" | "general"
    usage_count: int = 0


class SkillManager:
    """技能管理器 — 存储、检索和执行技能。"""

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._load_skills()
        if not self.skills:
            self._init_builtin_skills()

    def _load_skills(self):
        if os.path.exists(SKILLS_FILE):
            with open(SKILLS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self.skills[item["name"]] = Skill(**item)

    def _save_skills(self):
        os.makedirs(os.path.dirname(SKILLS_FILE), exist_ok=True)
        data = [asdict(s) for s in self.skills.values()]
        with open(SKILLS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _init_builtin_skills(self):
        builtins = [
            Skill(
                name="文献综述",
                description="对某个研究主题进行系统的文献综述",
                steps=[
                    "使用 search_arxiv 搜索相关论文（多个关键词组合）",
                    "从搜索结果中筛选最相关的 5-10 篇论文",
                    "使用 fetch_webpage 获取论文详情",
                    "使用 summarize_text 总结每篇论文的要点",
                    "综合分析，撰写文献综述（研究现状、主要方法、发展趋势）",
                    "使用 save_research_finding 保存关键发现",
                ],
                tools_needed=["search_arxiv", "fetch_webpage", "summarize_text", "save_research_finding"],
                category="literature",
            ),
            Skill(
                name="论文精读",
                description="深入阅读和分析一篇特定论文",
                steps=[
                    "获取论文全文（通过 URL 或本地文件）",
                    "提取论文结构：摘要、引言、方法、实验、结论",
                    "分析核心方法和创新点",
                    "评估实验设计和结果的合理性",
                    "总结论文的优缺点和潜在改进方向",
                    "保存关键发现到记忆系统",
                ],
                tools_needed=["fetch_webpage", "ingest_paper", "search_knowledge_base", "summarize_text"],
                category="literature",
            ),
            Skill(
                name="研究方案设计",
                description="为一个研究问题设计详细的研究方案",
                steps=[
                    "明确研究问题和目标",
                    "检索相关文献了解现有方法",
                    "使用 multi_agent_collaborate 召集多专家讨论方案",
                    "设计研究方法和实验流程",
                    "明确评估指标和基线对比",
                    "撰写研究方案文档",
                ],
                tools_needed=["search_arxiv", "search_knowledge_base", "multi_agent_collaborate"],
                category="analysis",
            ),
            Skill(
                name="学术写作辅助",
                description="辅助撰写论文的各个部分",
                steps=[
                    "确认要撰写的部分（摘要/引言/方法/实验/结论）",
                    "从知识库检索相关素材",
                    "回忆之前的研究发现",
                    "使用 multi_agent_collaborate 由写作专家撰写初稿",
                    "由审查专家给出修改建议",
                    "输出最终版本",
                ],
                tools_needed=["search_knowledge_base", "recall_memories", "multi_agent_collaborate"],
                category="writing",
            ),
        ]
        for skill in builtins:
            self.skills[skill.name] = skill
        self._save_skills()

    def add_skill(self, skill: Skill):
        self.skills[skill.name] = skill
        self._save_skills()

    def get_skill(self, name: str) -> Skill | None:
        skill = self.skills.get(name)
        if skill:
            skill.usage_count += 1
            self._save_skills()
        return skill

    def list_skills(self, category: str | None = None) -> list[Skill]:
        skills = list(self.skills.values())
        if category:
            skills = [s for s in skills if s.category == category]
        return sorted(skills, key=lambda s: s.usage_count, reverse=True)

    def search_skills(self, keyword: str) -> list[Skill]:
        keyword_lower = keyword.lower()
        return [
            s for s in self.skills.values()
            if keyword_lower in s.name.lower() or keyword_lower in s.description.lower()
        ]
