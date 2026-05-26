"""
Orchestrator — 多 Agent 协调器

负责将复杂任务拆分为子任务，分配给合适的专家 Agent，
最终汇总各专家的结果给出综合回答。
"""

import json
import logging

from core.llm import chat
from agents.specialists import EXPERT_REGISTRY, ExpertAgent

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """你是一个科研任务规划专家。你的职责是将用户的复杂科研任务拆分为子任务，并分配给合适的专家。

可用的专家：
- literature: 学术文献检索与分析专家
- data_analysis: 数据分析与可视化专家
- writing: 学术写作专家
- review: 质量审查与评审专家

请以 JSON 格式输出任务分配方案，格式如下：
{
    "plan_summary": "任务规划概述",
    "subtasks": [
        {"expert": "专家名称", "task": "具体子任务描述", "depends_on": []}
    ]
}

注意：
- depends_on 是依赖的子任务索引列表（0-based），空列表表示无依赖可并行
- 根据任务复杂度合理拆分，简单任务不要过度拆分
- 每个子任务描述要具体、可执行
"""


class Orchestrator:
    """多 Agent 协调器。"""

    def __init__(self):
        self._experts: dict[str, ExpertAgent] = {}

    def _get_expert(self, name: str) -> ExpertAgent:
        if name not in self._experts:
            cls = EXPERT_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"未知专家：{name}")
            self._experts[name] = cls()
        return self._experts[name]

    def plan(self, task: str) -> dict:
        """用 LLM 生成任务执行计划。"""
        response = chat(
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content

        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1:
            return {"plan_summary": "无法解析计划", "subtasks": [{"expert": "literature", "task": task, "depends_on": []}]}
        return json.loads(content[start:end])

    def execute(self, task: str) -> str:
        """规划并执行多 Agent 协作任务。"""
        logger.info("Multi-Agent 协作任务: %s", task)

        plan = self.plan(task)
        logger.info("任务规划: %s", plan.get("plan_summary", ""))

        subtasks = plan.get("subtasks", [])
        results = {}

        for i, sub in enumerate(subtasks):
            expert_name = sub["expert"]
            sub_task = sub["task"]
            deps = sub.get("depends_on", [])

            context_parts = []
            for dep_idx in deps:
                if dep_idx in results:
                    context_parts.append(f"[子任务{dep_idx}结果]\n{results[dep_idx]}")
            context = "\n\n".join(context_parts)

            logger.info("子任务 %d → %s: %s", i, expert_name, sub_task)

            try:
                expert = self._get_expert(expert_name)
                result = expert.run(sub_task, context=context)
                results[i] = result
                logger.info("%s 完成子任务 %d", expert_name, i)
            except Exception as e:
                results[i] = f"执行出错：{e}"
                logger.exception("子任务 %d 出错", i)

        summary_parts = [f"[子任务{i} - {sub['expert']}]\n{results.get(i, '无结果')}" for i, sub in enumerate(subtasks)]
        summary_context = "\n\n---\n\n".join(summary_parts)

        final_response = chat(
            messages=[
                {"role": "system", "content": "你是科研助手，请将各专家的分析结果综合成一份完整、有条理的回答。使用中文。"},
                {"role": "user", "content": f"原始任务：{task}\n\n各专家分析结果：\n{summary_context}\n\n请综合以上结果，给出完整回答。"},
            ],
            temperature=0.5,
        )

        return final_response.choices[0].message.content
