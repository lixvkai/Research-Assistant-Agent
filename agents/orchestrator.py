"""
Orchestrator — 基于 LangGraph 的多 Agent 协调器

图结构：
    START → plan → execute → synthesize → reflect → [需修订?] ─是→ synthesize
                                                        └─否→ END

- plan：LLM 规划，用 Pydantic `Plan` 校验，失败走兜底
- execute：按依赖顺序调度专家 Agent
- synthesize：融合各专家结果（修订时带上反思批评）
- reflect：对融合稿自我审查，不合格则回到 synthesize 重写
"""

import json
import logging
from typing import TypedDict

from pydantic import ValidationError
from langgraph.graph import END, START, StateGraph

from core.llm import chat
from core.mcp import MCPServer, create_default_mcp_server
from core.schemas import Plan, Reflection, SubTask
from config.settings import MAX_REFLECTIONS
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

SYNTHESIS_PROMPT = "你是科研助手，请将各专家的分析结果综合成一份完整、有条理的回答。使用中文。"

REFLECT_PROMPT = """你是严格的质量审查员。请评估下面这份针对原始任务的综合回答是否完整、准确、有条理。

只输出 JSON：{"sufficient": true/false, "critique": "若不充分，指出具体不足"}"""


class _OrchState(TypedDict):
    """Orchestrator 图的运行状态。"""
    task: str
    plan: Plan
    results: dict
    draft: str
    reflections: int
    done: bool
    critique: str


class Orchestrator:
    """基于 LangGraph 的多 Agent 协调器。"""

    def __init__(self):
        self._experts: dict[str, ExpertAgent] = {}
        self._mcp_server: MCPServer | None = None
        self._graph = None

    # ── 资源 ────────────────────────────────────────────────────

    def _get_mcp_server(self) -> MCPServer:
        if self._mcp_server is None:
            self._mcp_server = create_default_mcp_server()
        return self._mcp_server

    def _get_expert(self, name: str) -> ExpertAgent:
        if name not in self._experts:
            cls = EXPERT_REGISTRY.get(name)
            if cls is None:
                raise ValueError(f"未知专家：{name}")
            self._experts[name] = cls(mcp_server=self._get_mcp_server())
        return self._experts[name]

    # ── 规划解析 ────────────────────────────────────────────────

    def plan(self, task: str) -> Plan:
        """用 LLM 生成任务执行计划，并用 Pydantic 校验；失败时走兜底计划。"""
        response = chat(
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return self._parse_plan(content, task)

    @staticmethod
    def _fallback_plan(task: str, summary: str = "无法解析计划，回退到单专家处理") -> Plan:
        return Plan(plan_summary=summary, subtasks=[SubTask(expert="literature", task=task)])

    def _parse_plan(self, content: str, task: str) -> Plan:
        """从 LLM 文本中提取 JSON 并用 Plan 校验。任何失败均回退到兜底计划。"""
        raw = self._extract_json(content)
        if raw is None:
            return self._fallback_plan(task)
        try:
            data = json.loads(raw)
            plan = Plan.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("规划解析失败，使用兜底计划：%s", e)
            return self._fallback_plan(task)
        if not plan.subtasks:
            return self._fallback_plan(task, summary=plan.plan_summary or "规划为空，回退到单专家处理")
        return plan

    @staticmethod
    def _extract_json(content: str) -> str | None:
        """从可能包裹 ```json 代码块或自然语言的文本中提取 JSON 对象串。"""
        if not content:
            return None
        text = content.strip()
        if "```" in text:
            segment = text.split("```", 2)
            if len(segment) >= 2:
                fenced = segment[1]
                if fenced.lstrip().lower().startswith("json"):
                    fenced = fenced.lstrip()[4:]
                text = fenced
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        return text[start:end]

    # ── 图节点 ──────────────────────────────────────────────────

    def _plan_node(self, state: _OrchState) -> dict:
        plan = self.plan(state["task"])
        logger.info("任务规划: %s", plan.plan_summary)
        return {"plan": plan}

    def _execute_node(self, state: _OrchState) -> dict:
        """按依赖顺序调度专家。依赖结果作为下游专家的上下文。"""
        results: dict[int, str] = {}
        for i, sub in enumerate(state["plan"].subtasks):
            expert_name = sub.expert.value
            context_parts = [
                f"[子任务{d}结果]\n{results[d]}" for d in sub.depends_on if d in results
            ]
            context = "\n\n".join(context_parts)
            logger.info("子任务 %d → %s: %s", i, expert_name, sub.task)
            try:
                result = self._get_expert(expert_name).run(sub.task, context=context)
                results[i] = result
                logger.info("%s 完成子任务 %d", expert_name, i)
            except Exception as e:
                results[i] = f"执行出错：{e}"
                logger.exception("子任务 %d 出错", i)
        return {"results": results}

    def _synthesize_node(self, state: _OrchState) -> dict:
        subtasks = state["plan"].subtasks
        results = state["results"]
        summary = "\n\n---\n\n".join(
            f"[子任务{i} - {sub.expert.value}]\n{results.get(i, '无结果')}"
            for i, sub in enumerate(subtasks)
        )
        user_content = (
            f"原始任务：{state['task']}\n\n各专家分析结果：\n{summary}\n\n"
            "请综合以上结果，给出完整回答。"
        )
        if state.get("critique"):
            user_content += f"\n\n[上一版的不足，请改进]\n{state['critique']}"

        resp = chat(
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
        )
        return {"draft": resp.choices[0].message.content}

    def _reflect_node(self, state: _OrchState) -> dict:
        if state["reflections"] >= MAX_REFLECTIONS:
            return {"done": True}
        verdict = self._reflect(state["task"], state["draft"])
        if verdict.sufficient:
            return {"done": True}
        logger.info("综合稿未通过反思，准备修订：%s", verdict.critique)
        return {"done": False, "critique": verdict.critique, "reflections": state["reflections"] + 1}

    def _reflect(self, task: str, draft: str) -> Reflection:
        try:
            resp = chat(
                messages=[
                    {"role": "system", "content": REFLECT_PROMPT},
                    {"role": "user", "content": f"原始任务：\n{task}\n\n综合回答：\n{draft}"},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}") + 1
            if start == -1 or end <= start:
                return Reflection()
            return Reflection.model_validate_json(content[start:end])
        except Exception as e:
            logger.warning("反思解析失败，默认通过：%s", e)
            return Reflection()

    @staticmethod
    def _route_after_reflect(state: _OrchState) -> str:
        return END if state["done"] else "synthesize"

    def _build_graph(self):
        g = StateGraph(_OrchState)
        g.add_node("plan", self._plan_node)
        g.add_node("execute", self._execute_node)
        g.add_node("synthesize", self._synthesize_node)
        g.add_node("reflect", self._reflect_node)
        g.add_edge(START, "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "synthesize")
        g.add_edge("synthesize", "reflect")
        g.add_conditional_edges("reflect", self._route_after_reflect,
                                {"synthesize": "synthesize", END: END})
        return g.compile()

    def _get_graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ── 对外接口 ────────────────────────────────────────────────

    def execute(self, task: str) -> str:
        """规划并执行多 Agent 协作任务，返回综合回答。"""
        logger.info("Multi-Agent 协作任务: %s", task)
        init_state: _OrchState = {
            "task": task,
            "plan": Plan(),
            "results": {},
            "draft": "",
            "reflections": 0,
            "done": False,
            "critique": "",
        }
        final = self._get_graph().invoke(
            init_state, config={"recursion_limit": MAX_REFLECTIONS * 4 + 10}
        )
        return final.get("draft", "") or "未能生成回答。"
