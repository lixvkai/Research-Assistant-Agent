"""
ReAct Agent — 基于 LangGraph 的推理-行动-反思图

图结构：
    START → agent → [有 tool_calls?] ─是→ tools → agent
                          └─否→ reflect → [需修订?] ─是→ agent
                                              └─否→ END

LLM 调用复用 `core/llm`（DeepSeek，OpenAI 兼容），LangGraph 仅负责编排。
`run_iter` 对外仍 yield 原有的 dict 事件，保证 Gradio UI 零回归。
"""

import json
import logging
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from core.llm import chat
from core.schemas import (
    Action,
    Answer,
    ErrorEvent,
    Observation,
    Reflection,
    ReflectionEvent,
    StepStart,
    Thought,
)
from config.settings import MAX_REACT_STEPS, MAX_REFLECTIONS

logger = logging.getLogger(__name__)

# 反思 / 工具循环可能多次往返，给图一个宽松的步数上限
_RECURSION_LIMIT = MAX_REACT_STEPS * 2 + MAX_REFLECTIONS * 4 + 10

SYSTEM_PROMPT = """你是一个专业的科研助手 Agent。你的职责是帮助研究人员进行文献检索、论文分析、数据处理和学术写作。

你必须按照 ReAct 框架进行推理：
1. **Thought（思考）**：分析用户的需求，思考应该如何处理
2. **Action（行动）**：选择并调用合适的工具来获取信息或执行操作
3. **Observation（观察）**：分析工具返回的结果
4. 重复以上步骤，直到能给出完整的回答

注意事项：
- 每一步都要先思考，再决定是否需要调用工具
- 如果不需要工具就能回答，直接回答即可
- 回答要专业、准确、有条理
- 使用中文回答
"""

REFLECT_PROMPT = """你是一个严格的质量审查员。请评估下面这份针对用户问题的回答是否充分、准确、有条理。

只输出 JSON，格式：
{"sufficient": true/false, "critique": "若不充分，指出具体不足；若充分，可留空"}

判定 sufficient=false 的情形：回答跑题、明显遗漏关键点、逻辑混乱、或包含明显错误。
若回答已经足够好，输出 sufficient=true。"""

REFLECT_FEEDBACK = (
    "[自我反思] 你上一版的回答存在以下不足：\n{critique}\n"
    "请针对这些问题改进，给出一份更完整、准确的最终回答。"
)


class ToolRegistry:
    """工具注册表 — 管理所有可用工具的定义和实现。"""

    def __init__(self):
        self._tools: dict[str, callable] = {}
        self._schemas: list[dict] = []

    def register(self, name: str, description: str, parameters: dict, func: callable):
        self._tools[name] = func
        self._schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })

    def get_schemas(self) -> list[dict]:
        return self._schemas

    def call(self, name: str, arguments: dict):
        if name not in self._tools:
            return f"错误：未找到工具 '{name}'"
        try:
            return self._tools[name](**arguments)
        except Exception as e:
            return f"工具 '{name}' 执行出错：{e}"

    @property
    def has_tools(self) -> bool:
        return len(self._tools) > 0


class _AgentState(TypedDict):
    """ReAct 图的运行状态。"""
    messages: Annotated[list, operator.add]
    step: int
    reflections: int
    done: bool
    final_answer: str
    last_critique: str
    question: str


class ReActAgent:
    """基于 LangGraph 的 ReAct 推理-行动-反思 Agent。"""

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        memory_manager=None,
        enable_reflection: bool = True,
    ):
        self.system_prompt = system_prompt
        self.tool_registry = ToolRegistry()
        self.memory_manager = memory_manager
        self.enable_reflection = enable_reflection
        self.conversation_history: list[dict] = []
        self._graph = None
        self._init_history()

    def _init_history(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def register_tool(self, name: str, description: str, parameters: dict, func: callable):
        self.tool_registry.register(name, description, parameters, func)
        self._graph = None  # 工具变化后需重新编译

    # ── 图节点 ──────────────────────────────────────────────────

    def _agent_node(self, state: _AgentState) -> dict:
        schemas = self.tool_registry.get_schemas() if self.tool_registry.has_tools else None
        resp = chat(messages=state["messages"], tools=schemas)
        msg = resp.choices[0].message
        return {"messages": [msg.model_dump()], "step": state["step"] + 1}

    def _tools_node(self, state: _AgentState) -> dict:
        last = state["messages"][-1]
        out: list[dict] = []
        for tc in last.get("tool_calls") or []:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            result = self.tool_registry.call(fn, args)
            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            out.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
        return {"messages": out}

    def _reflect_node(self, state: _AgentState) -> dict:
        last = state["messages"][-1]
        answer = last.get("content") or ""
        # 达到最大推理步数时强制优雅收尾，不再触发反思/回环。
        if state["step"] >= MAX_REACT_STEPS:
            # 末步仍想调用工具说明是被截断 → 给明确提示；否则采用其内容。
            if last.get("tool_calls"):
                return {"done": True, "final_answer": "达到最大推理步数，未能得出完整结论。"}
            return {"done": True, "final_answer": answer or "达到最大推理步数，未能得出完整结论。"}
        if not self.enable_reflection or state["reflections"] >= MAX_REFLECTIONS:
            return {"done": True, "final_answer": answer}

        verdict = self._reflect(state["question"], answer)
        if verdict.sufficient:
            return {"done": True, "final_answer": answer, "last_critique": verdict.critique}

        feedback = {"role": "user", "content": REFLECT_FEEDBACK.format(critique=verdict.critique)}
        return {
            "messages": [feedback],
            "reflections": state["reflections"] + 1,
            "last_critique": verdict.critique,
            "done": False,
        }

    def _reflect(self, question: str, answer: str) -> Reflection:
        try:
            resp = chat(
                messages=[
                    {"role": "system", "content": REFLECT_PROMPT},
                    {"role": "user", "content": f"用户问题：\n{question}\n\n待评估回答：\n{answer}"},
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
    def _route_after_agent(state: _AgentState) -> str:
        has_tools = bool(state["messages"][-1].get("tool_calls"))
        # 超过最大步数则强制收尾，避免无限工具循环触发 GraphRecursionError。
        if has_tools and state["step"] < MAX_REACT_STEPS:
            return "tools"
        return "reflect"

    @staticmethod
    def _route_after_reflect(state: _AgentState) -> str:
        return END if state["done"] else "agent"

    def _build_graph(self):
        g = StateGraph(_AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_node("reflect", self._reflect_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route_after_agent,
                                {"tools": "tools", "reflect": "reflect"})
        g.add_edge("tools", "agent")
        g.add_conditional_edges("reflect", self._route_after_reflect,
                                {"agent": "agent", END: END})
        return g.compile()

    def _get_graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ── 对外接口 ────────────────────────────────────────────────

    def run_iter(self, user_input: str):
        """Generator that yields ReAct events for UI consumption.

        事件类型：step_start / thought / action / observation / reflection / answer / error。
        最终答案以单个 answer 事件输出；推理轨迹逐节点流式 yield。
        """
        if self.memory_manager:
            mem_ctx = self.memory_manager.get_context_for_prompt(user_input)
            enriched = (
                f"[记忆上下文]\n{mem_ctx}\n\n[用户问题]\n{user_input}"
                if mem_ctx else user_input
            )
        else:
            enriched = user_input

        self.conversation_history.append({"role": "user", "content": enriched})

        init_state: _AgentState = {
            "messages": list(self.conversation_history),
            "step": 0,
            "reflections": 0,
            "done": False,
            "final_answer": "",
            "last_critique": "",
            "question": user_input,
        }

        final_answer = ""
        current_step = 0
        try:
            for chunk in self._get_graph().stream(
                init_state,
                stream_mode="updates",
                config={"recursion_limit": _RECURSION_LIMIT},
            ):
                for node, upd in chunk.items():
                    if not upd:
                        continue
                    if node == "agent":
                        current_step = upd.get("step", current_step)
                        msg = upd["messages"][-1]
                        yield StepStart(step=current_step, max_steps=MAX_REACT_STEPS).model_dump()
                        tool_calls = msg.get("tool_calls")
                        # 达到步数上限时会被强制收尾、工具不会真正执行，故不发 action 事件，
                        # 避免出现"有 action 无 observation"的悬空轨迹。
                        if tool_calls and current_step >= MAX_REACT_STEPS:
                            tool_calls = None
                        if tool_calls:
                            if msg.get("content"):
                                yield Thought(content=msg["content"], step=current_step).model_dump()
                            for tc in tool_calls:
                                try:
                                    args = json.loads(tc["function"]["arguments"])
                                except (json.JSONDecodeError, TypeError):
                                    args = {}
                                yield Action(
                                    tool=tc["function"]["name"], args=args, step=current_step
                                ).model_dump()
                    elif node == "tools":
                        for m in upd.get("messages", []):
                            yield Observation(result=m["content"], step=current_step).model_dump()
                    elif node == "reflect":
                        if upd.get("done"):
                            final_answer = upd.get("final_answer", final_answer)
                        elif upd.get("last_critique"):
                            yield ReflectionEvent(
                                sufficient=False,
                                critique=upd["last_critique"],
                                step=current_step,
                            ).model_dump()
        except Exception as e:
            logger.exception("graph run failed")
            yield ErrorEvent(content=str(e)).model_dump()
            return

        final_answer = final_answer or "未能生成回答。"
        self.conversation_history.append({"role": "assistant", "content": final_answer})
        if self.memory_manager:
            self.memory_manager.record_interaction(user_input, final_answer)
        yield Answer(content=final_answer).model_dump()

    def reset(self):
        """重置对话历史。"""
        self._init_history()
