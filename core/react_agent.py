"""
ReAct Agent — 基于 LangGraph 的推理-行动-反思图

图结构：
    START → prepare → agent → [有 tool_calls?] ─是→ tools → agent
                                   └─否→ reflect → [需修订?] ─是→ agent
                                            └─否→ finalize → END

设计要点：
- **会话状态交给 checkpointer**：按 `thread_id`（=会话 id）隔离消息历史，
  每轮只把新的用户消息喂进图，历史由 LangGraph 维护。这样多会话天然隔离，
  且支持从数据库回灌历史（`load_history`）让 Agent「记得」被切换回来的对话。
- **上下文窗口有界**：`prepare` 节点裁剪超窗消息，溢出部分交短期记忆压缩成摘要，
  避免「全量历史 + 摘要」双份注入。
- **记忆上下文不落历史**：system prompt 与记忆上下文在调用 LLM 时临时拼装，
  不写入 checkpoint，避免每轮堆叠过期的记忆快照。
- **finalize 与 reflect 分离**：反思只管质量判定，收尾统一在 finalize。
- **反思预算与工具预算分离**：每次修订通过 `step_budget` 追加配额，互不侵占。
"""

import json
import logging
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from config.settings import (
    CONTEXT_TRIM_TRIGGER,
    MAX_CONTEXT_MESSAGES,
    MAX_REACT_STEPS,
    MAX_REFLECTIONS,
    REFLECTION_STEP_BONUS,
    TOOL_MAX_WORKERS,
)
from core.budget import BudgetExceeded, run_budget
from core.llm import chat
from core.observability import (
    TraceScope,
    bind_trace_scope,
    capture_value,
    observe_operation,
)
from core.parallel import run_in_parallel
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

logger = logging.getLogger(__name__)

# 反思 / 工具循环可能多次往返，给图一个宽松的步数上限
_RECURSION_LIMIT = MAX_REACT_STEPS * 2 + MAX_REFLECTIONS * (REFLECTION_STEP_BONUS + 4) + 10

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

_CAPPED_ANSWER = "达到最大推理步数，未能得出完整结论。"


class ToolRegistry:
    """工具注册表 — 管理所有可用工具的定义和实现。"""

    def __init__(self):
        self._tools: dict[str, Callable[..., Any]] = {}
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, description: str, parameters: dict,
                 func: Callable[..., Any]) -> None:
        """按名称注册（重复注册同名工具为覆盖，不会产生重复的 function schema）。"""
        self._tools[name] = func
        self._schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    def get_schemas(self) -> list[dict]:
        return list(self._schemas.values())

    def call(self, name: str, arguments: dict):
        observation_type = "agent" if name == "multi_agent_collaborate" else "tool"
        with observe_operation(
            name,
            as_type=observation_type,
            input=arguments,
            metadata={"tool_name": name},
        ) as observation:
            if name not in self._tools:
                result = f"错误：未找到工具 '{name}'"
                if observation is not None:
                    observation.update(
                        output=result, level="ERROR", status_message="tool not found"
                    )
                return result
            try:
                result = self._tools[name](**arguments)
                if observation is not None:
                    observation.update(output=capture_value(result))
                return result
            except Exception as e:
                logger.exception("工具 '%s' 执行出错", name)
                result = f"工具 '{name}' 执行出错：{e}"
                if observation is not None:
                    observation.update(
                        output=capture_value(result),
                        level="ERROR",
                        status_message=type(e).__name__,
                    )
                return result

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


class _Replace(list):
    """消息 reducer 的替换标记：返回本类型表示整体替换而非追加。"""


def _merge_messages(left: list, right: list) -> list:
    if isinstance(right, _Replace):
        return list(right)
    return list(left) + list(right)


def trim_messages(
    messages: list[dict],
    limit: int | None = None,
    trigger: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """高低水位裁剪，返回 (保留, 溢出)。

    超过 `trigger`（高水位）才裁剪，一裁就裁到 `limit`（低水位）。两者留出的差值
    是缓冲区：如果裁剪目标等于触发线，窗口会永远贴着阈值，每轮都要重新压缩一次摘要。

    切点会向后推进，避免留下没有对应 assistant.tool_calls 的孤立 tool 消息
    （那会被 OpenAI 兼容接口判为非法请求）。
    """
    if limit is None:
        limit = MAX_CONTEXT_MESSAGES
    if trigger is None:
        trigger = CONTEXT_TRIM_TRIGGER
    # 配置反了（触发线低于裁剪目标）时退化为「触发即裁到目标」，不至于每轮空转
    trigger = max(trigger, limit)

    if limit <= 0 or len(messages) <= trigger:
        return list(messages), []

    cut = len(messages) - limit
    while cut < len(messages) and messages[cut].get("role") == "tool":
        cut += 1
    return list(messages[cut:]), list(messages[:cut])


class _AgentState(TypedDict):
    """ReAct 图的运行状态（messages 由 checkpointer 跨轮维护）。"""
    messages: Annotated[list, _merge_messages]
    step: int
    step_budget: int
    reflections: int
    done: bool
    final_answer: str
    last_critique: str
    question: str
    memory_context: str
    session: str


class ReActAgent:
    """基于 LangGraph 的 ReAct 推理-行动-反思 Agent。"""

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        memory_manager=None,
        enable_reflection: bool = True,
        session_id: str | None = None,
    ):
        self.system_prompt = system_prompt
        self.tool_registry = ToolRegistry()
        self.memory_manager = memory_manager
        self.enable_reflection = enable_reflection
        self.session_id = session_id
        self._graph = None
        self._checkpointer = None

    # ── 工具 ────────────────────────────────────────────────────

    def register_tool(self, name: str, description: str, parameters: dict,
                      func: Callable[..., Any]) -> None:
        had = name in self.tool_registry
        self.tool_registry.register(name, description, parameters, func)
        if not had:
            self._graph = None  # 工具集合变化后需重新编译

    # ── 会话状态（checkpointer） ────────────────────────────────

    def _get_checkpointer(self):
        if self._checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver

            self._checkpointer = MemorySaver()
        return self._checkpointer

    def _resolve_session(self, session_id: str | None = None) -> str:
        """统一的会话键 —— 同时用作 LangGraph thread_id 与记忆作用域键。"""
        return str(session_id if session_id is not None else (self.session_id or "default"))

    def _thread_config(self, session_id: str | None = None) -> dict:
        return {
            "configurable": {"thread_id": self._resolve_session(session_id)},
            "recursion_limit": _RECURSION_LIMIT,
        }

    def get_messages(self, session_id: str | None = None) -> list[dict]:
        """读取某会话已累积的对话消息。"""
        try:
            snapshot = self._get_graph().get_state(self._thread_config(session_id))
        except Exception:
            return []
        values = getattr(snapshot, "values", None) or {}
        return list(values.get("messages") or [])

    @property
    def conversation_history(self) -> list[dict]:
        """当前会话的完整消息（含开头的 system prompt），供 UI / 导出使用。"""
        return [{"role": "system", "content": self.system_prompt}] + self.get_messages()

    def _write_messages(self, messages: list[dict], session_id: str | None = None) -> None:
        self._get_graph().update_state(
            self._thread_config(session_id), {"messages": _Replace(messages)}
        )

    def load_history(self, messages: list[dict], session_id: str | None = None) -> None:
        """把持久化的历史回灌到某会话，使 Agent「记得」被切换回来的对话。"""
        clean = [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in messages
            if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
        ]
        kept, _ = trim_messages(clean)
        self._write_messages(kept, session_id)

    def reset(self, session_id: str | None = None) -> None:
        """清空某会话的对话历史与短期记忆。"""
        self._write_messages([], session_id)
        if self.memory_manager is not None:
            self.memory_manager.reset_session(self._resolve_session(session_id))

    # ── 图节点 ──────────────────────────────────────────────────

    def _prepare_node(self, state: _AgentState) -> dict:
        """裁剪上下文窗口，把溢出消息交给短期记忆压缩。"""
        kept, overflow = trim_messages(state["messages"])
        if not overflow:
            return {}

        updates: dict = {"messages": _Replace(kept)}
        if self.memory_manager is not None:
            session = state.get("session")
            try:
                self.memory_manager.absorb_overflow(overflow, session)
                # 摘要是在本节点才更新的，而 memory_context 在进图前就取好了。
                # 不刷新的话，这一轮刚被移出窗口的消息既不在 messages 里、也不在
                # 摘要里，会出现「上一轮说的事这轮忘了、下轮又想起来」的空窗。
                updates["memory_context"] = self.memory_manager.get_context_for_prompt(
                    state.get("question", ""), session
                )
            except Exception as e:
                logger.warning("压缩溢出上下文失败（忽略）：%s", e)
        logger.info("上下文窗口裁剪：移出 %d 条消息", len(overflow))
        return updates

    def _llm_messages(self, state: _AgentState) -> list[dict]:
        """拼装本次调用的消息：system + 记忆上下文（临时） + 会话消息。"""
        messages = [{"role": "system", "content": self.system_prompt}]
        memory_context = state.get("memory_context") or ""
        if memory_context:
            messages.append({"role": "system", "content": f"[记忆上下文]\n{memory_context}"})
        messages.extend(state["messages"])
        return messages

    def _agent_node(self, state: _AgentState) -> dict:
        schemas = self.tool_registry.get_schemas() if self.tool_registry.has_tools else None
        resp = chat(
            messages=self._llm_messages(state),
            tools=schemas,
            observation_name="react-agent",
        )
        msg = resp.choices[0].message
        return {"messages": [msg.model_dump()], "step": state["step"] + 1}

    def _tools_node(self, state: _AgentState) -> dict:
        last = state["messages"][-1]
        tool_calls = last.get("tool_calls") or []

        def _invoke(tc: dict) -> dict:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            result = self.tool_registry.call(fn, args)
            result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            return {"role": "tool", "tool_call_id": tc["id"], "content": result_str}

        # 同一轮内的多个 tool_call 相互独立，可并行执行
        out = run_in_parallel(_invoke, tool_calls, max_workers=TOOL_MAX_WORKERS)
        return {"messages": out}

    def _reflect_node(self, state: _AgentState) -> dict:
        """只负责质量判定：不达标则追加批评与步数配额，回到 agent 重做。"""
        if not self.enable_reflection or state["reflections"] >= MAX_REFLECTIONS:
            return {"done": True}

        answer = (state["messages"][-1].get("content") or "")
        verdict = self._reflect(state["question"], answer)
        if verdict.sufficient:
            return {"done": True, "last_critique": verdict.critique}

        feedback = {"role": "user", "content": REFLECT_FEEDBACK.format(critique=verdict.critique)}
        return {
            "messages": [feedback],
            "reflections": state["reflections"] + 1,
            # 修订不占用工具循环的步数预算
            "step_budget": state["step_budget"] + REFLECTION_STEP_BONUS,
            "last_critique": verdict.critique,
            "done": False,
        }

    def _finalize_node(self, state: _AgentState) -> dict:
        """统一收尾：产出最终答案（含步数封顶时的优雅提示）。"""
        last = state["messages"][-1]
        answer = (last.get("content") or "").strip()
        capped = state["step"] >= state["step_budget"]
        if capped and (last.get("tool_calls") or not answer):
            return {"done": True, "final_answer": _CAPPED_ANSWER}
        return {"done": True, "final_answer": answer or "未能生成回答。"}

    def _reflect(self, question: str, answer: str) -> Reflection:
        try:
            resp = chat(
                messages=[
                    {"role": "system", "content": REFLECT_PROMPT},
                    {"role": "user", "content": f"用户问题：\n{question}\n\n待评估回答：\n{answer}"},
                ],
                temperature=0.2,
                observation_name="react-reflection",
            )
            content = resp.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}") + 1
            if start == -1 or end <= start:
                return Reflection()
            return Reflection.model_validate_json(content[start:end])
        except BudgetExceeded:
            raise
        except Exception as e:
            logger.warning("反思解析失败，默认通过：%s", e)
            return Reflection()

    # ── 路由 ────────────────────────────────────────────────────

    def _route_after_agent(self, state: _AgentState) -> str:
        has_tools = bool(state["messages"][-1].get("tool_calls"))
        if has_tools and state["step"] < state["step_budget"]:
            return "tools"
        # 超预算强制收尾，避免无限工具循环触发 GraphRecursionError
        if state["step"] >= state["step_budget"] or not self.enable_reflection:
            return "finalize"
        return "reflect"

    @staticmethod
    def _route_after_reflect(state: _AgentState) -> str:
        return "finalize" if state["done"] else "agent"

    def _build_graph(self):
        g = StateGraph(_AgentState)
        g.add_node("prepare", self._prepare_node)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.add_node("reflect", self._reflect_node)
        g.add_node("finalize", self._finalize_node)

        g.add_edge(START, "prepare")
        g.add_edge("prepare", "agent")
        g.add_conditional_edges("agent", self._route_after_agent,
                                {"tools": "tools", "reflect": "reflect", "finalize": "finalize"})
        g.add_edge("tools", "agent")
        g.add_conditional_edges("reflect", self._route_after_reflect,
                                {"agent": "agent", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile(checkpointer=self._get_checkpointer())

    def _get_graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph

    # ── 对外接口 ────────────────────────────────────────────────

    def run_iter(
        self,
        user_input: str,
        session_id: str | None = None,
        trace_scope: TraceScope | None = None,
    ):
        """Generator that yields ReAct events for UI consumption.

        事件类型：step_start / thought / action / observation / reflection / answer / error。
        最终答案以单个 answer 事件输出；推理轨迹逐节点流式 yield。
        """
        session = self._resolve_session(session_id)
        memory_context = ""
        if self.memory_manager is not None:
            try:
                memory_context = self.memory_manager.get_context_for_prompt(user_input, session)
            except Exception as e:
                logger.warning("获取记忆上下文失败（忽略）：%s", e)

        # 只把新的用户消息喂进图，历史由 checkpointer 维护
        init_state = {
            "messages": [{"role": "user", "content": user_input}],
            "step": 0,
            "step_budget": MAX_REACT_STEPS,
            "reflections": 0,
            "done": False,
            "final_answer": "",
            "last_critique": "",
            "question": user_input,
            "memory_context": memory_context,
            "session": session,
        }

        final_answer = ""
        current_step = 0
        step_budget = MAX_REACT_STEPS
        try:
            # Gradio 可能在不同 contextvars.Context 中逐次推进同步生成器。
            # ContextVar 的 token 不能跨 Context reset，因此预算作用域不能跨 yield。
            # 保留同一个预算对象，但只在每次 next(graph_stream) 期间临时绑定。
            with run_budget() as request_budget:
                pass
            graph_stream = self._get_graph().stream(
                init_state,
                stream_mode="updates",
                config=self._thread_config(session_id),
            )
            stream_done = object()
            observation_names = {
                "prepare": "react.prepare-context",
                "agent": "react.generate-response",
                "tools": "react.execute-tools",
                "reflect": "react.evaluate-response",
                "finalize": "react.finalize-response",
            }
            while True:
                with run_budget(request_budget), bind_trace_scope(trace_scope):
                    with observe_operation(
                        "react-graph-step",
                        as_type="chain",
                        metadata={"react_step": current_step},
                    ) as graph_observation:
                        chunk = next(graph_stream, stream_done)
                        if chunk is stream_done:
                            if graph_observation is not None:
                                graph_observation.update(name="react.complete")
                        elif graph_observation is not None:
                            node_names = list(chunk)
                            stable_names = [
                                observation_names.get(node, f"react.{node}")
                                for node in node_names
                            ]
                            graph_observation.update(
                                name="+".join(stable_names),
                                output={"nodes": node_names},
                            )
                if chunk is stream_done:
                    break

                terminal_chunk = "finalize" in chunk
                for node, upd in chunk.items():
                    if not upd:
                        continue
                    if node == "agent":
                        current_step = upd.get("step", current_step)
                        msg = upd["messages"][-1]
                        yield StepStart(step=current_step, max_steps=step_budget).model_dump()
                        tool_calls = msg.get("tool_calls")
                        # 超预算时工具不会真正执行，故不发 action 事件，
                        # 避免出现"有 action 无 observation"的悬空轨迹。
                        if tool_calls and current_step >= step_budget:
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
                        step_budget = upd.get("step_budget", step_budget)
                        if not upd.get("done") and upd.get("last_critique"):
                            yield ReflectionEvent(
                                sufficient=False,
                                critique=upd["last_critique"],
                                step=current_step,
                            ).model_dump()
                    elif node == "finalize":
                        final_answer = upd.get("final_answer", final_answer)
                # finalize 直接连到 END；无需额外 next() 来发现流已耗尽，
                # 否则会产生一个没有业务工作的 react.complete 空 observation。
                if terminal_chunk:
                    break
        except BudgetExceeded as e:
            logger.warning("运行预算耗尽：%s", e)
            final_answer = final_answer or f"本次请求已达资源上限（{e}），以下是已获得的部分结论。"
        except Exception as e:
            logger.exception("graph run failed")
            yield ErrorEvent(content=str(e)).model_dump()
            return

        final_answer = final_answer or "未能生成回答。"
        if self.memory_manager is not None:
            self.memory_manager.record_interaction(user_input, final_answer, session)
        yield Answer(content=final_answer).model_dump()
