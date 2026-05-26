"""
ReAct Agent — 核心推理-行动循环

实现 Thought → Action → Observation → Thought 循环，
通过 DeepSeek 的 function calling 能力驱动工具调用。
"""

import json
import logging

from core.llm import chat, chat_stream
from config.settings import MAX_REACT_STEPS

logger = logging.getLogger(__name__)

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


class ReActAgent:
    """ReAct 推理-行动 Agent。"""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, memory_manager=None):
        self.system_prompt = system_prompt
        self.tool_registry = ToolRegistry()
        self.memory_manager = memory_manager
        self.conversation_history: list[dict] = []
        self._init_history()

    def _init_history(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def register_tool(self, name: str, description: str, parameters: dict, func: callable):
        self.tool_registry.register(name, description, parameters, func)

    def run_iter(self, user_input: str):
        """Generator that yields ReAct events for UI consumption.

        Event types:
            step_start   — a new reasoning step begins
            thought      — the model's chain-of-thought (if any)
            action       — a tool call is about to execute
            observation  — the tool's return value
            answer_token — a single streamed token of the final answer
            answer       — the final complete response
            error        — something went wrong
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

        for step in range(1, MAX_REACT_STEPS + 1):
            yield {"type": "step_start", "step": step, "max_steps": MAX_REACT_STEPS}

            try:
                tools = self.tool_registry.get_schemas() if self.tool_registry.has_tools else None
                response = chat(messages=self.conversation_history, tools=tools)
                message = response.choices[0].message
            except Exception as e:
                logger.exception("LLM call failed")
                yield {"type": "error", "content": str(e)}
                return

            if message.tool_calls:
                if message.content:
                    yield {"type": "thought", "content": message.content, "step": step}

                self.conversation_history.append(message.model_dump())

                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)

                    yield {"type": "action", "tool": func_name, "args": func_args, "step": step}

                    result = self.tool_registry.call(func_name, func_args)
                    result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

                    yield {"type": "observation", "result": result_str, "step": step}

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    })
                continue

            # Stream the final answer token-by-token
            if message.content:
                self.conversation_history.append({"role": "assistant", "content": message.content})
                if self.memory_manager:
                    self.memory_manager.record_interaction(user_input, message.content)
                yield {"type": "answer", "content": message.content}
                return

            collected = ""
            for token in chat_stream(messages=self.conversation_history):
                collected += token
                yield {"type": "answer_token", "token": token, "partial": collected}

            self.conversation_history.append({"role": "assistant", "content": collected})
            if self.memory_manager:
                self.memory_manager.record_interaction(user_input, collected)
            yield {"type": "answer", "content": collected}
            return

        yield {"type": "answer", "content": "达到最大推理步数，未能得出结论。"}

    def reset(self):
        """重置对话历史。"""
        self._init_history()
