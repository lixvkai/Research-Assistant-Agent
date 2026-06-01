"""专家 Agent 基类 — 每个专家内嵌一个 ReAct Agent，按类别拥有专属工具集。"""

from core.react_agent import ReActAgent


class ExpertAgent:
    """专家 Agent 基类。"""

    name: str = "ExpertAgent"
    role: str = "通用专家"
    description: str = "通用专家 Agent"
    tool_categories: tuple[str, ...] = ()

    def __init__(self, mcp_server=None):
        self.system_prompt = self._build_system_prompt()
        # 专家级不做反思：质量审查由 Orchestrator 层统一负责，避免子任务反思放大成本。
        self._agent = ReActAgent(system_prompt=self.system_prompt, enable_reflection=False)

        if self.tool_categories:
            if mcp_server is None:
                from core.mcp import create_default_mcp_server
                mcp_server = create_default_mcp_server()
            mcp_server.bind_to_agent(self._agent, categories=self.tool_categories)

    def _build_system_prompt(self) -> str:
        return (
            f"你是{self.role}。{self.description}\n"
            "请按 ReAct 框架工作：先思考、必要时调用工具获取信息、再基于结果给出"
            "专业、简洁的回答。无需调用工具时直接作答。请用中文回答。"
        )

    def run(self, task: str, context: str = "") -> str:
        """执行任务并返回最终答案（drain ReAct 生成器）。"""
        user_msg = f"[上下文]\n{context}\n\n[任务]\n{task}" if context else task

        self._agent.reset()
        final_answer = ""
        try:
            for event in self._agent.run_iter(user_msg):
                etype = event["type"]
                if etype == "answer":
                    final_answer = event["content"]
                elif etype == "answer_token":
                    final_answer = event["partial"]
                elif etype == "error":
                    return f"[{self.name}] 执行出错：{event['content']}"
            return final_answer or "[无回答]"
        except Exception as e:
            return f"[{self.name}] 执行出错：{e}"
