"""专家 Agent 基类 — 每个专家继承此类，拥有独立的角色和工具集。"""

from core.llm import chat


class ExpertAgent:
    """专家 Agent 基类。"""

    name: str = "ExpertAgent"
    role: str = "通用专家"
    description: str = "通用专家 Agent"

    def __init__(self):
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"你是{self.role}。{self.description}\n请用中文回答，保持专业和简洁。"

    def run(self, task: str, context: str = "") -> str:
        """执行任务并返回结果。"""
        user_msg = task
        if context:
            user_msg = f"[上下文]\n{context}\n\n[任务]\n{task}"

        try:
            response = chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[{self.name}] 执行出错：{e}"
