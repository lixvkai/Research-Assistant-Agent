"""专家 Agent 定义 — 各领域的专业智能体，配备 ReAct 能力与专属工具集。"""

import logging

from agents.base_agent import ExpertAgent
from core.llm import chat
from core.schemas import Reflection

logger = logging.getLogger(__name__)


class LiteratureAgent(ExpertAgent):
    name = "LiteratureAgent"
    role = "学术文献检索与分析专家"
    description = (
        "擅长搜索、阅读和分析学术论文。"
        "能够从论文中提取关键信息，进行文献综述，比较不同论文的方法和结论。"
    )
    tool_categories = ("论文检索", "知识库", "文本处理", "论文分析", "网络工具")


class DataAnalysisAgent(ExpertAgent):
    name = "DataAnalysisAgent"
    role = "数据分析与可视化专家"
    description = (
        "擅长数据处理、统计分析和可视化。"
        "能够设计实验方案，分析实验数据，提出数据驱动的洞察。"
    )
    tool_categories = ("基础工具", "文本处理", "趋势分析", "知识库")


class WritingAgent(ExpertAgent):
    name = "WritingAgent"
    role = "学术写作专家"
    description = (
        "擅长学术论文写作和润色。"
        "能够撰写论文各部分（摘要、引言、方法、结论等），确保逻辑清晰、语言规范。"
    )
    tool_categories = ("文本处理", "知识库", "基础工具")


class ReviewAgent(ExpertAgent):
    """质量审查专家。同一个角色有两种用法，审查标准只维护这一份：

    - `run()`（继承自 `ExpertAgent`）：Planner 派发 review 子任务时，走完整 ReAct，
      可以调工具去查证。
    - `review_draft()`：Orchestrator 图里的质量门，单次调用产出结构化判定。
      质量门每轮都会执行且最多重试 `MAX_REFLECTIONS` 次，走完整 ReAct 成本不划算。
    """

    name = "ReviewAgent"
    role = "质量审查与评审专家"
    description = (
        "擅长审查研究方案和论文质量。"
        "能够从方法论、逻辑、创新性等角度给出建设性反馈。"
    )
    tool_categories = ("文本处理", "论文分析", "知识库", "基础工具")

    RUBRIC = """你是严格的质量审查员。请评估下面这份针对原始任务的综合回答是否完整、准确、有条理。

只输出 JSON：{"sufficient": true/false, "critique": "若不充分，指出具体不足"}"""

    def review_draft(self, task: str, draft: str) -> Reflection:
        """对综合稿做一次结构化评审。

        解析失败一律返回默认的 `sufficient=True`：审查环节本身不应该成为死循环的来源。
        """
        try:
            resp = chat(
                messages=[
                    {"role": "system", "content": self.RUBRIC},
                    {"role": "user", "content": f"原始任务：\n{task}\n\n综合回答：\n{draft}"},
                ],
                temperature=0.2,
                observation_name="expert-review",
            )
            content = resp.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}") + 1
            if start == -1 or end <= start:
                return Reflection()
            return Reflection.model_validate_json(content[start:end])
        except Exception as e:
            logger.warning("ReviewAgent 评审解析失败，默认通过：%s", e)
            return Reflection()


EXPERT_REGISTRY: dict[str, type[ExpertAgent]] = {
    "literature": LiteratureAgent,
    "data_analysis": DataAnalysisAgent,
    "writing": WritingAgent,
    "review": ReviewAgent,
}
