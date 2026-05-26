"""专家 Agent 定义 — 各领域的专业智能体。"""

from agents.base_agent import ExpertAgent


class LiteratureAgent(ExpertAgent):
    name = "LiteratureAgent"
    role = "学术文献检索与分析专家"
    description = (
        "擅长搜索、阅读和分析学术论文。"
        "能够从论文中提取关键信息，进行文献综述，比较不同论文的方法和结论。"
    )


class DataAnalysisAgent(ExpertAgent):
    name = "DataAnalysisAgent"
    role = "数据分析与可视化专家"
    description = (
        "擅长数据处理、统计分析和可视化。"
        "能够设计实验方案，分析实验数据，提出数据驱动的洞察。"
    )


class WritingAgent(ExpertAgent):
    name = "WritingAgent"
    role = "学术写作专家"
    description = (
        "擅长学术论文写作和润色。"
        "能够撰写论文各部分（摘要、引言、方法、结论等），确保逻辑清晰、语言规范。"
    )


class ReviewAgent(ExpertAgent):
    name = "ReviewAgent"
    role = "质量审查与评审专家"
    description = (
        "擅长审查研究方案和论文质量。"
        "能够从方法论、逻辑、创新性等角度给出建设性反馈。"
    )


EXPERT_REGISTRY: dict[str, type[ExpertAgent]] = {
    "literature": LiteratureAgent,
    "data_analysis": DataAnalysisAgent,
    "writing": WritingAgent,
    "review": ReviewAgent,
}
