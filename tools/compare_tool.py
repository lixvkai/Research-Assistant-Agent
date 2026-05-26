"""论文对比分析工具 — 自动从多个维度对比两篇论文。"""

from core.llm import chat


def compare_papers(paper1_content: str, paper2_content: str, focus: str = "") -> str:
    """对比两篇论文的方法、实验、结论等。"""
    prompt = (
        "请对以下两篇论文进行详细的对比分析，从以下维度生成对比表格：\n"
        "1. 研究问题\n2. 核心方法\n3. 实验设计\n4. 主要结果\n5. 创新点\n6. 局限性\n\n"
        "请使用 Markdown 表格格式输出。\n"
    )
    if focus:
        prompt += f"\n特别关注：{focus}\n"

    prompt += f"\n---\n**论文 1：**\n{paper1_content[:3000]}\n\n---\n**论文 2：**\n{paper2_content[:3000]}"

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"对比分析出错：{e}"


TOOL_DEFINITION = {
    "name": "compare_papers",
    "description": "对比分析两篇论文，从研究问题、方法、实验、结果、创新点等维度生成对比表格。",
    "parameters": {
        "type": "object",
        "properties": {
            "paper1_content": {
                "type": "string",
                "description": "第一篇论文的内容文本",
            },
            "paper2_content": {
                "type": "string",
                "description": "第二篇论文的内容文本",
            },
            "focus": {
                "type": "string",
                "description": "可选，特别关注的对比维度",
                "default": "",
            },
        },
        "required": ["paper1_content", "paper2_content"],
    },
    "func": compare_papers,
}
