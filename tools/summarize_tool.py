"""文本摘要工具 — 用 LLM 生成摘要。"""

from core.llm import chat


def summarize_text(text: str, focus: str = "") -> str:
    """利用 LLM 对长文本进行摘要。"""
    prompt = "请对以下文本生成一份简洁的学术摘要（中文），保留关键信息和数据。"
    if focus:
        prompt += f"\n重点关注：{focus}"
    prompt += f"\n\n---\n{text[:4000]}"

    try:
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"摘要生成出错：{e}"


TOOL_DEFINITION = {
    "name": "summarize_text",
    "description": "对长文本生成学术摘要。可指定关注重点，Agent 会用 LLM 提炼核心内容。",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "需要摘要的长文本内容",
            },
            "focus": {
                "type": "string",
                "description": "可选，指定关注的重点方向，例如 '方法论' 或 '实验结果'",
                "default": "",
            },
        },
        "required": ["text"],
    },
    "func": summarize_text,
}
