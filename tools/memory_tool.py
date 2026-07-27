"""记忆工具 — 让 Agent 可以主动存取记忆。"""

from memory.memory_store import get_memory_manager


def save_research_finding(content: str, importance: float = 0.7) -> str:
    """保存一个研究发现到长期记忆。"""
    mgr = get_memory_manager()
    mgr.save_finding(content, importance)
    return f"已保存研究发现：{content[:100]}..."


def save_user_preference(content: str) -> str:
    """保存用户研究偏好。"""
    mgr = get_memory_manager()
    mgr.save_preference(content)
    return f"已记录用户偏好：{content}"


def recall_memories(keyword: str) -> str:
    """搜索相关记忆。"""
    mgr = get_memory_manager()
    results = mgr.long_term.search(keyword, limit=5)
    if not results:
        return f"未找到与 '{keyword}' 相关的记忆。"
    lines = [f"- [{m.category} | 重要度:{m.importance}] {m.content}" for m in results]
    return f"找到 {len(results)} 条相关记忆：\n" + "\n".join(lines)


def get_recent_memories() -> str:
    """获取最近的记忆。"""
    mgr = get_memory_manager()
    results = mgr.long_term.get_recent(limit=10)
    if not results:
        return "暂无历史记忆。"
    lines = [f"- [{m.category}] {m.content} ({m.timestamp[:10]})" for m in results]
    return "最近的记忆：\n" + "\n".join(lines)


TOOL_DEFINITIONS = [
    {
        "name": "save_research_finding",
        "description": "保存重要的研究发现到长期记忆中，以便将来参考。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "要保存的研究发现内容"},
                "importance": {"type": "number", "description": "重要程度 0-1，默认 0.7", "default": 0.7},
            },
            "required": ["content"],
        },
        "func": save_research_finding,
    },
    {
        "name": "save_user_preference",
        "description": "记录用户的研究偏好（如研究领域、关注方向等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "用户偏好描述"},
            },
            "required": ["content"],
        },
        "func": save_user_preference,
    },
    {
        "name": "recall_memories",
        "description": "从长期记忆中检索与关键词相关的历史记忆。",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["keyword"],
        },
        "func": recall_memories,
    },
    {
        "name": "get_recent_memories",
        "description": "查看最近保存的记忆。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
        "func": get_recent_memories,
    },
]
