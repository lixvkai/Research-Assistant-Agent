"""MemoryManager 单例：UI 与工具层必须共用同一实例。"""

from memory.memory_store import get_memory_manager
from tools import memory_tool


def test_memory_tool_uses_shared_singleton():
    mgr = get_memory_manager()
    # tools 层写入的记忆，单例侧必须能立刻搜到
    assert memory_tool.save_research_finding("singleton-check-finding", 0.5).startswith("已保存")
    hits = mgr.long_term.search("singleton-check-finding", limit=3)
    assert any("singleton-check-finding" in m.content for m in hits)
    assert get_memory_manager() is mgr
