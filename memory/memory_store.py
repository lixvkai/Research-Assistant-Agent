"""
Memory 记忆系统

- 短期记忆：当前会话的对话摘要（滑动窗口 + 摘要压缩）
- 长期记忆：跨会话的用户偏好、研究历史（SQLite 持久化）
"""

import sqlite3
import datetime
from dataclasses import dataclass
from config.settings import MEMORY_DB_PATH
from core.llm import chat


@dataclass
class MemoryEntry:
    """一条记忆记录。"""
    id: int | None
    category: str          # "preference" | "research_topic" | "interaction" | "finding"
    content: str
    timestamp: str
    importance: float      # 0.0 ~ 1.0


class ShortTermMemory:
    """短期记忆 — 管理当前会话的上下文窗口。"""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: list[dict] = []
        self.summary: str = ""

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content, "time": datetime.datetime.now().isoformat()})
        if len(self.turns) > self.max_turns:
            self._compress()

    def _compress(self):
        """将前半部分对话压缩为摘要，保留最近的对话。"""
        half = len(self.turns) // 2
        old_turns = self.turns[:half]
        self.turns = self.turns[half:]

        old_text = "\n".join(f"[{t['role']}]: {t['content'][:200]}" for t in old_turns)
        prompt = f"请用中文简洁地总结以下对话的关键信息（不超过200字）：\n\n{old_text}"

        if self.summary:
            prompt = f"已有摘要：{self.summary}\n\n新增对话：\n{old_text}\n\n请合并生成新摘要（不超过300字）。"

        try:
            response = chat(messages=[{"role": "user", "content": prompt}], temperature=0.2)
            self.summary = response.choices[0].message.content
        except Exception:
            self.summary += f"\n[自动摘要失败，保留原始记录片段] {old_text[:200]}"

    def get_context(self) -> str:
        """获取当前记忆上下文，用于注入 Prompt。"""
        parts = []
        if self.summary:
            parts.append(f"[对话历史摘要]\n{self.summary}")
        return "\n".join(parts)


class LongTermMemory:
    """长期记忆 — SQLite 持久化存储。"""

    def __init__(self, db_path: str = MEMORY_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL DEFAULT 0.5
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")

    def save(self, entry: MemoryEntry) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO memories (category, content, timestamp, importance) VALUES (?, ?, ?, ?)",
                (entry.category, entry.content, entry.timestamp, entry.importance),
            )
            return cursor.lastrowid

    def search(self, keyword: str, category: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        query = "SELECT id, category, content, timestamp, importance FROM memories WHERE content LIKE ?"
        params: list = [f"%{keyword}%"]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [MemoryEntry(id=r[0], category=r[1], content=r[2], timestamp=r[3], importance=r[4]) for r in rows]

    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, category, content, timestamp, importance FROM memories ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [MemoryEntry(id=r[0], category=r[1], content=r[2], timestamp=r[3], importance=r[4]) for r in rows]

    def get_by_category(self, category: str, limit: int = 20) -> list[MemoryEntry]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, category, content, timestamp, importance FROM memories WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        return [MemoryEntry(id=r[0], category=r[1], content=r[2], timestamp=r[3], importance=r[4]) for r in rows]


_default_manager: "MemoryManager | None" = None


def get_memory_manager() -> "MemoryManager":
    """进程级 MemoryManager 单例 — UI 与工具层共用同一份记忆。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = MemoryManager()
    return _default_manager


class MemoryManager:
    """记忆管理器 — 统一管理短期和长期记忆。"""

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def record_interaction(self, user_input: str, agent_response: str):
        """记录一轮交互。"""
        self.short_term.add_turn("user", user_input)
        self.short_term.add_turn("assistant", agent_response)

    def save_finding(self, content: str, importance: float = 0.7):
        """保存一个研究发现。"""
        self.long_term.save(MemoryEntry(
            id=None,
            category="finding",
            content=content,
            timestamp=datetime.datetime.now().isoformat(),
            importance=importance,
        ))

    def save_preference(self, content: str):
        """保存用户偏好。"""
        self.long_term.save(MemoryEntry(
            id=None,
            category="preference",
            content=content,
            timestamp=datetime.datetime.now().isoformat(),
            importance=0.8,
        ))

    def get_context_for_prompt(self, current_query: str = "") -> str:
        """为 Agent Prompt 生成记忆上下文。"""
        parts = []

        # 短期记忆
        short = self.short_term.get_context()
        if short:
            parts.append(short)

        # 长期记忆 — 相关记忆
        if current_query:
            relevant = self.long_term.search(current_query, limit=5)
            if relevant:
                memories = "\n".join(f"- [{m.category}] {m.content}" for m in relevant)
                parts.append(f"[相关历史记忆]\n{memories}")

        # 用户偏好
        prefs = self.long_term.get_by_category("preference", limit=5)
        if prefs:
            pref_text = "\n".join(f"- {p.content}" for p in prefs)
            parts.append(f"[用户偏好]\n{pref_text}")

        return "\n\n".join(parts) if parts else ""
