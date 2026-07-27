"""
Memory 记忆系统

职责划分（重构后）：
- ShortTermMemory：**只**负责被 Agent 上下文窗口挤出的历史，压缩为摘要。
  窗口本身由 Agent 持有，二者不再各存一份对话，避免「全量历史 + 摘要」双份注入。
- LongTermMemory：跨会话的偏好 / 发现（SQLite 持久化），检索走分词打分或语义向量，
  不再用「整句 LIKE」（中文长句几乎不可能子串命中）。
- MemoryManager：统一入口，并提供会话结束时的记忆固化（consolidate）。

会话作用域：短期记忆摘要与待固化的对话缓冲按 `session_id` 分桶存放，
长期记忆则是全局共享的（跨会话记住用户偏好本就是它的职责）。
这样同一进程并发服务多个会话时，A 会话的摘要不会漏进 B 会话的 prompt。
"""

import datetime
import difflib
import logging
import sqlite3
import threading
from dataclasses import dataclass

from config.settings import (
    MEMORY_DB_PATH,
    MEMORY_DEDUP_THRESHOLD,
    MEMORY_SEMANTIC_SEARCH,
)
from core.llm import chat
from utils.text import keywords, normalize, tokenize

logger = logging.getLogger(__name__)

MEMORY_CATEGORIES = ("preference", "research_topic", "interaction", "finding")

_MAX_FALLBACK_SUMMARY = 1200


@dataclass
class MemoryEntry:
    """一条记忆记录。"""
    id: int | None
    category: str          # 见 MEMORY_CATEGORIES
    content: str
    timestamp: str
    importance: float      # 0.0 ~ 1.0
    session_id: str | None = None


class ShortTermMemory:
    """短期记忆 — 维护「已被移出上下文窗口」的对话摘要。"""

    def __init__(self):
        self.summary: str = ""

    def absorb(self, messages: list[dict]) -> None:
        """接收被窗口挤出的消息，合并压缩进摘要。"""
        texts = []
        for m in messages:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            texts.append(f"[{m.get('role', '?')}]: {content[:300]}")
        if not texts:
            return

        old_text = "\n".join(texts)
        if self.summary:
            prompt = (
                f"已有摘要：{self.summary}\n\n新增对话：\n{old_text}\n\n"
                "请合并生成新摘要（不超过300字）。"
            )
        else:
            prompt = f"请用中文简洁地总结以下对话的关键信息（不超过200字）：\n\n{old_text}"

        try:
            response = chat(messages=[{"role": "user", "content": prompt}], temperature=0.2)
            self.summary = (response.choices[0].message.content or "").strip() or self.summary
        except Exception as e:
            logger.warning("短期记忆摘要失败，保留截断片段：%s", e)
            merged = f"{self.summary}\n[摘要失败片段] {old_text}".strip()
            self.summary = merged[-_MAX_FALLBACK_SUMMARY:]

    def get_context(self) -> str:
        """获取摘要上下文，用于注入 Prompt。"""
        if not self.summary:
            return ""
        return f"[对话历史摘要]\n{self.summary}"

    def reset(self) -> None:
        self.summary = ""


class LongTermMemory:
    """长期记忆 — SQLite 持久化存储。"""

    def __init__(self, db_path: str = MEMORY_DB_PATH, semantic: bool | None = None):
        self.db_path = db_path
        self._semantic_enabled = MEMORY_SEMANTIC_SEARCH if semantic is None else semantic
        self._semantic = None
        self._lock = threading.Lock()
        self._init_db()

    # ── schema ────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        # WAL：允许读写并发，降低多会话下的 database is locked
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        import os

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        with self._connect() as conn:
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
            # 轻量迁移：为将来的多用户/会话隔离预留 session_id
            cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()}
            if "session_id" not in cols:
                conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT")

    # ── 语义索引（可选） ──────────────────────────────────────

    def _get_semantic(self):
        """懒加载向量索引；任何失败都降级为关键词检索。"""
        if not self._semantic_enabled:
            return None
        if self._semantic is None:
            try:
                from memory.semantic_index import SemanticMemoryIndex

                self._semantic = SemanticMemoryIndex()
            except Exception as e:
                logger.warning("记忆语义索引不可用，降级为关键词检索：%s", e)
                self._semantic_enabled = False
                return None
        return self._semantic

    # ── 写入 ──────────────────────────────────────────────────

    def save(self, entry: MemoryEntry) -> int:
        """保存记忆；与同类已有记忆近重复时合并（提升重要度并刷新时间）。"""
        with self._lock:
            existing = self._find_duplicate(entry)
            if existing is not None:
                self._merge_duplicate(existing, entry)
                return existing.id or -1

            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO memories (category, content, timestamp, importance, session_id) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (entry.category, entry.content, entry.timestamp,
                     entry.importance, entry.session_id),
                )
                new_id = cursor.lastrowid

        index = self._get_semantic()
        if index is not None:
            try:
                index.add(new_id, entry.content, entry.category)
            except Exception as e:
                logger.warning("写入记忆向量索引失败（忽略）：%s", e)
        return new_id

    def _find_duplicate(self, entry: MemoryEntry) -> MemoryEntry | None:
        """在同类记忆中查找近重复项（归一化完全相同或相似度超阈值）。"""
        target = normalize(entry.content)
        if not target:
            return None
        candidates = self.get_by_category(entry.category, limit=200)
        for cand in candidates:
            cand_norm = normalize(cand.content)
            if cand_norm == target:
                return cand
            if difflib.SequenceMatcher(None, cand_norm, target).ratio() >= MEMORY_DEDUP_THRESHOLD:
                return cand
        return None

    def _merge_duplicate(self, existing: MemoryEntry, incoming: MemoryEntry) -> None:
        importance = max(existing.importance, incoming.importance)
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET importance = ?, timestamp = ? WHERE id = ?",
                (importance, incoming.timestamp, existing.id),
            )
        logger.debug("记忆去重：合并到 id=%s", existing.id)

    # ── 读取 ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_entry(r) -> MemoryEntry:
        return MemoryEntry(
            id=r[0], category=r[1], content=r[2], timestamp=r[3],
            importance=r[4], session_id=r[5] if len(r) > 5 else None,
        )

    _SELECT = "SELECT id, category, content, timestamp, importance, session_id FROM memories"

    def search(self, keyword: str, category: str | None = None, limit: int = 10) -> list[MemoryEntry]:
        """按子串精确匹配检索（供明确关键词的场景使用）。"""
        query = f"{self._SELECT} WHERE content LIKE ?"
        params: list = [f"%{keyword}%"]
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def search_relevant(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """按自然语言查询检索相关记忆：语义索引优先，否则分词打分。"""
        if not query or not query.strip():
            return []

        index = self._get_semantic()
        if index is not None:
            try:
                ids = index.query(query, limit=limit)
                if ids:
                    return self._get_by_ids(ids)
            except Exception as e:
                logger.warning("记忆语义检索失败，降级为关键词：%s", e)

        return self._keyword_search(query, limit)

    def _get_by_ids(self, ids: list[int]) -> list[MemoryEntry]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"{self._SELECT} WHERE id IN ({placeholders})", ids
            ).fetchall()
        entries = {r[0]: self._row_to_entry(r) for r in rows}
        return [entries[i] for i in ids if i in entries]

    def _keyword_search(self, query: str, limit: int) -> list[MemoryEntry]:
        """分词粗筛 + Python 侧打分：命中 token 数 × 重要度 × 时间新近度。"""
        terms = keywords(query)
        if not terms:
            return []

        clause = " OR ".join(["content LIKE ?"] * len(terms))
        params = [f"%{t}%" for t in terms]
        with self._connect() as conn:
            rows = conn.execute(
                f"{self._SELECT} WHERE {clause} "
                "ORDER BY timestamp DESC LIMIT 200",
                params,
            ).fetchall()

        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, MemoryEntry]] = []
        for r in rows:
            entry = self._row_to_entry(r)
            content_tokens = set(tokenize(entry.content))
            if not content_tokens:
                continue
            overlap = len(query_tokens & content_tokens)
            if overlap == 0:
                continue
            # Jaccard 式覆盖率，避免长记忆天然占优
            coverage = overlap / len(query_tokens)
            score = coverage * (0.5 + entry.importance)
            scored.append((score, entry))

        # 分数降序，同分按时间新近优先
        scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        return [e for _, e in scored[:limit]]

    def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                f"{self._SELECT} ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_by_category(self, category: str, limit: int = 20) -> list[MemoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                f"{self._SELECT} WHERE category = ? ORDER BY importance DESC, timestamp DESC LIMIT ?",
                (category, limit),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]


CONSOLIDATE_PROMPT = """你要从一段科研助手对话中提取值得长期记住的信息。

只输出 JSON 数组，每项形如：
{"category": "preference|research_topic|finding", "content": "一句话事实", "importance": 0.0~1.0}

要求：
- preference：用户稳定的研究偏好/习惯；research_topic：用户关注的研究方向；finding：有价值的研究结论
- 只提取跨会话仍然有用的信息，忽略一次性的寒暄与操作细节
- 最多 5 条；没有值得记的就输出 []"""


@dataclass
class _SessionMemory:
    """单个会话的记忆作用域（短期摘要 + 待固化对话）。"""
    short_term: ShortTermMemory
    pending_turns: list[dict]


_DEFAULT_SCOPE = "default"
_MAX_PENDING_TURNS = 40


class MemoryManager:
    """记忆管理器 — 统一管理短期和长期记忆。

    短期部分按会话分桶：所有读写方法都接受可选的 `session_id`，
    省略时回落到实例自身的 `session_id`，再回落到默认桶。
    """

    def __init__(self, session_id: str | None = None):
        self.long_term = LongTermMemory()
        self.session_id = session_id
        self._scopes: dict[str, _SessionMemory] = {}
        self._scopes_lock = threading.Lock()

    # ── 会话作用域 ────────────────────────────────────────────

    def _scope_key(self, session_id: str | None = None) -> str:
        key = session_id if session_id is not None else self.session_id
        return str(key) if key is not None else _DEFAULT_SCOPE

    def _scope(self, session_id: str | None = None) -> _SessionMemory:
        key = self._scope_key(session_id)
        with self._scopes_lock:
            scope = self._scopes.get(key)
            if scope is None:
                scope = _SessionMemory(short_term=ShortTermMemory(), pending_turns=[])
                self._scopes[key] = scope
            return scope

    @property
    def short_term(self) -> ShortTermMemory:
        """当前默认作用域的短期记忆（供单会话调用方与测试直接访问）。"""
        return self._scope().short_term

    @property
    def _pending_turns(self) -> list[dict]:
        return self._scope().pending_turns

    # ── 写入 ──────────────────────────────────────────────────

    def record_interaction(self, user_input: str, agent_response: str,
                           session_id: str | None = None):
        """记录一轮交互，供后续 consolidate 提取长期记忆。"""
        turns = self._scope(session_id).pending_turns
        turns.append({"role": "user", "content": user_input})
        turns.append({"role": "assistant", "content": agent_response})
        # 只保留最近若干轮，避免无界增长（原地裁剪，保持引用不变）
        if len(turns) > _MAX_PENDING_TURNS:
            del turns[:-_MAX_PENDING_TURNS]

    def absorb_overflow(self, messages: list[dict], session_id: str | None = None) -> None:
        """接收 Agent 上下文窗口溢出的消息（由 Agent 调用）。"""
        self._scope(session_id).short_term.absorb(messages)

    def save_finding(self, content: str, importance: float = 0.7):
        """保存一个研究发现。"""
        self._save("finding", content, importance)

    def save_preference(self, content: str):
        """保存用户偏好。"""
        self._save("preference", content, 0.8)

    def _save(self, category: str, content: str, importance: float,
              session_id: str | None = None):
        self.long_term.save(MemoryEntry(
            id=None,
            category=category,
            content=content,
            timestamp=datetime.datetime.now().isoformat(),
            importance=importance,
            session_id=session_id if session_id is not None else self.session_id,
        ))

    def consolidate(self, session_id: str | None = None) -> int:
        """从近期对话中抽取长期记忆并落库，返回新增条数。

        解决「长期记忆只有在模型主动调工具时才会增长」的问题：会话结束/新建对话时调用。
        """
        turns = self._scope(session_id).pending_turns
        if not turns:
            return 0

        transcript = "\n".join(
            f"[{t['role']}]: {(t['content'] or '')[:400]}" for t in turns
        )
        try:
            resp = chat(
                messages=[
                    {"role": "system", "content": CONSOLIDATE_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("记忆固化失败（忽略）：%s", e)
            return 0

        items = self._parse_consolidation(content)
        saved = 0
        for item in items:
            category = item.get("category")
            text = (item.get("content") or "").strip()
            if category not in MEMORY_CATEGORIES or not text:
                continue
            try:
                importance = float(item.get("importance", 0.6))
            except (TypeError, ValueError):
                importance = 0.6
            self._save(category, text, max(0.0, min(1.0, importance)), session_id)
            saved += 1

        turns.clear()
        if saved:
            logger.info("记忆固化：新增/合并 %d 条长期记忆", saved)
        return saved

    @staticmethod
    def _parse_consolidation(content: str) -> list[dict]:
        import json

        text = (content or "").strip()
        start, end = text.find("["), text.rfind("]") + 1
        if start == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start:end])
        except json.JSONDecodeError:
            return []
        return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []

    # ── 读取 ──────────────────────────────────────────────────

    def get_context_for_prompt(self, current_query: str = "",
                               session_id: str | None = None) -> str:
        """为 Agent Prompt 生成记忆上下文（每轮重新生成，不写入对话历史）。"""
        parts = []

        short = self._scope(session_id).short_term.get_context()
        if short:
            parts.append(short)

        if current_query:
            relevant = self.long_term.search_relevant(current_query, limit=5)
            if relevant:
                memories = "\n".join(f"- [{m.category}] {m.content}" for m in relevant)
                parts.append(f"[相关历史记忆]\n{memories}")

        prefs = self.long_term.get_by_category("preference", limit=5)
        if prefs:
            pref_text = "\n".join(f"- {p.content}" for p in prefs)
            parts.append(f"[用户偏好]\n{pref_text}")

        return "\n\n".join(parts) if parts else ""

    def reset_session(self, session_id: str | None = None) -> None:
        """清理某会话的短期状态（长期记忆保留）。"""
        scope = self._scope(session_id)
        scope.short_term.reset()
        scope.pending_turns.clear()


_default_manager: "MemoryManager | None" = None


def get_memory_manager() -> "MemoryManager":
    """进程级 MemoryManager 单例 — UI 与工具层共用同一份记忆。"""
    global _default_manager
    if _default_manager is None:
        _default_manager = MemoryManager()
    return _default_manager
