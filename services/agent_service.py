"""Agent 运行时服务 — 会话管理 + 单轮对话的事件流。

与旧版 `app.py` 的区别在于**没有模块级的「当前会话」**：所有方法都以
`session_id` 为参数，会话之间的对话状态由 ReAct 图的 checkpointer 按
`thread_id` 隔离，短期记忆由 MemoryManager 按同一个键分桶。

三条并发相关的约定：
1. 同一会话不允许并发推理 —— 调用方须先 `acquire_session()`，
   拿不到锁即说明上一条消息还在处理（对应 HTTP 409）。
2. 不同会话可以并发，因为图状态与短期记忆都是按会话隔离的。
3. checkpointer 目前是进程内的 MemorySaver，所以首次访问一个历史会话时
   需要从 SQLite 回灌消息（`_ensure_hydrated`），进程重启后同样如此。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator

from memory.chat_history import ChatHistoryStore
from memory.memory_store import get_memory_manager

logger = logging.getLogger(__name__)

_TITLE_MAX_LEN = 30
_DEFAULT_TITLE = "新对话"


class SessionBusyError(RuntimeError):
    """该会话已有推理在进行中。"""


class SessionNotFoundError(LookupError):
    """会话不存在。"""


def derive_title(message: str) -> str:
    """用首条用户消息生成会话标题。"""
    text = (message or "").strip()
    if not text:
        return _DEFAULT_TITLE
    return text[:_TITLE_MAX_LEN] + ("…" if len(text) > _TITLE_MAX_LEN else "")


class SessionGuard:
    """`acquire_session()` 返回的持有凭证，释放后该会话才能接受下一条消息。"""

    def __init__(self, lock: threading.Lock):
        self._lock = lock
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._lock.release()

    def __enter__(self) -> "SessionGuard":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


class AgentService:
    """会话 + 对话的业务入口。"""

    def __init__(self, agent=None, history_store=None, memory=None,
                 tools: list[dict] | None = None):
        self._history = history_store if history_store is not None else ChatHistoryStore()
        self._memory = memory if memory is not None else get_memory_manager()
        self._tools = tools
        self._agent = agent if agent is not None else self._build_agent()
        self._locks: dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self._hydrated: set[int] = set()
        self._hydrate_guard = threading.Lock()

    def _build_agent(self):
        from core.mcp import get_default_mcp_server
        from core.react_agent import ReActAgent

        agent = ReActAgent(memory_manager=self._memory)
        server = get_default_mcp_server()
        server.bind_to_agent(agent)
        if self._tools is None:
            self._tools = server.list_tools()
        return agent

    @property
    def agent(self):
        return self._agent

    # ── 元信息 ────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        if self._tools is None:
            from core.mcp import get_default_mcp_server

            self._tools = get_default_mcp_server().list_tools()
        return list(self._tools)

    def list_skills(self) -> list[dict]:
        """已定义的技能（失败时返回空列表，技能系统不可用不应阻断主流程）。"""
        try:
            from skills.skill_manager import SkillManager

            skills = SkillManager().list_skills()
        except Exception as e:
            logger.warning("读取技能列表失败（忽略）：%s", e)
            return []
        return [
            {
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "usage_count": s.usage_count,
                "success_rate": s.success_rate,
            }
            for s in skills
        ]

    # ── 会话 ──────────────────────────────────────────────────

    def create_session(self, title: str | None = None) -> dict:
        clean = (title or "").strip() or _DEFAULT_TITLE
        session_id = self._history.create_session(clean)
        # 新会话没有历史，直接标记为已回灌，省掉一次无用的 SQLite 查询
        self._hydrated.add(session_id)
        return self.get_session(session_id)

    def get_session(self, session_id: int) -> dict:
        info = self._history.get_session(session_id)
        if info is None:
            raise SessionNotFoundError(f"会话 {session_id} 不存在")
        return info

    def list_sessions(self, limit: int = 30) -> list[dict]:
        return self._history.list_sessions(limit=limit)

    def get_messages(self, session_id: int) -> list[dict]:
        self.get_session(session_id)
        return self._history.get_messages(session_id)

    def ensure_session(self, session_id: int | None, first_message: str) -> int:
        """返回可用的会话 id：为空时按首条消息新建，否则校验其存在。"""
        if session_id is None:
            return self.create_session(derive_title(first_message))["id"]
        self.get_session(session_id)
        return session_id

    def delete_session(self, session_id: int) -> None:
        self.get_session(session_id)
        self._history.delete_session(session_id)
        self._forget(session_id)

    def reset_session(self, session_id: int) -> None:
        """结束一段会话：先把内容固化进长期记忆，再清空对话状态与短期记忆。"""
        scope = str(session_id)
        try:
            self._memory.consolidate(scope)
        except Exception as e:
            logger.warning("会话 %s 记忆固化失败（忽略）：%s", session_id, e)
        self._forget(session_id)

    def _forget(self, session_id: int) -> None:
        """丢弃某会话的进程内状态（图消息 + 短期记忆）。"""
        try:
            self._agent.reset(session_id=str(session_id))
        except Exception as e:
            logger.warning("清理会话 %s 的图状态失败（忽略）：%s", session_id, e)
        self._hydrated.discard(session_id)

    # ── 并发控制 ──────────────────────────────────────────────

    def acquire_session(self, session_id: int) -> SessionGuard:
        """独占某会话，防止同一会话被并发推理（LangGraph 状态会被写坏）。"""
        with self._locks_guard:
            lock = self._locks.setdefault(session_id, threading.Lock())
        if not lock.acquire(blocking=False):
            raise SessionBusyError(f"会话 {session_id} 正在处理上一条消息")
        return SessionGuard(lock)

    def is_busy(self, session_id: int) -> bool:
        with self._locks_guard:
            lock = self._locks.get(session_id)
        return bool(lock and lock.locked())

    # ── 历史回灌 ──────────────────────────────────────────────

    def _ensure_hydrated(self, session_id: int) -> None:
        """首次访问某历史会话时，把落库的消息回灌进图状态。

        checkpointer 是进程内的，所以「切回旧会话」和「进程重启后继续旧会话」
        都需要这一步，否则 Agent 会以为这是一段全新的对话。
        """
        if session_id in self._hydrated:
            return
        with self._hydrate_guard:
            if session_id in self._hydrated:
                return
            messages = self._history.get_messages(session_id)
            if messages:
                self._agent.load_history(messages, session_id=str(session_id))
            self._hydrated.add(session_id)

    # ── 对话 ──────────────────────────────────────────────────

    def stream_chat(self, session_id: int, message: str) -> Iterator[dict]:
        """执行一轮对话，逐个产出 ReAct 事件。

        调用方须已持有该会话的 `SessionGuard`。落库的助手消息是**纯文本答案**，
        推理轨迹只作为事件流推给客户端、不写进历史 —— 否则渲染用的 HTML
        会在下一轮被当成对话内容重新喂给 LLM。
        """
        text = (message or "").strip()
        if not text:
            return

        # 在唯一的写入入口校验会话，避免非法 id 产生挂不到任何会话的孤儿消息
        self.get_session(session_id)
        self._ensure_hydrated(session_id)
        self._history.save_message(session_id, "user", text)

        answer = ""
        error = ""
        for event in self._agent.run_iter(text, session_id=str(session_id)):
            etype = event.get("type")
            if etype == "answer":
                answer = event.get("content") or ""
            elif etype == "error":
                error = event.get("content") or ""
            yield event

        final = answer or (f"处理出错：{error}" if error else "")
        if final:
            self._history.save_message(session_id, "assistant", final)
