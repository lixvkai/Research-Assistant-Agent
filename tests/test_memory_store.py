"""记忆系统测试：去重 / 分词召回 / 摘要吸收 / 记忆固化。"""

import datetime

import memory.memory_store as ms
from memory.memory_store import LongTermMemory, MemoryEntry, MemoryManager, ShortTermMemory


def _entry(content, category="finding", importance=0.7):
    return MemoryEntry(
        id=None,
        category=category,
        content=content,
        timestamp=datetime.datetime.now().isoformat(),
        importance=importance,
    )


def _store(tmp_path):
    return LongTermMemory(db_path=str(tmp_path / "mem.db"), semantic=False)


def test_save_and_get_recent(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("检索增强生成可以缓解幻觉"))
    recent = lt.get_recent(limit=5)
    assert len(recent) == 1
    assert "检索增强生成" in recent[0].content


def test_dedup_merges_identical_content(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("扩散模型采样可以加速", importance=0.5))
    lt.save(_entry("扩散模型采样可以加速", importance=0.9))
    rows = lt.get_by_category("finding", limit=10)
    assert len(rows) == 1, "同一条记忆重复保存不应产生多行"
    assert rows[0].importance == 0.9, "重要度应取较高值"


def test_dedup_merges_near_duplicate(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("用户偏好使用 PyTorch 进行实验"))
    lt.save(_entry("用户偏好使用 PyTorch 进行实验。"))
    assert len(lt.get_by_category("finding", limit=10)) == 1


def test_distinct_content_not_merged(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("扩散模型采样加速"))
    lt.save(_entry("大语言模型的推理能力评估"))
    assert len(lt.get_by_category("finding", limit=10)) == 2


def test_relevant_search_beats_whole_sentence_like(tmp_path):
    """回归：整句 LIKE 几乎不可能命中，分词召回必须能命中。"""
    lt = _store(tmp_path)
    lt.save(_entry("用户关注扩散模型的采样加速方法"))

    query = "帮我分析一下扩散模型这两年的研究趋势"
    assert lt.search(query) == [], "整句子串匹配本就不该命中（这是修复前的行为）"
    hits = lt.search_relevant(query, limit=5)
    assert hits and "扩散模型" in hits[0].content


def test_relevant_search_matches_english_term(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("用户在做 retrieval augmented generation 相关研究"))
    hits = lt.search_relevant("我想了解 retrieval augmented generation 的进展")
    assert hits


def test_relevant_search_empty_query(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("x"))
    assert lt.search_relevant("") == []


def test_relevant_search_no_match(tmp_path):
    lt = _store(tmp_path)
    lt.save(_entry("扩散模型采样加速"))
    assert lt.search_relevant("quantum chromodynamics lattice") == []


def test_short_term_absorb_builds_summary(monkeypatch, mkresp):
    monkeypatch.setattr(ms, "chat", lambda messages, **kw: mkresp("这是摘要"))
    st = ShortTermMemory()
    st.absorb([
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
    ])
    assert st.summary == "这是摘要"
    assert "[对话历史摘要]" in st.get_context()


def test_short_term_absorb_ignores_empty(monkeypatch):
    called = {"n": 0}

    def boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("不应调用 LLM")

    monkeypatch.setattr(ms, "chat", boom)
    st = ShortTermMemory()
    st.absorb([{"role": "tool", "content": ""}])
    assert st.summary == ""
    assert called["n"] == 0


def test_short_term_absorb_failure_is_bounded(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("LLM 挂了")

    monkeypatch.setattr(ms, "chat", boom)
    st = ShortTermMemory()
    for _ in range(50):
        st.absorb([{"role": "user", "content": "x" * 500}])
    assert len(st.summary) <= ms._MAX_FALLBACK_SUMMARY


def test_consolidate_extracts_long_term_memories(monkeypatch, tmp_path, mkresp):
    payload = (
        '[{"category":"preference","content":"用户偏好中文回答","importance":0.9},'
        '{"category":"finding","content":"RAG 可缓解幻觉","importance":0.7}]'
    )
    monkeypatch.setattr(ms, "chat", lambda messages, **kw: mkresp(payload))

    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    mgr.record_interaction("我想研究 RAG", "好的")

    assert mgr.consolidate() == 2
    assert mgr.long_term.get_by_category("preference", limit=5)
    assert mgr.long_term.get_by_category("finding", limit=5)


def test_consolidate_ignores_bad_json(monkeypatch, tmp_path, mkresp):
    monkeypatch.setattr(ms, "chat", lambda messages, **kw: mkresp("我不知道"))
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    mgr.record_interaction("q", "a")
    assert mgr.consolidate() == 0


def test_consolidate_rejects_unknown_category(monkeypatch, tmp_path, mkresp):
    monkeypatch.setattr(
        ms, "chat",
        lambda messages, **kw: mkresp('[{"category":"nonsense","content":"x"}]'),
    )
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    mgr.record_interaction("q", "a")
    assert mgr.consolidate() == 0


def test_consolidate_without_turns_is_noop(tmp_path):
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    assert mgr.consolidate() == 0


def test_reset_session_clears_short_term(tmp_path):
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    mgr.short_term.summary = "旧会话摘要"
    mgr.record_interaction("q", "a")
    mgr.reset_session()
    assert mgr.short_term.summary == ""
    assert mgr._pending_turns == []


def test_short_term_memory_is_scoped_per_session(monkeypatch, tmp_path, mkresp):
    """回归：A 会话的对话摘要不能漏进 B 会话的 prompt。

    短期摘要曾是进程级共享的，单会话 UI 下看不出问题，
    但 HTTP 接口并发服务多会话时会直接串味。
    """
    monkeypatch.setattr(ms, "chat", lambda messages, **kw: mkresp("A 会话在聊扩散模型"))
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)

    mgr.absorb_overflow([{"role": "user", "content": "扩散模型怎么加速"}], session_id="A")

    assert "扩散模型" in mgr.get_context_for_prompt("问题", session_id="A")
    assert mgr.get_context_for_prompt("问题", session_id="B") == ""


def test_pending_turns_are_scoped_per_session(monkeypatch, tmp_path, mkresp):
    monkeypatch.setattr(ms, "chat", lambda messages, **kw: mkresp("[]"))
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)

    mgr.record_interaction("A 的问题", "A 的回答", session_id="A")
    # B 会话没有待固化内容，固化应是空操作
    assert mgr.consolidate("B") == 0
    assert len(mgr._scope("A").pending_turns) == 2


def test_reset_session_only_clears_target_scope(tmp_path):
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    mgr._scope("A").short_term.summary = "A 的摘要"
    mgr._scope("B").short_term.summary = "B 的摘要"

    mgr.reset_session("A")
    assert mgr._scope("A").short_term.summary == ""
    assert mgr._scope("B").short_term.summary == "B 的摘要"


def test_pending_turns_are_bounded(tmp_path):
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    for i in range(50):
        mgr.record_interaction(f"q{i}", f"a{i}")
    assert len(mgr._pending_turns) == ms._MAX_PENDING_TURNS
    # 保留的应该是最近的几轮
    assert mgr._pending_turns[-1]["content"] == "a49"


def test_context_for_prompt_composes_sections(tmp_path):
    mgr = MemoryManager()
    mgr.long_term = _store(tmp_path)
    mgr.short_term.summary = "之前聊了扩散模型"
    mgr.save_preference("用户偏好中文")

    ctx = mgr.get_context_for_prompt("扩散模型的趋势")
    assert "[对话历史摘要]" in ctx
    assert "[用户偏好]" in ctx
