"""中英文混合文本的轻量分词 — 供 BM25 检索与记忆关键词召回共用。

不引入 jieba 等额外依赖：ASCII 走单词切分，CJK 走二元组（bigram）切分。
bigram 对中文检索的召回效果远好于把整句当一个 token。
"""

from __future__ import annotations

import re

_ASCII_WORD = re.compile(r"[a-zA-Z0-9_\-+#.]{2,}")
_CJK_CHAR = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "are", "was", "were", "from",
    "什么", "怎么", "哪些", "以及", "关于", "一下", "我们", "可以", "帮我",
}


def tokenize(text: str) -> list[str]:
    """切分为检索用 token：小写 ASCII 词 + 中文 bigram（单字兜底）。"""
    if not text:
        return []

    tokens: list[str] = []
    for m in _ASCII_WORD.finditer(text):
        w = m.group(0).lower()
        if w not in _STOPWORDS:
            tokens.append(w)

    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        for i in range(len(run) - 1):
            bigram = run[i : i + 2]
            if bigram not in _STOPWORDS:
                tokens.append(bigram)

    return tokens


def keywords(text: str, limit: int = 12) -> list[str]:
    """抽取用于 SQL LIKE 粗筛的关键词：较长的 ASCII 词 + 中文二/三字片段。"""
    if not text:
        return []

    out: list[str] = []
    seen: set[str] = set()

    for m in _ASCII_WORD.finditer(text):
        w = m.group(0).lower()
        if len(w) >= 3 and w not in _STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)

    for run in _CJK_RUN.findall(text):
        # 中文取 2~3 字片段，兼顾召回与噪声
        size = 2 if len(run) < 3 else 3
        for i in range(0, max(1, len(run) - size + 1)):
            frag = run[i : i + size]
            if len(frag) >= 2 and frag not in _STOPWORDS and frag not in seen:
                seen.add(frag)
                out.append(frag)

    return out[:limit]


def has_cjk(text: str) -> bool:
    return bool(_CJK_CHAR.search(text or ""))


def normalize(text: str) -> str:
    """用于近重复判定的归一化：去空白、转小写。"""
    return re.sub(r"\s+", "", (text or "").strip().lower())
