"""文档加载与分块 — 将 PDF/文本转为可检索的块。"""

import os
import re
import logging
from dataclasses import dataclass

import numpy as np

from rag.embeddings import get_sentence_model

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """一个文本块。"""
    content: str
    metadata: dict


def load_pdf(file_path: str) -> str:
    """从 PDF 文件提取文本。"""
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_text(file_path: str) -> str:
    """从纯文本文件读取内容。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_document(file_path: str) -> str:
    """自动检测文件类型并加载内容。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".txt", ".md", ".tex"):
        return load_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}")


# ── 分句 ──────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """按中英文句子边界切分。"""
    parts = re.split(r'(?<=[。！？；.!?\n])\s*', text)
    return [s.strip() for s in parts if s.strip()]


# ── 固定窗口分块（fallback） ──────────────────────────────────

def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """固定窗口分块 — 作为语义分块的 fallback。"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            for sep in ["\n\n", "。", ".\n", ". ", "\n"]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunks.append(text[start:end].strip())
        start = end - chunk_overlap

    return [c for c in chunks if c]


# ── 语义分块 ──────────────────────────────────────────────────

def split_text_semantic(
    text: str,
    max_chunk_size: int = 800,
    min_chunk_size: int = 100,
    breakpoint_percentile: int = 25,
) -> list[str]:
    """语义分块 — 在相邻句子语义相似度骤降处切分。

    流程:
      1. 按句子边界切分
      2. 用 SentenceTransformer 编码每句话
      3. 计算相邻句子余弦相似度
      4. 低于 breakpoint_percentile 分位数处断开
      5. 同时遵守 max_chunk_size 上限
    失败时自动回退到固定窗口分块。
    """
    sentences = _split_into_sentences(text)

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    if len(text) <= max_chunk_size:
        return [text.strip()]

    try:
        model = get_sentence_model()
        embeddings = model.encode(sentences, show_progress_bar=False)

        similarities = []
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            similarities.append(sim)

        threshold = float(np.percentile(similarities, breakpoint_percentile))

        chunks = []
        current = [sentences[0]]

        for i in range(1, len(sentences)):
            merged = "\n".join(current + [sentences[i]])
            current_len = len("\n".join(current))

            should_break = (
                (similarities[i - 1] < threshold and current_len >= min_chunk_size)
                or len(merged) > max_chunk_size
            )

            if should_break:
                chunks.append("\n".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])

        if current:
            chunks.append("\n".join(current))

        result = [c.strip() for c in chunks if c.strip()]
        logger.info(f"语义分块: {len(sentences)} 句 → {len(result)} 块")
        return result

    except Exception as e:
        logger.warning(f"语义分块失败，回退到固定窗口: {e}")
        return split_text(text)


# ── 文档处理入口 ──────────────────────────────────────────────

def process_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[TextChunk]:
    """完整处理: 加载文件 → 语义分块（失败则回退固定窗口）→ TextChunk 列表。"""
    text = load_document(file_path)
    filename = os.path.basename(file_path)

    raw_chunks = split_text_semantic(text, max_chunk_size=max(chunk_size, 800))

    return [
        TextChunk(
            content=chunk,
            metadata={"source": filename, "chunk_index": i, "total_chunks": len(raw_chunks)},
        )
        for i, chunk in enumerate(raw_chunks)
    ]
