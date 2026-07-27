"""共享嵌入模型单例 — 分块 / 入库 / 检索复用同一份 SentenceTransformer。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_sentence_model = None

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def get_sentence_model() -> "SentenceTransformer":
    """懒加载并缓存 all-MiniLM-L6-v2。"""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("加载嵌入模型：%s", EMBEDDING_MODEL_NAME)
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _sentence_model


class SharedEmbeddingFunction:
    """ChromaDB EmbeddingFunction，委托给共享 SentenceTransformer 单例。"""

    def __call__(self, input):
        model = get_sentence_model()
        embeddings = model.encode(list(input), show_progress_bar=False)
        return embeddings.tolist()

    def name(self) -> str:
        return f"shared-{EMBEDDING_MODEL_NAME}"
