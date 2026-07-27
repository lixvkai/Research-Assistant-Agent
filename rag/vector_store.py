"""向量存储 — 基于 ChromaDB 的文档索引与检索。"""

import chromadb

from config.settings import CHROMA_PERSIST_DIR
from rag.embeddings import SharedEmbeddingFunction


class VectorStore:
    """封装 ChromaDB，提供文档索引和语义检索。"""

    def __init__(self, collection_name: str = "research_papers"):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._ef = SharedEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, texts: list[str], metadatas: list[dict], ids: list[str]):
        """批量添加文档到向量库。"""
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """语义搜索，返回最相关的文档块。"""
        results = self.collection.query(query_texts=[query], n_results=top_k)

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return docs

    def delete_by_source(self, source_filename: str) -> int:
        """删除某个来源文件的所有文档块，返回删除数量。"""
        results = self.collection.get(where={"source": source_filename})
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> dict:
        """获取向量库统计信息。"""
        return {
            "collection": self.collection.name,
            "document_count": self.collection.count(),
        }
