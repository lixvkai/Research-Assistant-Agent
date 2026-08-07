"""向量存储 — 基于 ChromaDB 的文档索引与检索。"""

from config.settings import CHROMA_PERSIST_DIR
from rag.embeddings import SharedEmbeddingFunction


class VectorStore:
    """封装 ChromaDB，提供文档索引和语义检索。"""

    def __init__(self, collection_name: str = "research_papers"):
        # ChromaDB 只在创建真实向量库时加载。这样纯检索结果转换测试不必
        # 在 GitHub Actions 中安装重量级 RAG 运行依赖。
        import chromadb
        from chromadb.config import Settings

        # 本地向量库不需要向 Chroma 发送匿名产品遥测；同时避免其与新版
        # PostHog SDK 的接口差异干扰应用启动。
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
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
            distance = float(results["distances"][0][i])
            docs.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": 1.0 - distance,
                "recall_rank": i + 1,
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

    def list_document_ids(self) -> list[str]:
        """返回当前 collection 的全部文档块 ID，供本地评测校验使用。"""
        # Chroma 始终返回 ids；请求 metadata 可兼容不接受空 include 的版本，
        # 同时避免读取文档正文与向量。
        return list(self.collection.get(include=["metadatas"])["ids"])

    def list_documents(self) -> list[dict]:
        """返回评测数据生成所需的文档块正文与元数据。"""
        results = self.collection.get(include=["documents", "metadatas"])
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        return [
            {
                "id": document_id,
                "content": documents[index] or "",
                "metadata": metadatas[index] or {},
            }
            for index, document_id in enumerate(results.get("ids") or [])
        ]
