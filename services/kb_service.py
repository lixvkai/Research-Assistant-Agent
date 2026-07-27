"""知识库服务 — 论文文件的上传 / 列举 / 删除 / 统计。

返回的是结构化数据而不是 HTML：Gradio 侧自己渲染，FastAPI 侧直接序列化成 JSON。
所有落盘路径都经过 `utils.path_safety` 校验，杜绝越目录写入。
"""

from __future__ import annotations

import logging
import os
import shutil

from config.settings import PAPERS_DIR
from utils.path_safety import resolve_under, safe_filename

logger = logging.getLogger(__name__)

SUPPORTED_DOC_EXTS = (".pdf", ".txt", ".md", ".tex")


class KnowledgeBaseService:

    def __init__(self, papers_dir: str = PAPERS_DIR):
        self.papers_dir = papers_dir

    # ── 查询 ──────────────────────────────────────────────────

    def list_files(self) -> list[dict]:
        if not os.path.isdir(self.papers_dir):
            return []
        files = []
        for name in sorted(os.listdir(self.papers_dir)):
            if not name.lower().endswith(SUPPORTED_DOC_EXTS):
                continue
            path = os.path.join(self.papers_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            files.append({"name": name, "size": size})
        return files

    def stats(self) -> dict:
        """文件数 + 向量块数。RAG 引擎不可用时块数记为 0 并记日志，不阻断调用方。"""
        files = self.list_files()
        collection, chunks = "", 0
        try:
            from tools.rag_tool import _get_engine

            engine_stats = _get_engine().get_stats()
            collection = engine_stats.get("collection", "")
            chunks = int(engine_stats.get("document_count", 0))
        except Exception as e:
            logger.warning("读取知识库统计失败：%s", e)
        return {"files": len(files), "chunks": chunks, "collection": collection}

    # ── 写入 ──────────────────────────────────────────────────

    def ingest_path(self, src_path: str, filename: str | None = None) -> dict:
        """把本地某个文件复制进论文目录并入库（Gradio 上传给的是临时文件路径）。"""
        try:
            name = safe_filename(filename or os.path.basename(src_path))
        except ValueError as e:
            return {"ok": False, "filename": str(filename or src_path), "error": str(e)}

        dest = os.path.join(self.papers_dir, name)
        os.makedirs(self.papers_dir, exist_ok=True)
        try:
            shutil.copy2(src_path, dest)
        except OSError as e:
            return {"ok": False, "filename": name, "error": f"写入失败：{e}"}
        return self._ingest(name)

    def ingest_bytes(self, filename: str, data: bytes) -> dict:
        """把上传的字节流写入论文目录并入库（FastAPI 的 UploadFile 走这条）。"""
        try:
            name = safe_filename(filename)
        except ValueError as e:
            return {"ok": False, "filename": filename, "error": str(e)}
        if not name.lower().endswith(SUPPORTED_DOC_EXTS):
            return {
                "ok": False,
                "filename": name,
                "error": f"不支持的文件类型，仅支持 {', '.join(SUPPORTED_DOC_EXTS)}",
            }

        os.makedirs(self.papers_dir, exist_ok=True)
        dest = os.path.join(self.papers_dir, name)
        try:
            with open(dest, "wb") as f:
                f.write(data)
        except OSError as e:
            return {"ok": False, "filename": name, "error": f"写入失败：{e}"}
        return self._ingest(name)

    def _ingest(self, filename: str) -> dict:
        try:
            path = resolve_under(self.papers_dir, filename)
        except ValueError as e:
            return {"ok": False, "filename": filename, "error": str(e)}
        try:
            from tools.rag_tool import _get_engine

            chunks = _get_engine().ingest_file(path)
        except Exception as e:
            logger.exception("导入知识库失败：%s", filename)
            return {"ok": False, "filename": filename, "error": f"导入失败：{e}"}
        return {"ok": True, "filename": filename, "chunks": chunks}

    def delete_file(self, filename: str) -> dict:
        try:
            name = safe_filename(filename)
            path = resolve_under(self.papers_dir, name)
        except ValueError as e:
            return {"ok": False, "filename": filename, "error": str(e)}

        try:
            from tools.rag_tool import _get_engine

            chunks = _get_engine().delete_file(name)
        except Exception as e:
            logger.exception("从向量库删除失败：%s", name)
            return {"ok": False, "filename": name, "error": f"删除失败：{e}"}

        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                return {"ok": False, "filename": name, "error": f"删除文件失败：{e}"}
        return {"ok": True, "filename": name, "chunks": chunks}
