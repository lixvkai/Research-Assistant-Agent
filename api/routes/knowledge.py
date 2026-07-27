"""知识库端点 —— 论文的上传 / 列举 / 删除。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from api.schemas import KBFileOut, KBIngestOut, KBStatsOut
from services import KnowledgeBaseService, get_kb_service

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# 论文体积上限，避免一次请求把内存吃满
MAX_UPLOAD_BYTES = 64 * 1024 * 1024


@router.get("/stats", response_model=KBStatsOut)
async def stats(service: KnowledgeBaseService = Depends(get_kb_service)) -> KBStatsOut:
    return KBStatsOut(**await run_in_threadpool(service.stats))


@router.get("/documents", response_model=list[KBFileOut])
async def list_documents(
    service: KnowledgeBaseService = Depends(get_kb_service),
) -> list[KBFileOut]:
    files = await run_in_threadpool(service.list_files)
    return [KBFileOut(**f) for f in files]


@router.post("/documents", response_model=KBIngestOut)
async def upload_document(
    file: UploadFile = File(...),
    service: KnowledgeBaseService = Depends(get_kb_service),
) -> KBIngestOut:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="上传内容为空"
        )
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限",
        )
    result = await run_in_threadpool(
        service.ingest_bytes, file.filename or "upload.txt", data
    )
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "导入失败"),
        )
    return KBIngestOut(**result)


@router.delete("/documents/{filename}", response_model=KBIngestOut)
async def delete_document(
    filename: str,
    service: KnowledgeBaseService = Depends(get_kb_service),
) -> KBIngestOut:
    result = await run_in_threadpool(service.delete_file, filename)
    if not result.get("ok"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "删除失败"),
        )
    return KBIngestOut(**result)
