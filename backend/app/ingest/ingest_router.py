# backend/app/ingest/ingest_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import shutil
import logging
from app.ingest.processor import document_processor
from app.ingest.indexer import DocumentIndexer

router = APIRouter(prefix="/ingest", tags=["文档接入"])
logger = logging.getLogger(__name__)

# 使用相对于 backend 目录的路径
UPLOAD_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/document")
async def ingest_document(
    file: UploadFile = File(...),
    product_name: str = Form("默认产品"),
    doc_type: str = Form("manual")
):
    """上传并处理单个产品文档，自动写入 Chroma 和 PostgreSQL"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    # 保存原始文件
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 处理文档 → 生成 chunks
        chunks = document_processor.process_document(file_path, product_name)

        if not chunks:
            raise HTTPException(status_code=400, detail="文档处理后未生成任何 chunk")

        # 调用索引器写入 Chroma 和 PostgreSQL
        indexer = DocumentIndexer()
        document_id = indexer.index_documents(chunks, product_name=product_name, doc_type=doc_type)

        chunk_count = len(chunks)
        logger.info(f"✅ 文档 {file.filename} 已成功写入 Chroma，document_id: {document_id}, chunk_count: {chunk_count}")

        return {
            "status": "success",
            "message": f"文档 {file.filename} 处理并索引完成",
            "doc_id": document_id,
            "chunk_count": chunk_count,
            "product_name": product_name,
            "doc_type": doc_type,
        }

    except Exception as e:
        logger.error(f"❌ 文档处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")
