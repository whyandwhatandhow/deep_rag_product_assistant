# backend/app/ingest/ingest_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import shutil
from app.ingest.processor import document_processor

router = APIRouter(prefix="/ingest", tags=["文档接入"])

UPLOAD_DIR = Path("../data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/document")
async def ingest_document(
    file: UploadFile = File(...),
    product_name: str = Form("默认产品")
):
    """上传并处理单个产品文档"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供文件名")

    # 保存原始文件
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 处理文档 → 生成 chunks
        chunks = document_processor.process_document(file_path, product_name)

        # TODO: 后续这里会调用索引模块存入 Chroma
        chunk_count = len(chunks)

        return {
            "status": "success",
            "message": f"文档 {file.filename} 处理完成",
            "doc_id": chunks[0].doc_id if chunks else None,
            "chunk_count": chunk_count,
            "product_name": product_name,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")