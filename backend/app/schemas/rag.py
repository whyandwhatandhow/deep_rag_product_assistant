# backend/app/schemas/rag.py
from pydantic import BaseModel
from typing import Optional, List


class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    product_name: str
    doc_type: str
    page_num: Optional[int] = None
    update_time: str
    source_file: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    metadata: DocumentMetadata