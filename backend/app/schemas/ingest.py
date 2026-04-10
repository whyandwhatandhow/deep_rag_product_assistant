# backend/app/schemas/ingest.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class DocumentMetadata(BaseModel):
    """文档级元数据"""
    source: str = Field(..., description="原始文件名或路径")
    product_name: Optional[str] = Field(None, description="产品名称")
    doc_type: str = Field(..., description="文档类型：manual/faq/policy/spec 等")
    version: Optional[str] = Field(None, description="文档版本")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    language: str = Field("zh", description="文档语言")
    title: Optional[str] = Field(None, description="文档标题")
    total_pages: Optional[int] = Field(None, description="总页数")


class ChunkMetadata(BaseModel):
    """每个 chunk 的元数据（用于向量库和 PG）"""
    document_id: str = Field(..., description="文档唯一ID")
    chunk_index: int = Field(..., description="当前 chunk 在文档中的序号")
    source: str = Field(..., description="来源文件名")
    page_number: Optional[int] = Field(None, description="页码")
    section_title: Optional[str] = Field(None, description="所在章节标题")
    product_name: Optional[str] = Field(None, description="产品名称")
    doc_type: str = Field(..., description="文档类型")
    update_time: Optional[datetime] = Field(None)
    # 可扩展字段
    extra: Dict[str, Any] = Field(default_factory=dict)


class IngestedDocument(BaseModel):
    """完整文档记录（存 PG 用）"""
    document_id: str
    filename: str
    metadata: DocumentMetadata
    chunk_count: int
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """向量库存储的最小单元"""
    page_content: str
    metadata: ChunkMetadata