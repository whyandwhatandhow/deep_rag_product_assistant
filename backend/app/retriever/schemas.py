# backend/app/retriever/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class RetrievedChunk(BaseModel):
    """单条检索到的 chunk（检索和重排后的统一数据结构）"""
    chunk_id: str = Field(..., description="Chunk 在 Chroma 中的 ID")
    document_id: str = Field(..., description="关联的文档 ID，对应 ingested_documents 表")

    product_name: str = Field(..., description="产品名称，例如：医学影像产品")
    doc_type: str = Field(default="paper", description="文档类型：paper、手册、FAQ 等")

    text: str = Field(..., description="chunk 的原始文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="完整 metadata")

    # 检索与重排相关字段
    score: float = Field(0.0, description="向量相似度或重排分数（越高越相关）")
    rank: Optional[int] = Field(None, description="重排后的排名，从 1 开始")

    # 可选的来源信息（用于生成 citations）
    filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None


class RetrievalResult(BaseModel):
    """一次检索的完整结果（可观察、可调试）"""
    query: str = Field(..., description="原始用户问题")
    rewritten_queries: List[str] = Field(default_factory=list, description="Query Rewrite 后的查询列表（未来扩展）")

    chunks: List[RetrievedChunk] = Field(..., description="召回的 chunk 列表")
    total_retrieved: int = Field(..., description="实际返回的 chunk 数量")

    used_metadata_filter: bool = Field(False, description="是否使用了 product 等 metadata 过滤")
    retrieval_time_ms: Optional[int] = Field(None, description="检索耗时（毫秒）")

    # 调试信息
    debug_info: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ContextAssemblyResult(BaseModel):
    """上下文聚合后的结果（传给 Generator 使用）"""
    context_str: str = Field(..., description="最终拼接好的上下文字符串（用于 Prompt）")
    used_chunks: List[RetrievedChunk] = Field(..., description="实际参与生成的 chunks（用于生成 citations）")
    compressed: bool = Field(False, description="是否进行了压缩")