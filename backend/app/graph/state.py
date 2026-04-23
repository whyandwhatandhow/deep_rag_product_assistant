# backend/app/graph/state.py
from typing import TypedDict, List, Annotated
from app.retriever.schemas import RetrievedChunk, RetrievalResult


class DeepRAGState(TypedDict):
    """深 RAG 主链路状态（所有节点共享）"""
    question: str  # 用户原始问题
    rewritten_query: str  # Query Rewrite 后的查询（当前阶段可简单用原 query）

    retrieval_result: RetrievalResult | None  # 检索结果
    reranked_chunks: List[RetrievedChunk]  # 重排后的 chunks

    context_str: str  # 最终拼接的上下文（传给 Generator）
    used_chunks: List[RetrievedChunk]  # 用于生成 citations 的 chunks

    answer: str  # 生成的最终答案
    citations: List[dict]  # Citation 列表
    confidence: float  # 置信度（0.0~1.0）
    has_evidence: bool  # 是否找到证据