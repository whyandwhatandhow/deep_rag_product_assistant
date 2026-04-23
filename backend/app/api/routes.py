from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import time
from app.schemas.response import RAGResponse, Citation, RetrievalMetadata, QueryInfo
from app.graph.workflow import deep_rag_chain

router = APIRouter(prefix="/api/v1", tags=["RAG问答"])


class QueryRequest(BaseModel):
    question: str
    product_filter: Optional[str] = None


@router.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    """
    RAG 查询接口

    - 接收用户问题
    - 通过深 RAG 链路处理
    - 返回结构化答案和引用
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    start_time = time.time()

    initial_state = {
        "question": request.question,
        "rewritten_query": request.question,
        "retrieval_result": None,
        "reranked_chunks": [],
        "context_str": "",
        "used_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "has_evidence": False,
    }

    result = deep_rag_chain.invoke(initial_state)

    total_time_ms = (time.time() - start_time) * 1000

    citations = [
        Citation(**cit) for cit in result.get("citations", [])
    ]

    retrieval_time_ms = total_time_ms * 0.6
    rerank_time_ms = total_time_ms * 0.2
    generation_time_ms = total_time_ms * 0.2

    # 修复 retrieval_result 访问方式
    total_chunks_retrieved = 0
    if result.get("retrieval_result"):
        retrieval_result = result.get("retrieval_result")
        if hasattr(retrieval_result, "chunks"):
            total_chunks_retrieved = len(retrieval_result.chunks)
        elif isinstance(retrieval_result, dict):
            total_chunks_retrieved = len(retrieval_result.get("chunks", []))

    return RAGResponse(
        answer=result.get("answer", ""),
        citations=citations,
        confidence=result.get("confidence", 0.0),
        has_evidence=result.get("has_evidence", False),
        query_info=QueryInfo(
            original_question=request.question,
            rewritten_query=result.get("rewritten_query", request.question)
        ),
        retrieval_metadata=RetrievalMetadata(
            retrieval_time_ms=retrieval_time_ms,
            total_chunks_retrieved=total_chunks_retrieved,
            chunks_after_rerank=len(result.get("reranked_chunks", [])),
            has_evidence=result.get("has_evidence", False)
        )
    )


@router.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "deep-rag-product-assistant",
        "version": "0.1.0"
    }
