# backend/app/graph/nodes.py
from app.retriever.hybrid import HybridRetriever
from app.llm.generator import Generator
from .state import DeepRAGState

from app.reranker.reranker import Reranker     # 使用上面简化版
from app.retriever.schemas import ContextAssemblyResult
import time

retriever = HybridRetriever()
reranker = Reranker()  # 默认使用 bge-reranker-large
generator = Generator()


def retrieve_node(state: DeepRAGState) -> DeepRAGState:
    """检索节点"""
    print(f"\n[Node 1] Retrieve | Question: {state['question']}")
    result = retriever.retrieve(
        query=state["question"],
        top_k=12,
        product_filter="医学影像产品"  # 当前只支持这一个产品，后续可动态
    )
    state["retrieval_result"] = result
    state["rewritten_query"] = state["question"]  # 暂不做复杂 rewrite
    return state


def rerank_node(state: DeepRAGState) -> DeepRAGState:
    """重排节点"""
    print(f"\n[Node 2] Rerank | 输入 {len(state['retrieval_result'].chunks)} 个 chunks")
    if not state["retrieval_result"].chunks:
        state["reranked_chunks"] = []
        return state

    reranked = reranker.rerank(
        query=state["rewritten_query"],
        chunks=state["retrieval_result"].chunks,
        top_k=6
    )
    state["reranked_chunks"] = reranked
    return state


def assemble_context_node(state: DeepRAGState) -> DeepRAGState:
    """上下文聚合节点（去重 + 简单压缩 + 结构化）"""
    print(f"\n[Node 3] Context Assembly | 输入 {len(state['reranked_chunks'])} 个 chunks")

    if not state["reranked_chunks"]:
        state["context_str"] = ""
        state["used_chunks"] = []
        state["has_evidence"] = False
        return state

    # 简单去重 + 拼接（后续可加 LLM 压缩）
    seen = set()
    unique_chunks = []
    context_parts = []

    for chunk in state["reranked_chunks"]:
        if chunk.text not in seen:
            seen.add(chunk.text)
            unique_chunks.append(chunk)
            context_parts.append(f"【来源：{chunk.product_name} - {chunk.doc_type}】\n{chunk.text}\n")

    state["used_chunks"] = unique_chunks
    state["context_str"] = "\n---\n".join(context_parts)
    state["has_evidence"] = len(unique_chunks) > 0
    return state


def generate_node(state: DeepRAGState) -> DeepRAGState:
    """生成节点（严格 grounded）"""
    print(f"\n[Node 4] Generate | 是否有证据: {state['has_evidence']}")

    if not state["has_evidence"] or not state["context_str"]:
        state["answer"] = "知识库中未找到相关依据，无法回答该问题。"
        state["citations"] = []
        state["confidence"] = 0.0
        return state

    answer = generator.generate(
        question=state["question"],
        context=state["context_str"]
    )

    # 简单构造 citations（可后续从 used_chunks + Postgres 补充更完整信息）
    citations = []
    for chunk in state["used_chunks"]:
        citations.append({
            "document_id": chunk.document_id,
            "filename": chunk.metadata.get("filename") or chunk.metadata.get("source", "product_manual.pdf"),
            "section_title": chunk.metadata.get("section_title"),
            "page_number": chunk.metadata.get("page"),
            "content_snippet": chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
        })

    state["answer"] = answer
    state["citations"] = citations
    state["confidence"] = min(0.95, len(state["used_chunks"]) * 0.2)  # 简单置信度计算，后续可优化
    return state