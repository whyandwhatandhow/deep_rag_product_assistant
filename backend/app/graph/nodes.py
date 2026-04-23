# backend/app/graph/nodes.py
from app.retriever.hybrid import HybridRetriever
from app.retriever.query_rewriter import QueryRewriter
from app.retriever.context_compressor import ContextCompressor
from app.llm.generator import Generator
from .state import DeepRAGState

from app.reranker.reranker import Reranker
from app.retriever.schemas import ContextAssemblyResult
import time

retriever = HybridRetriever()
reranker = Reranker()
generator = Generator()
query_rewriter = QueryRewriter(strategy="multi-query", num_queries=3)
context_compressor = ContextCompressor(max_context_length=4000)


def retrieve_node(state: DeepRAGState) -> DeepRAGState:
    """检索节点（带 Query Rewrite）"""
    question = state["question"]
    print(f"\n[Node 1] Retrieve | Question: {question}")

    rewritten_queries = query_rewriter.rewrite(question)
    state["rewritten_query"] = rewritten_queries[0] if rewritten_queries else question
    print(f"[Node 1] Query Rewrite | 生成 {len(rewritten_queries)} 个查询变体")
    for i, q in enumerate(rewritten_queries, 1):
        print(f"  - 查询 {i}: {q}")

    all_chunks = []
    seen_texts = set()

    for idx, query in enumerate(rewritten_queries):
        result = retriever.retrieve(
            query=query,
            top_k=50
        )
        for chunk in result.chunks:
            chunk_key = f"{chunk.document_id}_{chunk.chunk_id}"
            if chunk_key not in seen_texts:
                seen_texts.add(chunk_key)
                all_chunks.append(chunk)

        if idx == 0:
            state["retrieval_result"] = result

    from app.retriever.schemas import RetrievalResult
    state["retrieval_result"] = RetrievalResult(
        query=question,
        chunks=all_chunks,
        total_retrieved=len(all_chunks),
        used_metadata_filter=False,
        retrieval_time_ms=0
    )

    print(f"[Node 1] 合并检索完成 | 共 {len(all_chunks)} 个唯一 chunks")
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
        top_k=10
    )
    state["reranked_chunks"] = reranked
    return state


def assemble_context_node(state: DeepRAGState) -> DeepRAGState:
    """上下文聚合节点（使用 ContextCompressor）"""
    print(f"\n[Node 3] Context Assembly | 输入 {len(state['reranked_chunks'])} 个 chunks")

    if not state["reranked_chunks"]:
        state["context_str"] = ""
        state["used_chunks"] = []
        state["has_evidence"] = False
        return state

    compressed_chunks, context_str = context_compressor.compress(
        chunks=state["reranked_chunks"],
        query=state["question"]
    )

    state["used_chunks"] = compressed_chunks
    state["context_str"] = context_str
    state["has_evidence"] = len(compressed_chunks) > 0

    print(f"[Node 3] 上下文压缩完成 | 保留 {len(compressed_chunks)} 个 chunks | 上下文长度: {len(context_str)} 字符")
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
    state["confidence"] = min(0.95, len(state["used_chunks"]) * 0.2)
    return state