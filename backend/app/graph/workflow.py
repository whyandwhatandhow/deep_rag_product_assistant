# backend/app/graph/workflow.py
from langgraph.graph import StateGraph, END
from .state import DeepRAGState
from .nodes import retrieve_node, rerank_node, assemble_context_node, generate_node


def build_deep_rag_workflow():
    """构建深 RAG 线性工作流"""
    workflow = StateGraph(DeepRAGState)

    # 添加节点
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("assemble_context", assemble_context_node)
    workflow.add_node("generate", generate_node)

    # 线性边（严格深 RAG，主链路无分支）
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "assemble_context")
    workflow.add_edge("assemble_context", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# 全局实例（推荐在 FastAPI startup 时初始化）
deep_rag_chain = build_deep_rag_workflow()