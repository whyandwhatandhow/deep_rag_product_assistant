# backend/app/reranker/reranker.py
from typing import List
from app.retriever.schemas import RetrievedChunk
import time


class Reranker:
    """简化版 Reranker（兼容当前环境）"""

    def __init__(self, model_name: str = "simple"):
        self.model_name = model_name
        print(f"Reranker 初始化完成 | 模式: {model_name}（简化版，后续可换 BGE）")

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 6) -> List[RetrievedChunk]:
        start = time.time()

        if not chunks:
            return []

        # 简化重排：当前使用原始向量 score 排序（后续替换为真实 reranker）
        # 如果没有 score，则使用均匀分数
        for chunk in chunks:
            if chunk.score == 0.0:
                chunk.score = 0.85 - (chunks.index(chunk) * 0.05)  # 简单衰减

        # 按 score 降序排序
        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)[:top_k]

        for i, chunk in enumerate(reranked):
            chunk.rank = i + 1

        rerank_time = int((time.time() - start) * 1000)
        print(f"Rerank 完成（简化版） | 返回 top-{len(reranked)} | 耗时 {rerank_time}ms")
        return reranked