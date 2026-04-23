# backend/app/retriever/context_compressor.py
from typing import List, Optional, Tuple
from app.retriever.schemas import RetrievedChunk


class ContextCompressor:
    def __init__(
        self,
        max_context_length: int = 8000,
        min_chunk_length: int = 30,
        compression_ratio: float = 0.7
    ):
        self.max_context_length = max_context_length
        self.min_chunk_length = min_chunk_length
        self.compression_ratio = compression_ratio

    def compress(
        self,
        chunks: List[RetrievedChunk],
        query: Optional[str] = None
    ) -> Tuple[List[RetrievedChunk], str]:
        """压缩上下文，保留最相关的 chunks"""
        if not chunks:
            return [], ""

        chunks = self._filter_short_chunks(chunks)
        chunks = self._deduplicate_chunks(chunks)
        chunks = self._score_and_rank_chunks(chunks, query)
        chunks = self._fit_to_context_limit(chunks)

        context_str = self._build_context_string(chunks)
        return chunks, context_str

    def _filter_short_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """过滤过短的 chunks，但如果过滤后太少则保留"""
        filtered = [
            chunk for chunk in chunks
            if len(chunk.text.strip()) >= self.min_chunk_length
        ]
        if len(filtered) < 2 and len(chunks) > 0:
            return chunks
        return filtered

    def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """去重 chunks"""
        seen_texts = set()
        unique_chunks = []
        for chunk in chunks:
            normalized = chunk.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_chunks.append(chunk)
        return unique_chunks

    def _score_and_rank_chunks(
        self,
        chunks: List[RetrievedChunk],
        query: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """根据相关性评分和排名"""
        if not query:
            return chunks

        query_terms = set(query.lower().split())

        def calculate_relevance_score(chunk: RetrievedChunk) -> float:
            score = 0.0
            text_lower = chunk.text.lower()

            if chunk.score > 0:
                score += chunk.score * 0.5

            for term in query_terms:
                if term in text_lower:
                    score += 0.1
                    if term in chunk.product_name.lower():
                        score += 0.15

            title = chunk.metadata.get("section_title", "").lower()
            if title and any(term in title for term in query_terms):
                score += 0.2

            return score

        scored_chunks = [
            (chunk, calculate_relevance_score(chunk))
            for chunk in chunks
        ]

        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in scored_chunks]

    def _fit_to_context_limit(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """将 chunks 限制在上下文长度内"""
        result = []
        current_length = 0

        for chunk in chunks:
            chunk_length = len(chunk.text)
            if current_length + chunk_length <= self.max_context_length:
                result.append(chunk)
                current_length += chunk_length
            else:
                remaining = self.max_context_length - current_length
                if remaining >= self.min_chunk_length:
                    truncated_chunk = RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        product_name=chunk.product_name,
                        doc_type=chunk.doc_type,
                        text=chunk.text[:remaining] + "...",
                        metadata=chunk.metadata,
                        score=chunk.score,
                        rank=chunk.rank
                    )
                    result.append(truncated_chunk)
                break

        return result

    def _build_context_string(self, chunks: List[RetrievedChunk]) -> str:
        """构建结构化上下文字符串"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            source_info = f"【{i}. 来源：{chunk.product_name} - {chunk.doc_type}】"
            if chunk.metadata.get("section_title"):
                source_info += f" | 章节：{chunk.metadata['section_title']}"
            if chunk.metadata.get("page"):
                source_info += f" | 页码：{chunk.metadata['page']}"

            context_parts.append(f"{source_info}\n{chunk.text}\n")

        return "\n---\n".join(context_parts)

    def compress_with_mmr(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        diversity_weight: float = 0.5
    ) -> List[RetrievedChunk]:
        """使用 Maximal Marginal Relevance (MMR) 进行压缩"""
        if not chunks:
            return []

        selected = []
        remaining = list(chunks)

        query_terms = set(query.lower().split())

        def get_relevance(chunk: RetrievedChunk) -> float:
            text_lower = chunk.text.lower()
            return sum(0.1 for term in query_terms if term in text_lower)

        def get_diversity(chunk1: RetrievedChunk, chunk2: RetrievedChunk) -> float:
            words1 = set(chunk1.text.lower().split())
            words2 = set(chunk2.text.lower().split())
            if not words1 or not words2:
                return 0.0
            overlap = len(words1 & words2)
            return overlap / len(words1 | words2)

        max_selected = int(len(chunks) * self.compression_ratio)

        while remaining and len(selected) < max_selected:
            best_score = -float('inf')
            best_chunk = None

            for chunk in remaining:
                relevance = get_relevance(chunk)
                diversity = min(
                    get_diversity(chunk, selected_chunk)
                    for selected_chunk in selected
                ) if selected else 0

                mmr_score = (1 - diversity_weight) * relevance + diversity_weight * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_chunk = chunk

            if best_chunk:
                selected.append(best_chunk)
                remaining.remove(best_chunk)

        return selected
