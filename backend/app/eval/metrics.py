# backend/app/eval/metrics.py
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float


@dataclass
class AnswerQualityMetrics:
    citation_coverage: float
    evidence_recall: float
    hallucination_score: float
    groundedness_score: float


class RetrievalEvaluator:
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate_retrieval(
        self,
        retrieved_chunks: List[Any],
        relevant_chunk_ids: Set[str],
        k: int = 10
    ) -> RetrievalMetrics:
        """评估检索性能"""
        retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks[:k]]

        true_positives = len(set(retrieved_ids) & relevant_chunk_ids)
        total_relevant = len(relevant_chunk_ids)
        total_retrieved = len(retrieved_ids)

        recall = true_positives / total_relevant if total_relevant > 0 else 0.0
        precision = true_positives / total_retrieved if total_retrieved > 0 else 0.0

        mrr = self._calculate_mrr(retrieved_ids, relevant_chunk_ids)
        ndcg = self._calculate_ndcg(retrieved_ids, relevant_chunk_ids, k)
        hit_rate = 1.0 if true_positives > 0 else 0.0

        return RetrievalMetrics(
            recall_at_k=round(recall, 4),
            precision_at_k=round(precision, 4),
            mrr=round(mrr, 4),
            ndcg_at_k=round(ndcg, 4),
            hit_rate=round(hit_rate, 4)
        )

    def _calculate_mrr(self, retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """计算 MRR (Mean Reciprocal Rank)"""
        for i, chunk_id in enumerate(retrieved_ids, 1):
            if chunk_id in relevant_ids:
                return 1.0 / i
        return 0.0

    def _calculate_ndcg(
        self,
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:
        """计算 NDCG (Normalized Discounted Cumulative Gain)"""
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_ids[:k], 1):
            if chunk_id in relevant_ids:
                dcg += 1.0 / (i ** 0.5)

        idcg = sum(1.0 / (i ** 0.5) for i in range(1, min(len(relevant_ids), k) + 1))

        return dcg / idcg if idcg > 0 else 0.0


class AnswerQualityEvaluator:
    def __init__(self):
        self.hallucination_keywords = [
            "不确定", "可能", "也许", "大概", "我认为", "据我所知",
            "无法确定", "不清楚", "未知", "未提及", "无法确认"
        ]

    def evaluate_answer_quality(
        self,
        answer: str,
        context: str,
        question: str,
        citations: List[Dict]
    ) -> AnswerQualityMetrics:
        """评估答案质量"""
        citation_coverage = self._calculate_citation_coverage(citations, context)
        evidence_recall = self._calculate_evidence_recall(answer, context)
        hallucination_score = self._detect_hallucination(answer, context)
        groundedness_score = self._calculate_groundedness(answer, context)

        return AnswerQualityMetrics(
            citation_coverage=round(citation_coverage, 4),
            evidence_recall=round(evidence_recall, 4),
            hallucination_score=round(hallucination_score, 4),
            groundedness_score=round(groundedness_score, 4)
        )

    def _calculate_citation_coverage(
        self,
        citations: List[Dict],
        context: str
    ) -> float:
        """计算引用覆盖率"""
        if not citations or not context:
            return 0.0

        context_lower = context.lower()
        covered_count = 0

        for citation in citations:
            snippet = citation.get("content_snippet", "").lower()
            if snippet and snippet in context_lower:
                covered_count += 1

        return covered_count / len(citations) if citations else 0.0

    def _calculate_evidence_recall(self, answer: str, context: str) -> float:
        """计算证据召回率"""
        if not answer or not context:
            return 0.0

        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        matching_words = answer_words & context_words
        return len(matching_words) / len(answer_words)

    def _detect_hallucination(self, answer: str, context: str) -> float:
        """检测幻觉分数（越低越好）"""
        if not answer or not context:
            return 1.0

        hallucination_count = 0
        answer_lower = answer.lower()

        for keyword in self.hallucination_keywords:
            if keyword in answer_lower:
                hallucination_count += 1

        uncertainty_phrases = [
            r"建议咨询.*专家",
            r"请.*确认",
            r"请联系.*获取",
            r"建议.*进一步",
        ]

        for phrase in uncertainty_phrases:
            if re.search(phrase, answer_lower):
                hallucination_count += 0.5

        hallucination_score = min(1.0, hallucination_count / 3.0)
        return hallucination_score

    def _calculate_groundedness(self, answer: str, context: str) -> float:
        """计算答案基于上下文的可靠程度"""
        if not answer or not context:
            return 0.0

        context_lower = context.lower()
        answer_lower = answer.lower()

        claim_patterns = [
            r"\d+%",
            r"\d+个",
            r"\d+年",
            r"\d+月",
            r"\d+日",
            r"提高了\d+",
            r"降低了\d+",
            r"增加了\d+",
        ]

        factual_claims = []
        supported_claims = 0

        for pattern in claim_patterns:
            matches = re.findall(pattern, answer_lower)
            factual_claims.extend(matches)

        for claim in factual_claims:
            if claim in context_lower:
                supported_claims += 1

        if not factual_claims:
            return 0.8

        return supported_claims / len(factual_claims)


class RAGEvaluator:
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.answer_evaluator = AnswerQualityEvaluator()

    def evaluate(
        self,
        question: str,
        retrieved_chunks: List[Any],
        relevant_chunk_ids: Set[str],
        answer: str,
        context: str,
        citations: List[Dict],
        k: int = 10
    ) -> Dict[str, Any]:
        """完整 RAG 评估"""
        retrieval_metrics = self.retrieval_evaluator.evaluate_retrieval(
            retrieved_chunks, relevant_chunk_ids, k
        )

        answer_metrics = self.answer_evaluator.evaluate_answer_quality(
            answer, context, question, citations
        )

        overall_score = (
            retrieval_metrics.recall_at_k * 0.4 +
            answer_metrics.groundedness_score * 0.4 +
            (1 - answer_metrics.hallucination_score) * 0.2
        )

        return {
            "retrieval_metrics": {
                "recall_at_k": retrieval_metrics.recall_at_k,
                "precision_at_k": retrieval_metrics.precision_at_k,
                "mrr": retrieval_metrics.mrr,
                "ndcg_at_k": retrieval_metrics.ndcg_at_k,
                "hit_rate": retrieval_metrics.hit_rate,
            },
            "answer_quality_metrics": {
                "citation_coverage": answer_metrics.citation_coverage,
                "evidence_recall": answer_metrics.evidence_recall,
                "hallucination_score": answer_metrics.hallucination_score,
                "groundedness_score": answer_metrics.groundedness_score,
            },
            "overall_score": round(overall_score, 4)
        }
