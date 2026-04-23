# backend/app/eval/evaluator.py
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from .metrics import RAGEvaluator, RetrievalEvaluator, AnswerQualityEvaluator


class EvaluationResult:
    def __init__(self, query: str, metrics: Dict[str, Any], timestamp: str = None):
        self.query = query
        self.metrics = metrics
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }


class TestDataset:
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path
        self.test_cases: List[Dict[str, Any]] = []
        if dataset_path:
            self.load(dataset_path)

    def load(self, dataset_path: str):
        """从文件加载测试数据集"""
        path = Path(dataset_path)
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                self.test_cases = json.load(f)
        elif path.suffix in [".csv", ".txt"]:
            self.test_cases = self._load_csv(path)

    def _load_csv(self, path: Path) -> List[Dict[str, Any]]:
        """加载 CSV 格式的测试数据"""
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                headers = lines[0].strip().split(",")
                for line in lines[1:]:
                    values = line.strip().split(",")
                    if len(values) >= len(headers):
                        case = dict(zip(headers, values))
                        cases.append(case)
        return cases

    def add_case(self, case: Dict[str, Any]):
        """添加测试用例"""
        self.test_cases.append(case)

    def save(self, dataset_path: str):
        """保存测试数据集"""
        path = Path(dataset_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.test_cases, f, ensure_ascii=False, indent=2)

    def get_cases_by_product(self, product_name: str) -> List[Dict[str, Any]]:
        """按产品名筛选测试用例"""
        return [
            case for case in self.test_cases
            if case.get("product_name") == product_name
        ]


class RAGEvaluatorSuite:
    def __init__(self, test_dataset: Optional[TestDataset] = None):
        self.evaluator = RAGEvaluator()
        self.test_dataset = test_dataset or TestDataset()
        self.evaluation_results: List[EvaluationResult] = []

    def evaluate_single(
        self,
        question: str,
        retrieved_chunks: List[Any],
        relevant_chunk_ids: set,
        answer: str,
        context: str,
        citations: List[Dict],
        k: int = 10
    ) -> EvaluationResult:
        """评估单个查询"""
        metrics = self.evaluator.evaluate(
            question=question,
            retrieved_chunks=retrieved_chunks,
            relevant_chunk_ids=relevant_chunk_ids,
            answer=answer,
            context=context,
            citations=citations,
            k=k
        )

        result = EvaluationResult(query=question, metrics=metrics)
        self.evaluation_results.append(result)
        return result

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
        k: int = 10
    ) -> Dict[str, Any]:
        """批量评估"""
        all_metrics = {
            "retrieval": {
                "avg_recall": 0.0,
                "avg_precision": 0.0,
                "avg_mrr": 0.0,
                "avg_ndcg": 0.0,
                "avg_hit_rate": 0.0,
            },
            "answer_quality": {
                "avg_citation_coverage": 0.0,
                "avg_evidence_recall": 0.0,
                "avg_hallucination_score": 0.0,
                "avg_groundedness": 0.0,
            },
            "overall_avg_score": 0.0,
        }

        count = len(results)
        if count == 0:
            return all_metrics

        for result in results:
            metrics = result.get("metrics", {})
            retrieval = metrics.get("retrieval_metrics", {})
            answer_quality = metrics.get("answer_quality_metrics", {})

            all_metrics["retrieval"]["avg_recall"] += retrieval.get("recall_at_k", 0)
            all_metrics["retrieval"]["avg_precision"] += retrieval.get("precision_at_k", 0)
            all_metrics["retrieval"]["avg_mrr"] += retrieval.get("mrr", 0)
            all_metrics["retrieval"]["avg_ndcg"] += retrieval.get("ndcg_at_k", 0)
            all_metrics["retrieval"]["avg_hit_rate"] += retrieval.get("hit_rate", 0)

            all_metrics["answer_quality"]["avg_citation_coverage"] += answer_quality.get("citation_coverage", 0)
            all_metrics["answer_quality"]["avg_evidence_recall"] += answer_quality.get("evidence_recall", 0)
            all_metrics["answer_quality"]["avg_hallucination_score"] += answer_quality.get("hallucination_score", 0)
            all_metrics["answer_quality"]["avg_groundedness"] += answer_quality.get("groundedness_score", 0)

            all_metrics["overall_avg_score"] += metrics.get("overall_score", 0)

        for key in all_metrics["retrieval"]:
            all_metrics["retrieval"][key] /= count
        for key in all_metrics["answer_quality"]:
            all_metrics["answer_quality"][key] /= count
        all_metrics["overall_avg_score"] /= count

        return all_metrics

    def generate_report(self) -> str:
        """生成评估报告"""
        if not self.evaluation_results:
            return "暂无评估结果"

        results_dicts = [r.to_dict() for r in self.evaluation_results]
        batch_metrics = self.evaluate_batch(results_dicts)

        report_lines = [
            "=" * 60,
            "RAG 系统评估报告",
            "=" * 60,
            "",
            f"评估用例数量: {len(self.evaluation_results)}",
            "",
            "检索性能指标:",
            f"  - 平均召回率 (Recall@K): {batch_metrics['retrieval']['avg_recall']:.4f}",
            f"  - 平均精确率 (Precision@K): {batch_metrics['retrieval']['avg_precision']:.4f}",
            f"  - 平均 MRR: {batch_metrics['retrieval']['avg_mrr']:.4f}",
            f"  - 平均 NDCG@K: {batch_metrics['retrieval']['avg_ndcg']:.4f}",
            f"  - 命中率: {batch_metrics['retrieval']['avg_hit_rate']:.4f}",
            "",
            "答案质量指标:",
            f"  - 平均引用覆盖率: {batch_metrics['answer_quality']['avg_citation_coverage']:.4f}",
            f"  - 平均证据召回率: {batch_metrics['answer_quality']['avg_evidence_recall']:.4f}",
            f"  - 平均幻觉分数: {batch_metrics['answer_quality']['avg_hallucination_score']:.4f}",
            f"  - 平均可靠程度: {batch_metrics['answer_quality']['avg_groundedness']:.4f}",
            "",
            f"综合得分: {batch_metrics['overall_avg_score']:.4f}",
            "",
            "=" * 60,
        ]

        return "\n".join(report_lines)
