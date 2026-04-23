# backend/app/api/eval_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.eval import RAGEvaluatorSuite, TestDataset

router = APIRouter(prefix="/api/v1/eval", tags=["评估"])


class EvaluationCase(BaseModel):
    question: str
    relevant_chunk_ids: List[str]
    answer: Optional[str] = None
    context: Optional[str] = None
    citations: Optional[List[Dict]] = []


class BatchEvaluationRequest(BaseModel):
    cases: List[EvaluationCase]
    k: int = 10


class EvaluationResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    cases: List[EvaluationCase],
    k: int = 10
):
    """
    评估接口

    - 接收测试用例列表
    - 返回评估指标和汇总
    """
    evaluator = RAGEvaluatorSuite()

    results = []
    for case in cases:
        result = evaluator.evaluate_single(
            question=case.question,
            retrieved_chunks=[],  # 需要实际检索结果
            relevant_chunk_ids=set(case.relevant_chunk_ids),
            answer=case.answer or "",
            context=case.context or "",
            citations=case.citations or [],
            k=k
        )
        results.append(result.to_dict())

    summary = evaluator.evaluate_batch(results, k)

    return EvaluationResponse(
        results=results,
        summary=summary
    )


@router.get("/report")
async def get_evaluation_report():
    """获取评估报告"""
    evaluator = RAGEvaluatorSuite()
    return {
        "report": evaluator.generate_report(),
        "total_evaluations": len(evaluator.evaluation_results)
    }


@router.post("/testset")
async def load_testset(file_path: str):
    """加载测试数据集"""
    try:
        test_dataset = TestDataset(file_path)
        return {
            "status": "success",
            "total_cases": len(test_dataset.test_cases),
            "products": list(set(c.get("product_name") for c in test_dataset.test_cases))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载测试集失败: {str(e)}")