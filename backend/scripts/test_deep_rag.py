# backend/scripts/test_deep_rag.py
import sys
import os
import time

# ==================== 关键修复：把项目根目录加入 Python 路径 ====================
# 确保无论从哪里运行，都能找到 "backend" 这个包
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)  # 把 D:\pyprocesses\deep_rag_product_assistant 加进去

print(f"项目根目录已加入 sys.path: {project_root}")
# =============================================================================

from app.graph.workflow import deep_rag_chain

if __name__ == "__main__":
    print("深 RAG 学术论文助手 - 第三阶段端到端测试启动...\n")

    test_questions = [
        "AnatoMask 是什么？它的主要功能是什么？",
        "AnatoMask 是如何提高医学影像分割效果的？",
        "ResNet 在哪些数据集上进行了测试？",
        "AnatoMask 与其他医学影像分割方法相比有什么优势？"
        "ResNet 是什么？它的主要功能是什么？"
        "ResNet 是如何提高医学影像分割效果的？"
        "叶鸣镝是谁"
    ]

    for idx, q in enumerate(test_questions, 1):
        print(f"\n{'=' * 100}")
        print(f"测试 {idx}/4: {q}")
        start_time = time.time()

        initial_state = {
            "question": q,
            "rewritten_query": q,
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

        print(f"\n最终答案：\n{result.get('answer', '无答案')}")
        print(f"\n引用来源（{len(result.get('citations', []))} 条）：")
        for i, cit in enumerate(result.get('citations', []), 1):
            print(
                f"  {i}. 文档ID: {cit.get('document_id')} | 文件: {cit.get('filename')} | 片段: {cit.get('content_snippet', '')[:100]}...")

        print(f"置信度: {result.get('confidence', 0):.2f} | 有证据: {result.get('has_evidence', False)}")
        print(f"总耗时: {int((time.time() - start_time) * 1000)} ms")