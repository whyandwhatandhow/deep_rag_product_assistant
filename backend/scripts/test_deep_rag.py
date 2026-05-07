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
        "我收集的这些文献都是大概哪个领域的内容呢？",
        "我收集的这些文献都是关于哪个主题的呢？",
        "我想通过你获取的数据画预测拟合图，你认为可以吗？",
        "我如果想预测一个合金系统的性能指标，我需要考虑哪些因素？，你可以给我一些提示吗？"
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