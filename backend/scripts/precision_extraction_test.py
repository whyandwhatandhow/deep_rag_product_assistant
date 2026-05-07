import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.retriever.hybrid import HybridRetriever
from app.llm.generator import Generator

print("\n=== 精准提取测试 ===\n")

retriever = HybridRetriever()
generator = Generator()

# 测试1：使用英文关键词检索（因为数据库是英文内容）
print("测试1：使用英文检索词")
query_en = "Zn Mg alloy mechanical properties tensile strength composition"
result_en = retriever.retrieve(query_en, top_k=3)

print(f"检索到 {result_en.total_retrieved} 个文档")

if result_en.chunks:
    first_chunk = result_en.chunks[0]
    print(f"\n第一个结果内容预览:\n{first_chunk.text[:500]}...")

# 测试2：直接用论文中的具体内容检索
print("\n\n测试2：用具体合金成分检索")
query_composition = "Zn-3.0Mg alloy ultimate tensile strength 280 MPa"
result_comp = retriever.retrieve(query_composition, top_k=3)
print(f"检索到 {result_comp.total_retrieved} 个文档")

if result_comp.chunks:
    first = result_comp.chunks[0]
    print(f"\n第一个结果:\n{first.text[:400]}...")

# 测试3：模拟完整提取流程
print("\n\n测试3：完整提取测试")

# 用一个能找到实际论文内容的查询
test_query = "Al-Zn-Mg alloy welding mechanical properties composition"
result = retriever.retrieve(test_query, top_k=5)

if result.chunks:
    # 构建上下文
    context_parts = []
    for i, chunk in enumerate(result.chunks):
        context_parts.append(f"[Chunk {i+1}]\n{chunk.text}")
    context = "\n\n".join(context_parts)

    # 提取提示词
    extraction_prompt = f"""You are a materials science expert. Extract structured data about Zn/Mg/Al alloys from the following paper content.

Paper Content:
{context}

Extract the following information in JSON format:
- alloy_system: The alloy system (e.g., Zn-Mg, Al-Zn-Mg)
- composition: Dict with element names as keys and weight percentages as values
- mechanical_properties: Dict with property names and values
- processing: Any processing parameters mentioned

Return ONLY valid JSON, no explanations."""

    print(f"构建的上下文长度: {len(context)} 字符")
    print(f"上下文预览（前200字符）:\n{context[:200]}...")

    # 调用生成器
    response = generator.generate(question=extraction_prompt, context="")

    print(f"\n\nLLM响应:\n{response}")

print("\n\n=== 诊断完成 ===")
print("如果上述测试能成功检索和提取，说明系统工作正常")
print("如果检索结果为空或内容不相关，说明需要优化检索策略")