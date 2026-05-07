import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 设置UTF-8输出
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.retriever.hybrid import HybridRetriever

retriever = HybridRetriever()
collection = retriever.vectorstore._collection

print("\n=== 数据库内容诊断 ===")
print(f"集合名称: {retriever.collection_name}")
print(f"总文档数: {collection.count()}")

# 获取几条样本数据，检查实际存储的内容
print("\n获取前10条文档样本:")
sample_data = collection.get(limit=10, include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(sample_data["documents"], sample_data["metadatas"])):
    print(f"\n--- 文档 {i+1} ---")
    print(f"元数据: {meta}")
    content_preview = doc[:300].replace('\n', ' ').replace('\r', ' ')
    print(f"内容预览 (前300字符): {content_preview}...")
    print(f"内容长度: {len(doc)} 字符")

# 检查文档类型分布
print("\n\n=== 按文档类型分布 ===")
all_data = collection.get(limit=10000, include=["metadatas"])

doc_types = {}
for meta in all_data["metadatas"]:
    doc_type = meta.get("doc_type") or meta.get("type", "unknown")
    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
    print(f"  {dt}: {count}")

# 检查是否有论文正文内容
print("\n\n=== 检查论文正文特征 ===")
has_abstract = 0
has_reference = 0
has_method = 0
has_result = 0

for doc in all_data["documents"]:
    doc_lower = doc.lower()
    if "abstract" in doc_lower or "摘要" in doc:
        has_abstract += 1
    if "reference" in doc_lower or "参考文献" in doc:
        has_reference += 1
    if "method" in doc_lower or "方法" in doc or "实验" in doc:
        has_method += 1
    if "result" in doc_lower or "结果" in doc or "性能" in doc:
        has_result += 1

print(f"包含摘要/Abstract的文档: {has_abstract}")
print(f"包含参考文献/Reference的文档: {has_reference}")
print(f"包含方法/Method的文档: {has_method}")
print(f"包含结果/Result的文档: {has_result}")

# 测试检索特定论文
print("\n\n=== 测试检索论文内容 ===")
test_queries = [
    "锌合金 成分 抗拉强度",
    "Zn-Mg 合金 力学性能",
    "Al-Zn-Mg 合金 时效",
    "锌合金 熔炼 铸造"
]

for query in test_queries:
    print(f"\n查询: {query}")
    result = retriever.retrieve(query, top_k=2)
    print(f"检索到 {result.total_retrieved} 个文档")
    if result.chunks:
        first_chunk = result.chunks[0]
        print(f"第一个结果预览: {first_chunk.text[:200]}...")