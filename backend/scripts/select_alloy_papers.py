import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retriever.hybrid import HybridRetriever
from collections import defaultdict

retriever = HybridRetriever()

print("\n=== 论文筛选分析 ===")
print(f"Collection 名称: {retriever.collection_name}")

collection = retriever.vectorstore._collection
total_chunks = collection.count()
print(f"总 Chunk 数量: {total_chunks}")

data = collection.get(limit=min(total_chunks, 40000), include=["metadatas", "documents"])

unique_docs = {}
for i, meta in enumerate(data["metadatas"]):
    doc_id = meta.get("document_id", "unknown")
    filename = meta.get("source_file") or meta.get("filename") or "unknown"
    title = meta.get("title", "")
    product_name = meta.get("product_name", "")
    doc_type = meta.get("doc_type", "")

    if doc_id not in unique_docs:
        unique_docs[doc_id] = {
            "filename": filename,
            "title": title,
            "product_name": product_name,
            "doc_type": doc_type,
            "chunks": 1
        }
    else:
        unique_docs[doc_id]["chunks"] += 1

print(f"\n唯一文档数量: {len(unique_docs)}")

by_type = defaultdict(list)
for doc_id, info in unique_docs.items():
    by_type[info["doc_type"]].append(info)

print("\n按文档类型分布:")
for doc_type, docs in sorted(by_type.items()):
    print(f"  {doc_type or '未分类'}: {len(docs)} 篇")

print("\n部分文档示例 (前20篇):")
for i, (doc_id, info) in enumerate(list(unique_docs.items())[:20]):
    print(f"  {i+1}. [{info['doc_type']}] {info['filename']}")
    print(f"     产品: {info['product_name']} | Chunks: {info['chunks']}")

print("\n关键词搜索 - 筛选锌相关论文:")

zn_keywords = ["zn", "zinc", "mg", "al-zn", "al-zn-mg", "mg-zn", "zn-mg"]
zn_papers = defaultdict(list)

for doc_id, info in unique_docs.items():
    filename_lower = info["filename"].lower()
    title_lower = info["title"].lower()
    product_lower = info["product_name"].lower()

    for kw in zn_keywords:
        if kw in filename_lower or kw in title_lower or kw in product_lower:
            zn_papers[kw].append(info)
            break

print("\n包含关键词的论文:")
for kw, papers in sorted(zn_papers.items(), key=lambda x: -len(x[1])):
    print(f"  '{kw}': {len(papers)} 篇")

by_product = defaultdict(list)
for doc_id, info in unique_docs.items():
    product_name = info["product_name"].lower()
    by_product[product_name].append(info)

print("\n按产品名称分类:")
for product, papers in sorted(by_product.items(), key=lambda x: -len(x[1])):
    print(f"  '{product}': {len(papers)} 篇")

pure_zn_papers = []
al_zn_mg_papers = []
mg_only_papers = []
other_papers = []

for doc_id, info in unique_docs.items():
    product_lower = info["product_name"].lower()
    filename_lower = info["filename"].lower()

    if "zn" in product_lower or "zinc" in product_lower:
        if "al-zn" in product_lower or "alzn" in product_lower:
            al_zn_mg_papers.append(info)
        elif "mg" in product_lower or "mg" in filename_lower:
            al_zn_mg_papers.append(info)
        else:
            pure_zn_papers.append(info)
    elif "mg" in product_lower or "mg" in filename_lower:
        if "zn" in product_lower or "zn" in filename_lower:
            al_zn_mg_papers.append(info)
        else:
            mg_only_papers.append(info)
    else:
        other_papers.append(info)

print(f"\n纯锌(Zn)合金论文: {len(pure_zn_papers)} 篇")
print(f"铝锌镁(Al-Zn-Mg)合金论文: {len(al_zn_mg_papers)} 篇")
print(f"纯镁(Mg)合金论文: {len(mg_only_papers)} 篇")
print(f"其他论文: {len(other_papers)} 篇")

total_zn_alloy = len(pure_zn_papers) + len(al_zn_mg_papers)
print(f"\n筛选建议:")
print(f"  - 锌合金研发核心关注: Zn({len(pure_zn_papers)}) + Al-Zn-Mg({len(al_zn_mg_papers)}) = {total_zn_alloy} 篇")
print(f"  - Mg论文可作为对比参考: {len(mg_only_papers)} 篇")
print(f"  - 总计导入论文: {len(unique_docs)} 篇")

if pure_zn_papers:
    print(f"\n纯锌论文列表 (共{len(pure_zn_papers)}篇):")
    for i, p in enumerate(pure_zn_papers[:10], 1):
        filename = p.get('filename', 'unknown')
        title = p.get('title', '')
        print(f"  {i}. {filename}")
        if title:
            print(f"     标题: {title[:50]}..." if len(title) > 50 else f"     标题: {title}")
    if len(pure_zn_papers) > 10:
        print(f"  ... 还有 {len(pure_zn_papers) - 10} 篇")