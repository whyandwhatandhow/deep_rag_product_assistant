# backend/scripts/debug_chroma_metadata.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retriever.hybrid import HybridRetriever

retriever = HybridRetriever()

print("\n🔍 === Chroma 完整诊断 ===")
print(f"持久化路径: {retriever.persist_dir}")
print(f"Collection 名称: {retriever.collection_name}")

# 1. 查看所有 collection
collections = retriever.client.list_collections()
print(f"当前 Chroma 中共有 {len(collections)} 个 collection：")
for coll in collections:
    print(f"   • {coll.name}")

# 2. 当前 collection 的统计
collection = retriever.vectorstore._collection
print(f"\n当前 collection '{retriever.collection_name}' 总 chunk 数量: {collection.count()}")

# 3. 取出前 5 条 metadata（关键！）
data = collection.get(limit=332, include=["metadatas", "documents"])

if not data["ids"]:
    print("❌ 当前 collection 为空！没有找到任何 chunk")
else:
    print(f"\n✅ 找到 {len(data['ids'])} 条记录，前 3 条 metadata keys：")
    for i in range(min(322, len(data["ids"]))):
        meta = data["metadatas"][i]
        print(f"Chunk {i+1}:")
        print(f"   keys → {list(meta.keys())}")
        print(f"   product 字段 → {meta.get('product') or meta.get('product_name') or '【未找到】'}")
        print(f"   document_id → {meta.get('document_id')}")
        print(f"   text preview → {data['documents'][i][:120]}...")
        print("-" * 80)