# backend/scripts/check_anatomask.py
import chromadb
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "backend"))

from app.core.config import settings

client = chromadb.PersistentClient(path='D:/pyprocesses/deep_rag_product_assistant/backend/data/chroma_db')
collection = client.get_collection(name='product_knowledge')

results = collection.get(limit=50)
print(f'总文档数: {collection.count()}')
print(f'获取的文档数: {len(results["documents"])}')

print('\n所有文档的产品名分布:')
product_names = {}
for meta in results['metadatas']:
    name = meta.get('product_name', 'Unknown')
    product_names[name] = product_names.get(name, 0) + 1

for name, count in product_names.items():
    print(f'  {name}: {count}')

print('\n包含 AnatoMask 的文档:')
anatomask_docs = []
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    if 'AnatoMask' in doc or 'anatamask' in doc.lower():
        anatomask_docs.append((doc, meta))
        print(f'\n文档 {i+1}:')
        print(f'内容长度: {len(doc)} 字符')
        print(f'内容前200字: {doc[:200]}...')
        print(f'产品名: {meta.get("product_name")}')
        print(f'文档ID: {meta.get("document_id")}')

print(f'\n总共找到 {len(anatomask_docs)} 个包含 AnatoMask 的文档')

# 测试相似度搜索 AnatoMask
print('\n测试相似度搜索 "AnatoMask":')
embeddings = __import__('langchain_huggingface', fromlist=['HuggingFaceEmbeddings']).HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={'normalize_embeddings': True}
)
query_embedding = embeddings.embed_query("AnatoMask")
search_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
print(f'搜索结果数: {len(search_results["documents"][0])}')
for i, (doc, meta, dist) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0], search_results["distances"][0])):
    similarity = 1 - dist
    print(f'\n结果 {i+1} (相似度: {similarity:.4f}):')
    print(f'内容: {doc[:200]}...')
    print(f'产品名: {meta.get("product_name")}')