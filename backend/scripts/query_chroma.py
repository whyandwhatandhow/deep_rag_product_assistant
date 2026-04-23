import chromadb
import os
from pathlib import Path
import sys

# 自动定位项目根目录，与 ingest.py 保持一致
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]   # scripts/ → backend/ → root
sys.path.insert(0, str(project_root / "backend"))

# 导入配置
from app.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings

# 设置工作目录为项目根目录
os.chdir(project_root)
print(f"当前工作目录: {os.getcwd()}")

# 连接到 Chroma 数据库
chroma_path = "backend/data/chroma_db"
print(f"Chroma 数据库路径: {chroma_path}")
print(f"Chroma 数据库绝对路径: {os.path.abspath(chroma_path)}")
print(f"Chroma 数据库目录是否存在: {os.path.exists(chroma_path)}")

client = chromadb.PersistentClient(path=chroma_path)

# 初始化嵌入模型，与 ingest.py 保持一致
embeddings = HuggingFaceEmbeddings(
    model_name=settings.embedding_model,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)
print(f"使用嵌入模型: {settings.embedding_model}")

# 列出所有集合
print("所有可用的集合:")
collections = client.list_collections()
for coll in collections:
    print(f"- {coll.name}")

# 选择集合（如果存在）
collection_name = "product_knowledge"
try:
    collection = client.get_collection(name=collection_name)
    print(f"\n成功连接到集合: {collection_name}")
    
    # 1. 查看集合信息
    print(f"集合名称: {collection.name}")
    print(f"文档数量: {collection.count()}")
    
    # 2. 查看所有文档（限制前 10 条）
    print("\n前 10 条文档:")
    data = collection.get(limit=10)
    for i, (doc, meta) in enumerate(zip(data['documents'], data['metadatas'])):
        print(f"\n文档 {i+1}:")
        print(f"内容: {doc[:100]}...")  # 只显示前 100 个字符
        print(f"元数据: {meta}")
    
    # 3. 相似度搜索
    print("\n相似度搜索结果（查询: '医学影像'）:")
    # 使用与索引时相同的嵌入模型生成查询向量
    query_text = "医学影像"
    query_embedding = embeddings.embed_query(query_text)
    print(f"查询向量维度: {len(query_embedding)}")
    
    results = collection.query(
        query_embeddings=[query_embedding],  # 使用预生成的嵌入
        n_results=3
    )
    for i, (doc, meta, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
        print(f"\n结果 {i+1} (相似度: {1-distance:.4f}):")
        print(f"内容: {doc[:100]}...")
        print(f"元数据: {meta}")
except Exception as e:
    print(f"\n错误: {e}")
    print("请检查集合名称是否正确")