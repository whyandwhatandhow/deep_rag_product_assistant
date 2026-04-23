import chromadb
import os

# 检查当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 连接到 Chroma 数据库
chroma_path = "../data/chroma_db"
print(f"Chroma 数据库路径: {chroma_path}")
print(f"路径是否存在: {os.path.exists(chroma_path)}")

try:
    # 使用原生 Chroma 客户端
    client = chromadb.PersistentClient(path=chroma_path)
    print("成功连接到 Chroma 数据库")
    
    # 列出所有集合
    collections = client.list_collections()
    print(f"找到集合数量: {len(collections)}")
    
    for col in collections:
        print(f"集合名称: {col.name}")
        print(f"文档数量: {col.count()}")
        
        # 查看前 3 条数据
        if col.count() > 0:
            print("前 3 条数据:")
            data = col.get(limit=3)
            print(f"ID: {data['ids']}")
            print(f"元数据: {data['metadatas']}")
            print(f"文档: {[doc[:100] + '...' if len(doc) > 100 else doc for doc in data['documents']]}")
            print()
        else:
            print("集合中没有文档")
            print()
            
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
