import chromadb
from app.core.config import settings

def check_chroma():
    # 1. 检查 Docker 容器里的 Chroma
    print("--- 正在检查 Docker 容器 (Port 8001) ---")
    try:
        http_client = chromadb.HttpClient(host='localhost', port=8001)
        collections = http_client.list_collections()
        print(f"找到集合数量: {len(collections)}")
        for col in collections:
            print(f"集合名称: {col.name} | 数据条数: {col.count()}")
    except Exception as e:
        print(f"无法连接 Docker Chroma: {e}")

    # 2. 检查本地磁盘路径的 Chroma
    print("\n--- 正在检查本地磁盘路径 (settings.chroma_persist_directory) ---")
    try:
        local_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
        collections = local_client.list_collections()
        print(f"找到集合数量: {len(collections)}")
        for col in collections:
            print(f"集合名称: {col.name} | 数据条数: {col.count()}")
    except Exception as e:
        print(f"无法读取本地磁盘路径: {e}")

if __name__ == "__main__":
    check_chroma()