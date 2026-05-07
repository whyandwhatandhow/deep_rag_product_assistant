"""检查Chroma数据库中的文档数量"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retriever.hybrid import HybridRetriever


def check_chroma_db():
    """检查Chroma数据库"""
    print("="*60)
    print("检查Chroma数据库")
    print("="*60)
    
    # 初始化检索器
    retriever = HybridRetriever()
    
    # 检查集合
    try:
        collection = retriever.client.get_collection(name=retriever.collection_name)
        total_docs = collection.count()
        print(f"集合 {retriever.collection_name} 中的文档数量: {total_docs}")
        
        # 获取一些样本
        print("\n获取5个文档样本:")
        samples = collection.get(limit=5)
        for i, doc in enumerate(samples['documents']):
            print(f"\n文档 {i+1}:")
            print(f"  内容: {doc[:100]}...")
            print(f"  元数据: {samples['metadatas'][i]}")
    except Exception as e:
        print(f"检查数据库时出错: {e}")


def main():
    """主函数"""
    check_chroma_db()


if __name__ == "__main__":
    main()
