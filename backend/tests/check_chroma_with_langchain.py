from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# 检查当前工作目录
print(f"当前工作目录: {os.getcwd()}")

# 初始化 Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

# 检查 product_knowledge 集合
chroma_path = "../data/chroma_db"
print(f"Chroma 数据库路径: {chroma_path}")
print(f"路径是否存在: {os.path.exists(chroma_path)}")

try:
    # 初始化 Chroma
    vectorstore = Chroma(
        collection_name="product_knowledge",
        embedding_function=embeddings,
        persist_directory=chroma_path,
    )
    print("成功初始化 Chroma")
    
    # 获取集合中的文档数量
    count = vectorstore._collection.count()
    print(f"集合中的文档数量: {count}")
    
    # 如果有文档，获取前 3 个
    if count > 0:
        print("前 3 个文档:")
        docs = vectorstore.similarity_search("测试", k=3)
        for i, doc in enumerate(docs):
            print(f"文档 {i+1}:")
            print(f"内容: {doc.page_content[:100]}...")
            print(f"元数据: {doc.metadata}")
            print()
    else:
        print("集合中没有文档")
        
    # 检查所有集合
    print("\n所有集合:")
    for col_name in vectorstore._client.list_collections():
        print(f"集合名称: {col_name.name}")
        print(f"文档数量: {col_name.count()}")
        print()
        
except Exception as e:
    print(f"错误: {e}")
