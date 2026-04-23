# backend/app/retriever/hybrid.py
import time
import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma  # ← 改用新包，解决 deprecation
from langchain_huggingface import HuggingFaceEmbeddings  # ← 新增
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Optional
from app.core.config import settings
from .schemas import RetrievalResult, RetrievedChunk


class HybridRetriever:
    """混合检索器（单例模式）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if HybridRetriever._initialized:
            return
        
        HybridRetriever._initialized = True
        
        # Embedding 函数（必须和 ingest 时一致）
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,  # BAAI/bge-small-zh-v1.5
            model_kwargs={"device": "cpu"}
        )

        # Chroma 初始化（关键修复：使用 PersistentClient 连接本地数据库）
        # 与 ingest.py 保持一致，计算绝对路径
        from pathlib import Path
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # hybrid.py → retriever/ → app/ → backend/ → root
        self.persist_dir = str(project_root / settings.chroma_persist_directory)
        print(f"项目根目录: {project_root}")
        print(f"Chroma 持久化目录（配置）: {settings.chroma_persist_directory}")
        print(f"Chroma 持久化目录（绝对路径）: {self.persist_dir}")
        print(f"Chroma 持久化目录是否存在: {os.path.exists(self.persist_dir)}")
        
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 查看所有集合
        collections = self.client.list_collections()
        print(f"所有可用的集合: {[coll.name for coll in collections]}")
        
        # 注释掉 HttpClient 连接，使用本地文件系统数据库
        # self.client = chromadb.HttpClient(host='localhost', port=8001)
        self.collection_name = "product_knowledge"
        
        # 检查集合是否存在
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f"集合 {self.collection_name} 存在，文档数量: {collection.count()}")
        except Exception as e:
            print(f"集合 {self.collection_name} 不存在: {e}")

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_function,  # ← 必须添加这一行！
        )

        # PostgreSQL 初始化
        self.db_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        self.engine = create_engine(self.db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        print(f"HybridRetriever 初始化完成 | Chroma: {self.persist_dir} | Embedding: {settings.embedding_model}")

    def _get_product_filter(self, product_name: Optional[str] = None) -> dict:
        """获取产品过滤条件（增强版：支持模糊匹配）"""
        if not product_name:
            print("未提供产品过滤条件，返回空过滤")
            return {}
        
        # 尝试完全匹配
        with self.SessionLocal() as session:
            # 1. 先尝试完全匹配
            result = session.execute(text("""
                SELECT metadata->>'product_name' as product_name
                FROM ingested_documents 
                WHERE metadata->>'product_name' = :product_name
                LIMIT 1
            """), {"product_name": product_name})
            row = result.fetchone()
            if row and row.product_name:
                print(f"找到完全匹配的产品: {row.product_name}")
                return {"product_name": {"$eq": row.product_name}}
            
            # 2. 尝试模糊匹配（忽略空格）
            product_name_no_space = product_name.replace(" ", "")
            result = session.execute(text("""
                SELECT metadata->>'product_name' as product_name
                FROM ingested_documents 
                WHERE REPLACE(metadata->>'product_name', ' ', '') = :product_name_no_space
                LIMIT 1
            """), {"product_name_no_space": product_name_no_space})
            row = result.fetchone()
            if row and row.product_name:
                print(f"找到模糊匹配的产品: {row.product_name}")
                return {"product_name": {"$eq": row.product_name}}
            
            # 3. 尝试部分匹配
            result = session.execute(text("""
                SELECT metadata->>'product_name' as product_name
                FROM ingested_documents 
                WHERE metadata->>'product_name' ILIKE :product_name_pattern
                LIMIT 1
            """), {"product_name_pattern": f"%{product_name}%"})
            row = result.fetchone()
            if row and row.product_name:
                print(f"找到部分匹配的产品: {row.product_name}")
                return {"product_name": {"$eq": row.product_name}}
        
        # 4. 无匹配时，返回空过滤（不过滤）
        print(f"未找到匹配的产品: {product_name}，返回空过滤")
        return {}

    def retrieve(self, query: str, top_k: int = 10, product_filter: Optional[str] = None) -> RetrievalResult:
        start = time.time()
        where = self._get_product_filter(product_filter)
        print(f"Hybrid Retrieval | query: {query[:60]}... | filter: {where}")

        # 调试：直接使用 Chroma 客户端查询，查看更多信息
        try:
            collection = self.client.get_collection(name=self.collection_name)
            print(f"集合文档总数: {collection.count()}")
            
            # 尝试不带过滤条件的查询
            if where:
                print("尝试不带过滤条件的查询...")
                all_docs = collection.get(limit=5)
                print(f"不带过滤条件的文档数量: {len(all_docs['documents'])}")
                if all_docs['documents']:
                    print(f"第一个文档: {all_docs['documents'][0][:50]}...")
                    print(f"第一个文档元数据: {all_docs['metadatas'][0]}")
        except Exception as e:
            print(f"调试查询失败: {e}")

        docs = self.vectorstore.similarity_search(
            query=query,
            k=top_k,
            filter=where if where else None
        )
        print(f"相似度搜索结果数量: {len(docs)}")

        chunks: List[RetrievedChunk] = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            chunk = RetrievedChunk(
                chunk_id=str(meta.get("id", i)),
                document_id=meta.get("document_id", ""),
                product_name=meta.get("product_name") or meta.get("product", ""),
                doc_type=meta.get("doc_type") or meta.get("type", "paper"),
                text=doc.page_content,
                metadata=meta,
                score=0.0,
                rank=i + 1
            )
            chunks.append(chunk)

        retrieval_time = int((time.time() - start) * 1000)
        result = RetrievalResult(
            query=query,
            chunks=chunks,
            total_retrieved=len(chunks),
            used_metadata_filter=bool(where),
            retrieval_time_ms=retrieval_time
        )
        print(f"召回完成 | {len(chunks)} 个 chunks | 耗时 {retrieval_time}ms")
        return result