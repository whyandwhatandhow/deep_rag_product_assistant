# backend/app/ingest/indexer.py
from typing import List
from langchain_core.documents import Document as LangchainDocument
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import uuid
import logging
from datetime import datetime
import json   # ← 必须导入
from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentIndexer:
    """文档索引器 - Chroma + PostgreSQL（完整版，已修复 dict → JSONB 问题）"""

    def __init__(self):
        # ==================== Embedding ====================
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
        logger.info(f"✅ 使用本地免费 Embedding: {settings.embedding_model}")

        # ==================== Chroma 向量库 ====================
        import os
        current_dir = os.getcwd()
        logger.info(f"当前工作目录: {current_dir}")
        logger.info(f"Chroma 持久化目录（配置）: {settings.chroma_persist_directory}")
        
        # 计算绝对路径
        chroma_path = os.path.abspath(settings.chroma_persist_directory)
        logger.info(f"Chroma 持久化目录（绝对路径）: {chroma_path}")
        logger.info(f"Chroma 持久化目录是否存在: {os.path.exists(chroma_path)}")
        
        try:
            # 使用原生 Chroma 客户端
            self.client = chromadb.PersistentClient(path=chroma_path)
            logger.info("成功连接到 Chroma 数据库")
            
            # 检查是否存在旧的集合，如果存在则删除
            collections = self.client.list_collections()
            for col in collections:
                if col.name == "product_knowledge":
                    logger.info("删除旧的 product_knowledge 集合")
                    self.client.delete_collection(name="product_knowledge")
                    break
            
            # 创建新的集合
            self.collection = self.client.create_collection(
                name="product_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"成功创建集合: product_knowledge")
            logger.info(f"集合当前文档数量: {self.collection.count()}")
        except Exception as e:
            logger.error(f"初始化 Chroma 客户端失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        # ==================== PostgreSQL ====================
        self.engine = create_engine(
            f"postgresql+psycopg2://{settings.postgres_user}:{settings.postgres_password}@"
            f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        )
        self.Session = sessionmaker(bind=self.engine)
        logger.info("✅ PostgreSQL 连接已准备好")

    def _save_document_to_pg(self, document_id: str, filename: str, metadata: dict, chunk_count: int):
        """保存元数据到 PostgreSQL（关键修复：dict 转 JSON）"""
        try:
            with self.Session() as session:
                session.execute(
                    text("""
                        INSERT INTO ingested_documents 
                        (document_id, filename, metadata, chunk_count, ingested_at)
                        VALUES (:doc_id, :filename, :metadata, :chunk_count, :ingested_at)
                        ON CONFLICT (document_id) DO UPDATE 
                        SET metadata = EXCLUDED.metadata,
                            chunk_count = EXCLUDED.chunk_count,
                            ingested_at = EXCLUDED.ingested_at
                    """),
                    {
                        "doc_id": document_id,
                        "filename": filename,
                        "metadata": json.dumps(metadata),      # ← 关键修复
                        "chunk_count": chunk_count,
                        "ingested_at": datetime.utcnow()
                    }
                )
                session.commit()
            logger.info("✅ PostgreSQL 元数据写入成功")
        except Exception as e:
            logger.error(f"❌ PostgreSQL 写入失败: {e}", exc_info=True)
            raise

    def index_documents(self, chunks: List[LangchainDocument],
                       product_name: str = None,
                       doc_type: str = "manual") -> str:
        """主入口"""
        if not chunks:
            raise ValueError("没有可索引的 chunks")

        document_id = chunks[0].metadata.get("document_id") or str(uuid.uuid4())[:16]
        filename = chunks[0].metadata.get("filename", "unknown.pdf")

        # 清洗 metadata（防止 None 值）
        cleaned_chunks = []
        for chunk in chunks:
            meta = chunk.metadata.copy()
            for k, v in list(meta.items()):
                if v is None:
                    if k in ["section_title", "product_name"]:
                        meta[k] = ""
                    elif k in ["page_number", "chunk_index", "parent_chunk_id"]:
                        meta[k] = 0
                    else:
                        meta.pop(k, None)
            meta["document_id"] = document_id
            if product_name:
                meta["product_name"] = product_name
            meta["doc_type"] = doc_type
            cleaned_chunks.append(LangchainDocument(
                page_content=chunk.page_content,
                metadata=meta
            ))

        logger.info(f"正在向量化并存入 Chroma: {len(cleaned_chunks)} 个 chunks...")
        # 记录添加前的文档数量
        try:
            before_count = self.collection.count()
            logger.info(f"添加前的文档数量: {before_count}")
        except Exception as e:
            logger.error(f"获取添加前的文档数量失败: {e}")
        
        # 准备数据
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for i, chunk in enumerate(cleaned_chunks):
            chunk_id = f"{document_id}_{i}"
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append(chunk.metadata)
            # 生成嵌入
            embedding = self.embeddings.embed_query(chunk.page_content)
            embeddings.append(embedding)
        
        # 添加文档
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            logger.info(f"成功添加 {len(ids)} 个文档到 Chroma")
        except Exception as e:
            logger.error(f"添加文档到 Chroma 失败: {e}")
            raise
        
        # 记录添加后的文档数量
        try:
            after_count = self.collection.count()
            logger.info(f"添加后的文档数量: {after_count}")
            logger.info(f"成功添加的文档数量: {after_count - before_count}")
        except Exception as e:
            logger.error(f"获取添加后的文档数量失败: {e}")
        
        logger.info(f"✅ Chroma 索引完成: {len(cleaned_chunks)} 个向量")

        # 写入 PostgreSQL
        doc_metadata = {
            "product_name": product_name or filename,
            "doc_type": doc_type,
            "source": filename,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        self._save_document_to_pg(document_id, filename, doc_metadata, len(cleaned_chunks))

        logger.info(f"🎉 文档索引完成！document_id = {document_id} | chunks = {len(cleaned_chunks)}")
        return document_id