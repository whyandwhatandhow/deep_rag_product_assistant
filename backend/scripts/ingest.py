# backend/scripts/ingest.py
from pathlib import Path
import argparse
import logging
import sys
import os

# ====================== 自动定位项目根目录 ======================
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]   # scripts/ → backend/ → root
sys.path.insert(0, str(project_root / "backend"))

# 设置工作目录为项目根目录（让 Chroma persist_directory 更稳定）
os.chdir(project_root)

from app.ingest.loader import DocumentLoader
from app.ingest.preprocessor import DocumentPreprocessor
from app.ingest.chunker import DocumentChunker
from app.ingest.indexer import DocumentIndexer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ingest_file(file_path: str, product_name: str = None, doc_type: str = "manual"):
    """单文件完整 ingest 流程"""
    logger.info(f"开始处理文件: {file_path}")

    # 1. 加载
    loader = DocumentLoader()
    raw_docs = loader.load_file(file_path)

    # 2. 预处理
    preprocessor = DocumentPreprocessor()
    processed_docs = preprocessor.preprocess(raw_docs, product_name=product_name, doc_type=doc_type)

    # 3. 分块
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=100)
    final_chunks = chunker.chunk(processed_docs)

    # 4. 索引
    indexer = DocumentIndexer()
    document_id = indexer.index_documents(final_chunks, product_name=product_name, doc_type=doc_type)

    logger.info(f"🎉 文件 ingest 完成！document_id: {document_id}")
    return document_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深 RAG 产品知识助手 - 文档摄入工具")
    parser.add_argument("--file", type=str, required=True, help="要摄入的文件路径（相对项目根目录）")
    parser.add_argument("--product", type=str, default=None, help="产品名称")
    parser.add_argument("--type", type=str, default="manual", help="文档类型 (manual/faq/policy/spec/paper)")

    args = parser.parse_args()

    # 支持相对路径（data/raw/xxx.pdf）
    full_path = project_root / args.file
    if not full_path.exists():
        logger.error(f"文件不存在: {full_path}")
        sys.exit(1)

    ingest_file(str(full_path), product_name=args.product, doc_type=args.type)

    #D:\pyprocesses\deep_rag_product_assistant\backend\.venv\Scripts\python.exe backend\scripts\ingest.py --file "backend/data/raw/ResNet.pdf" --product "ResNet" --type "paper"