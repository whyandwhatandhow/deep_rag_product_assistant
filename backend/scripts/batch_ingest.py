# backend/scripts/batch_ingest.py
from pathlib import Path
import argparse
import logging
import sys
import os
from datetime import datetime
from typing import List, Tuple

# ====================== 自动定位项目根目录 ======================
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
sys.path.insert(0, str(project_root / "backend"))

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

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".md"}


def scan_files(directory: Path, recursive: bool = False) -> List[Path]:
    """扫描目录下的所有支持的文件"""
    files = []
    if recursive:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def ingest_single_file(file_path: Path, product_name: str, doc_type: str, indexer: DocumentIndexer) -> Tuple[bool, str]:
    """处理单个文件，返回 (是否成功, message)"""
    try:
        logger.info(f"开始处理: {file_path.name}")

        loader = DocumentLoader()
        raw_docs = loader.load_file(str(file_path))

        if not raw_docs:
            return False, "文档加载为空"

        preprocessor = DocumentPreprocessor()
        processed_docs = preprocessor.preprocess(raw_docs, product_name=product_name, doc_type=doc_type)

        chunker = DocumentChunker(chunk_size=1200, chunk_overlap=200)
        final_chunks = chunker.chunk(processed_docs)

        document_id = indexer.index_documents(final_chunks, product_name=product_name, doc_type=doc_type)

        return True, f"成功, document_id: {document_id}"
    except Exception as e:
        return False, f"失败: {str(e)}"


def batch_ingest(
    directory: str,
    product_name: str = None,
    doc_type: str = "manual",
    recursive: bool = False,
    skip_existing: bool = True
):
    """批量导入目录下所有支持的文件"""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"目录不存在: {dir_path}")
        sys.exit(1)

    logger.info(f"=" * 60)
    logger.info(f"批量导入开始")
    logger.info(f"目录: {dir_path}")
    logger.info(f"递归扫描: {recursive}")
    logger.info(f"产品名称: {product_name or '自动从文件名提取'}")
    logger.info(f"文档类型: {doc_type}")
    logger.info(f"=" * 60)

    files = scan_files(dir_path, recursive)

    if not files:
        logger.warning(f"在 {dir_path} 中未找到支持的文件类型: {SUPPORTED_EXTENSIONS}")
        return

    logger.info(f"找到 {len(files)} 个文件待处理\n")

    # 预创建索引器（只加载一次模型）
    try:
        indexer = DocumentIndexer()
        logger.info("✅ 索引器初始化完成")
    except Exception as e:
        logger.error(f"❌ 索引器初始化失败: {e}")
        return

    results = []
    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] 正在处理: {file_path.name}")

        success, message = ingest_single_file(file_path, product_name, doc_type, indexer)
        results.append((file_path.name, success, message))

        if success:
            success_count += 1
            logger.info(f"✅ {file_path.name}: {message}")
        else:
            fail_count += 1
            logger.error(f"❌ {file_path.name}: {message}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"批量导入完成!")
    logger.info(f"总计: {len(files)} | 成功: {success_count} | 失败: {fail_count}")
    logger.info(f"{'=' * 60}")

    if fail_count > 0:
        logger.warning("失败文件列表:")
        for name, success, msg in results:
            if not success:
                logger.warning(f"  - {name}: {msg}")

    report_path = project_root / f"batch_ingest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"批量导入报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"目录: {dir_path}\n")
        f.write(f"产品名称: {product_name or '自动从文件名提取'}\n")
        f.write(f"文档类型: {doc_type}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"总计: {len(files)} | 成功: {success_count} | 失败: {fail_count}\n\n")
        f.write("成功文件:\n")
        for name, success, msg in results:
            if success:
                f.write(f"  ✅ {name}: {msg}\n")
        if fail_count > 0:
            f.write("\n失败文件:\n")
            for name, success, msg in results:
                if not success:
                    f.write(f"  ❌ {name}: {msg}\n")

    logger.info(f"报告已保存: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深 RAG 产品知识助手 - 批量文档摄入工具")
    parser.add_argument("--dir", type=str, required=True, help="要导入的目录路径（相对项目根目录）")
    parser.add_argument("--product", type=str, default=None, help="统一的产品名称（不指定则从文件名提取）")
    parser.add_argument("--type", type=str, default="manual", help="文档类型 (manual/faq/policy/spec/paper)")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    parser.add_argument("--no-skip", action="store_true", help="不禁用跳过已存在文件（暂未实现）")

    args = parser.parse_args()

    full_dir = project_root / args.dir
    batch_ingest(
        directory=str(full_dir),
        product_name=args.product,
        doc_type=args.type,
        recursive=args.recursive
    )

    # & "D:/pyprocesses/deep_rag_product_assistant/backend/.venv/Scripts/python.exe" "d:/pyprocesses/deep_rag_product_assistant/backend/scripts/batch_ingest.py" --dir "backend/data/raw/Al-Zn-Mg" --product "Al-Zn-Mg 合金" --type "paper" --recursive

