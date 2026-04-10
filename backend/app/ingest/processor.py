# backend/app/ingest/processor.py
from typing import List, Dict
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from app.schemas.rag import DocumentMetadata, Chunk


class DocumentProcessor:
    """文档预处理和切块处理器"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
        )

    def load_document(self, file_path: Path) -> List[Dict]:
        """根据文件类型加载文档"""
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif suffix in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")

        return loader.load()

    def process_document(self, file_path: Path, product_name: str = "默认产品") -> List[Chunk]:
        """完整处理一个文档：加载 → 切块 → 转为 Chunk"""
        docs = self.load_document(file_path)

        chunks: List[Chunk] = []
        doc_id = str(uuid.uuid4())

        for i, doc in enumerate(docs):
            # 切块
            split_texts = self.text_splitter.split_text(doc.page_content)

            for j, text in enumerate(split_texts):
                metadata = DocumentMetadata(
                    doc_id=doc_id,
                    title=file_path.stem,
                    product_name=product_name,
                    doc_type=self._guess_doc_type(file_path),
                    page_num=doc.metadata.get("page", i + 1),
                    update_time=Path(file_path).stat().st_mtime.__str__(),
                    source_file=file_path.name,
                )

                chunk = Chunk(
                    chunk_id=f"{doc_id}_{j}",
                    doc_id=doc_id,
                    content=text.strip(),
                    metadata=metadata,
                )
                chunks.append(chunk)

        return chunks

    def _guess_doc_type(self, file_path: Path) -> str:
        """简单推断文档类型"""
        name = file_path.name.lower()
        if "手册" in name or "manual" in name:
            return "manual"
        elif "faq" in name or "问答" in name:
            return "faq"
        elif "政策" in name or "售后" in name:
            return "policy"
        return "other"


# 单例实例（后续可注入）
document_processor = DocumentProcessor()