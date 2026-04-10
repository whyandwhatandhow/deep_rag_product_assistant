# backend/app/ingest/chunker.py
from typing import List
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class DocumentChunker:
    """智能分块器 - Hierarchical + 固定大小 + 重叠（适配产品知识文档）"""

    def __init__(
            self,
            chunk_size: int = 600,  # tokens，大约 400-500 汉字，适合大多数 LLM context
            chunk_overlap: int = 100,  # 重叠量，保证上下文连贯
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 使用 LangChain 官方推荐的 RecursiveCharacterTextSplitter（对中英文都友好）
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],  # 中英文分隔符优先
            keep_separator=True,
        )

    def chunk(self, docs: List[LangchainDocument]) -> List[LangchainDocument]:
        """
        主入口：对预处理后的文档列表进行分块
        返回的每个 chunk 都会继承并丰富原始 metadata
        """
        all_chunks = []

        for doc in docs:
            # 如果是标题行（section_title 不为空），我们优先单独作为一个小 chunk
            if doc.metadata.get("section_title"):
                title_chunk = LangchainDocument(
                    page_content=doc.metadata["section_title"],
                    metadata=doc.metadata.copy()
                )
                all_chunks.append(title_chunk)

            # 对正文内容进行分块
            if doc.page_content:
                split_docs = self.splitter.split_documents([doc])

                for i, split_doc in enumerate(split_docs):
                    # 继承并丰富 metadata
                    new_metadata = split_doc.metadata.copy()
                    new_metadata.update({
                        "chunk_index": len(all_chunks),  # 全局序号
                        "parent_chunk_id": doc.metadata.get("chunk_index"),  # 层级追溯
                        "is_title": False,
                    })
                    all_chunks.append(LangchainDocument(
                        page_content=split_doc.page_content,
                        metadata=new_metadata
                    ))

        logger.info(f"✅ 分块完成: {len(all_chunks)} 个最终 chunk（原始 {len(docs)}）")
        return all_chunks