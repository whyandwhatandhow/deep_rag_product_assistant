# backend/app/ingest/preprocessor.py
from typing import List
from langchain_core.documents import Document as LangchainDocument
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """文档预处理器 - 清洗 + 标题结构识别 + metadata 丰富（适配你当前的 loader）"""

    def __init__(self):
        # 常见噪声模式（可后续扩展）
        self.noise_patterns = [
            r"^\s*Page \d+\s*$",  # 页码单独一行
            r"^\s*©.*$",  # 版权声明
            r"^\s*All rights reserved.*$",
            r"^\s*\d+\s*$",  # 纯数字行
        ]

    def _clean_text(self, text: str) -> str:
        """基础清洗"""
        if not text:
            return ""

        # 去多余空白
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        # 去多余空格
        text = re.sub(r" +", " ", text)

        # 移除常见噪声
        for pattern in self.noise_patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)

        return text.strip()

    def _detect_section_title(self, text: str) -> tuple[str, str]:
        """简单标题识别（支持中英文产品文档常见格式）"""
        # 标题特征：短文本 + 以数字/大写/特殊符号开头
        if len(text) > 120:  # 太长不可能是标题
            return text, ""

        # 常见标题模式
        title_patterns = [
            r"^(第[一二三四五六七八九十]+[章节])",  # 中文章节
            r"^(Chapter|Section|Part)\s*\d+",  # 英文
            r"^([0-9]+\.)",  # 数字标题
            r"^([A-Z][A-Z\s&]+)$",  # 全大写标题
        ]

        for pattern in title_patterns:
            match = re.match(pattern, text.strip())
            if match:
                return "", text.strip()  # 返回 (content, section_title)

        return text, ""

    def preprocess(self, docs: List[LangchainDocument],
                   product_name: str = None,
                   doc_type: str = "manual") -> List[LangchainDocument]:
        """
        主处理入口
        product_name: 可在 ingest 时手动传入（如 "智能音箱Pro"）
        doc_type: manual / faq / policy / spec 等
        """
        processed = []

        for i, doc in enumerate(docs):
            original_text = doc.page_content
            cleaned = self._clean_text(original_text)

            if not cleaned:
                continue  # 过滤空内容

            # 标题识别
            content, section_title = self._detect_section_title(cleaned)

            # 丰富 metadata
            new_metadata = doc.metadata.copy()
            new_metadata.update({
                "product_name": product_name or doc.metadata.get("product_name"),
                "doc_type": doc_type,
                "section_title": section_title or doc.metadata.get("section_title"),
                "preprocessed": True,
                "chunk_index": i,
                "update_time": datetime.utcnow().isoformat(),
            })

            # 如果是标题行，content 可能为空，我们把标题也存成一个 chunk（便于 Hierarchical）
            final_content = content or section_title

            processed.append(LangchainDocument(
                page_content=final_content,
                metadata=new_metadata
            ))

        logger.info(f"✅ 预处理完成: {len(processed)} 个干净 chunk（原始 {len(docs)}）")
        return processed