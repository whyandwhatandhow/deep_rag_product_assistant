# backend/app/retriever/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from .schemas import RetrievedChunk, RetrievalResult


class BaseRetriever(ABC):
    """检索器基类"""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        product_filter: Optional[str] = None
    ) -> RetrievalResult:
        """检索方法"""
        pass