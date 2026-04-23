from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Citation(BaseModel):
    document_id: str
    filename: str
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    content_snippet: str
    relevance_score: Optional[float] = None


class RetrievalMetadata(BaseModel):
    retrieval_time_ms: float
    total_chunks_retrieved: int
    chunks_after_rerank: int
    has_evidence: bool


class QueryInfo(BaseModel):
    original_question: str
    rewritten_query: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float = 0.0
    has_evidence: bool

    query_info: Optional[QueryInfo] = None
    retrieval_metadata: Optional[RetrievalMetadata] = None

    timestamp: str = None

    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        super().__init__(**data)