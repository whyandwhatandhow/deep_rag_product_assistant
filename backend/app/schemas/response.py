from pydantic import BaseModel
from typing import List, Optional

class Citation(BaseModel):
    document_id: str
    filename: str
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    content_snippet: str

class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float = 0.0
    has_evidence: bool