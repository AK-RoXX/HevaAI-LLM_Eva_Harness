from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    relevance: float = Field(ge=0, le=1)


class QAResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)
    abstained: bool
    citations: list[Citation]
    model: str


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
