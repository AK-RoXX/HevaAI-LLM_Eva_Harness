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


class RetrievalTraceItem(BaseModel):
    chunk_id: str
    rank: int
    score: float = Field(ge=0, le=1)
    text: str
    document_id: str


class RetrievalTrace(BaseModel):
    retrieved_chunks: list[RetrievalTraceItem]
    retrieval_threshold: float
    retrieval_abstained: bool
    abstention_reason: str | None = None


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class EvalQARequest(QARequest):
    """Evaluation-only provider override; the public /qa contract is unchanged."""

    provider: str | None = None
    model: str | None = None
