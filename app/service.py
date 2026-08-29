from .config import settings
from .llm import LLMClient
from .models import Citation, QAResponse
from .store import DocumentStore


class QAService:
    def __init__(self, store: DocumentStore, llm: LLMClient | None):
        self.store, self.llm = store, llm

    def ask(self, question: str) -> QAResponse:
        retrieved = self.store.search(question, settings.top_k)
        if not retrieved or retrieved[0][1] < settings.abstain_score_threshold:
            return QAResponse(
                answer="The documents do not provide enough information to answer this question.",
                confidence=0.0,
                abstained=True,
                citations=[],
                model=settings.gemini_model,
            )
        evidence = [(c.chunk_id, c.text, s) for c, s in retrieved]
        result = self.llm.answer(question, evidence)
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                relevance=max(0, min(1, s)),
            )
            for c, s in retrieved
        ]
        return QAResponse(
            answer=str(result["answer"]),
            confidence=max(0, min(1, float(result["confidence"]))),
            abstained=bool(result["abstained"]),
            citations=citations,
            model=settings.gemini_model,
        )
