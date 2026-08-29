from .config import settings
from .llm import LLMClient
from .models import Citation, QAResponse
from .store import DocumentStore


class QAService:
    def __init__(self, store: DocumentStore, llm: LLMClient) -> None:
        self.store = store
        self.llm = llm

    def ask(self, question: str) -> QAResponse:
        retrieved = self.store.search(question, settings.top_k)
        if not retrieved or retrieved[0][1] < settings.abstain_score_threshold:
            return QAResponse(
                answer="The documents do not provide enough information to answer this question.",
                confidence=0.0,
                abstained=True,
                citations=[],
                model=settings.llm_model,
            )

        evidence = [(c.chunk_id, c.text, score) for c, score in retrieved]
        result = self.llm.answer(question, evidence)
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                relevance=max(0.0, min(1.0, score)),
            )
            for c, score in retrieved
        ]
        return QAResponse(
            answer=str(result.get("answer", "")),
            confidence=max(0.0, min(1.0, float(result.get("confidence", 0.0)))),
            abstained=bool(result.get("abstained", False)),
            citations=citations,
            model=settings.llm_model,
        )
