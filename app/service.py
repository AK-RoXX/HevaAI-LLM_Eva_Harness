from .config import settings
from .llm import LLMClient
from .models import Citation, QAResponse, RetrievalTrace
from .store import DocumentStore


class QAService:
    def __init__(
        self,
        store: DocumentStore,
        llm: LLMClient | None,
    ):
        self.store = store
        self.llm = llm

    def ask(self, question: str) -> QAResponse:

        response, _ = self._ask(question)
        return response

    def ask_with_trace(self, question: str, llm_override: LLMClient | None = None) -> tuple[QAResponse, RetrievalTrace]:
        return self._ask(question, llm_override)

    def _ask(self, question: str, llm_override: LLMClient | None = None) -> tuple[QAResponse, RetrievalTrace]:

        active_llm = llm_override or self.llm

        if active_llm is None:
            raise RuntimeError(
                "No LLM provider is configured."
            )

        retrieved = self.store.search(
            question,
            settings.top_k,
        )

        if (
            not retrieved
            or retrieved[0][1]
            < settings.abstain_score_threshold
        ):
            response = QAResponse(
                answer=(
                    "The documents do not provide enough "
                    "information to answer this question."
                ),
                confidence=0.0,
                abstained=True,
                citations=[],
                model=active_llm.model_name,
            )
            trace = RetrievalTrace(
                retrieved_chunks=[
                    {
                        "chunk_id": chunk.chunk_id,
                        "rank": rank,
                        "score": max(0, min(1, score)),
                        "text": chunk.text,
                        "document_id": chunk.document_id,
                    }
                    for rank, (chunk, score) in enumerate(retrieved, 1)
                ],
                retrieval_threshold=settings.abstain_score_threshold,
                retrieval_abstained=True,
                abstention_reason=(
                    "no_retrieval_results" if not retrieved else "below_retrieval_threshold"
                ),
            )
            return response, trace

        evidence = [
            (c.chunk_id, c.text, s)
            for c, s in retrieved
        ]

        result = active_llm.answer(
            question,
            evidence,
        )

        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                relevance=max(
                    0,
                    min(1, s),
                ),
            )
            for c, s in retrieved
        ]

        response = QAResponse(
            answer=str(result["answer"]),
            confidence=max(
                0,
                min(
                    1,
                    float(result["confidence"]),
                ),
            ),
            abstained=bool(
                result["abstained"]
            ),
            citations=citations,
            model=active_llm.model_name,
        )
        trace = RetrievalTrace(
            retrieved_chunks=[
                {
                    "chunk_id": chunk.chunk_id,
                    "rank": rank,
                    "score": max(0, min(1, score)),
                    "text": chunk.text,
                    "document_id": chunk.document_id,
                }
                for rank, (chunk, score) in enumerate(retrieved, 1)
            ],
            retrieval_threshold=settings.abstain_score_threshold,
            retrieval_abstained=False,
        )
        return response, trace
