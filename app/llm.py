from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .config import settings


SYSTEM_PROMPT = """You are a grounded document question-answering system.

Rules:
1. Answer ONLY using the supplied evidence.
2. Never invent facts that are absent from the evidence.
3. If the evidence is insufficient, set abstained=true.
4. If the question contains a false premise, correct it using the evidence.
5. Keep the answer concise and factual.
6. confidence must represent your estimated probability that the answer is fully
   supported by the supplied evidence.
7. Do not use outside knowledge.
"""


class LLMAnswer(BaseModel):
    answer: str = Field(
        description="Concise answer based only on the supplied evidence."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability from 0 to 1 that the answer is fully supported."
    )
    abstained: bool = Field(
        description="True when the evidence is insufficient to answer."
    )


class LLMClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def answer(
        self,
        question: str,
        evidence: list[tuple[str, str, float]],
    ) -> dict:

        evidence_text = "\n\n".join(
            f"[{chunk_id}] relevance={score:.4f}\n{text}"
            for chunk_id, text, score in evidence
        )

        prompt = f"""Question:
{question}

Evidence:
{evidence_text}
"""

        response = self.client.models.generate_content(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=settings.llm_temperature,
                seed=settings.llm_seed,
                response_mime_type="application/json",
                response_schema=LLMAnswer,
            ),
        )

        if response.parsed is not None:
            result = response.parsed
        else:
            result = LLMAnswer.model_validate_json(response.text)

        return result.model_dump()