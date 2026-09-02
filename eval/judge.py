from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from app.config import settings


class GroundingJudgment(BaseModel):
    supported: bool
    completeness: str = "complete"
    contradiction: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    rationale: str


class GeminiGroundingJudge:
    def __init__(self):
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def judge(self, question, answer, citations):
        evidence = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in citations)
        prompt = f"""Evaluate whether every factual claim in the ANSWER is supported by the EVIDENCE.
The EVIDENCE is the only source of truth. Do not use outside knowledge.
QUESTION: {question}
ANSWER: {answer}
EVIDENCE:\n{evidence}
Check every material claim, including names, numbers, dates, negation, qualifiers, and relationships.
Set supported=true only if all claims are supported. Set completeness to complete, partial, or unsupported.
Set contradiction=true if the answer conflicts with the evidence. Identify unsupported claims and missing facts explicitly."""
        r = self.client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=GroundingJudgment,
            ),
        )
        return (r.parsed or GroundingJudgment.model_validate_json(r.text)).model_dump()
