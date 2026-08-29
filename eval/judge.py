from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from app.config import settings


class GroundingJudgment(BaseModel):
    supported: bool
    unsupported_claims: list[str] = Field(default_factory=list)
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
Return supported=true only if all material factual claims are supported. Identify unsupported claims explicitly."""
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
