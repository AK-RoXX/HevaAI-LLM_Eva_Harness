from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .llm import LLMClient
from .models import EvalQARequest, QARequest, QAResponse
from .service import QAService
from .store import DocumentStore

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="Heva Adversarial Q&A SUT", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
store = DocumentStore()
try:
    llm = LLMClient()
except Exception as exc:
    print(f"LLM initialization failed: {exc}")
    llm = None

service = QAService(store, llm) if llm else None
evaluation_clients = {}

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": llm is not None,
        "provider": llm.provider if llm else None,
        "model": llm.model if llm else None,
    }

@app.get("/documents")
def documents():
    return store.list_documents()


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    data = await file.read()
    document_id = uuid4().hex[:12]
    try:
        text = store.extract(file.filename or "document.txt", data)
        count = store.add_document(document_id, file.filename or "document.txt", text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if count == 0:
        raise HTTPException(400, "Document contains no extractable text")
    return {"document_id": document_id, "filename": file.filename, "chunks": count}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not store.delete_document(document_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": document_id}


@app.post("/qa", response_model=QAResponse)
def qa(request: QARequest):
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="No LLM provider is configured or available.",
        )

    try:
        return service.ask(request.question)

    except Exception as exc:
        import traceback

        print("\n" + "=" * 70)
        print("LLM ERROR")
        print("=" * 70)
        print(f"Provider: {llm.provider if llm else 'unknown'}")
        print(f"Model: {llm.model if llm else 'unknown'}")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        traceback.print_exc()
        print("=" * 70 + "\n")

        error_message = str(exc).lower()

        # Gemini quota/rate-limit detection
        if any(
            x in error_message
            for x in [
                "quota",
                "resource exhausted",
                "rate limit",
                "429",
                "too many requests",
            ]
        ):
            detail = (
                f"LLM quota exhausted or rate limit reached. "
                f"Provider={llm.provider if llm else 'unknown'}, "
                f"Model={llm.model if llm else 'unknown'}. "
                f"Original error: {exc}"
            )
        else:
            detail = (
                f"LLM request failed. "
                f"Provider={llm.provider if llm else 'unknown'}, "
                f"Model={llm.model if llm else 'unknown'}. "
                f"Error: {exc}"
            )

        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from exc


@app.post("/qa/eval")
def qa_eval(request: EvalQARequest):
    """Evaluation-only endpoint; /qa remains backward compatible."""
    if service is None:
        raise HTTPException(status_code=503, detail="No LLM provider is configured or available.")
    try:
        override = None
        if request.provider or request.model:
            key = (request.provider or settings.llm_provider, request.model or "")
            override = evaluation_clients.get(key)
            if override is None:
                override = LLMClient(provider=key[0], model=request.model)
                evaluation_clients[key] = override
        response, trace = service.ask_with_trace(request.question, override)
        return {"response": response.model_dump(), "retrieval_trace": trace.model_dump()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"LLM request failed: {exc}") from exc
