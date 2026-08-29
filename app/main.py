from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import settings
from .llm import LLMClient
from .models import QARequest, QAResponse
from .service import QAService
from .store import DocumentStore

BASE_DIR = Path(__file__).resolve().parent.parent
app = FastAPI(title="Heva Adversarial Q&A SUT", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
store = DocumentStore()
llm = LLMClient() if settings.gemini_api_key else None
service = QAService(store, llm) if llm else None


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": bool(settings.gemini_api_key),
        "model": settings.gemini_model,
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
        raise HTTPException(503, "GEMINI_API_KEY is not configured")
    return service.ask(request.question)
