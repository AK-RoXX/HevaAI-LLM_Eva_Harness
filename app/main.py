from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from .config import settings
from .llm import LLMClient
from .models import QARequest, QAResponse
from .service import QAService
from .store import DocumentStore

app = FastAPI(title="Heva Adversarial Q&A SUT", version="0.1.0")
store = DocumentStore()
llm = None

if settings.gemini_api_key:
    llm = LLMClient()

service = QAService(store, llm)


@app.get("/health")
def health():
    return {"status": "ok"}


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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if count == 0:
        raise HTTPException(status_code=400, detail="Document contains no extractable text")
    return {"document_id": document_id, "filename": file.filename, "chunks": count}


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id}


@app.post("/qa", response_model=QAResponse)
def qa(request: QARequest):
    if not settings.gemini_api_key:
        raise HTTPException(status_code=503, detail="LLM_API_KEY is not configured")
    return service.ask(request.question)


# @app.post("/qa", response_model=QAResponse)
# def qa(request: QARequest):
#     if llm is None:
#         raise HTTPException(
#             status_code=503,
#             detail="GEMINI_API_KEY is not configured"
#         )

#     return service.ask(request.question)
