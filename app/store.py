from dataclasses import dataclass
import re
from pathlib import Path
from threading import Lock
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str


class DocumentStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._chunks: list[Chunk] = []
        self._documents: dict[str, str] = {}
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    @staticmethod
    def extract(filename: str, data: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".txt", ".md"}:
            return data.decode("utf-8", errors="replace")
        if suffix == ".pdf":
            with fitz.open(stream=data, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)
        raise ValueError("Only .txt, .md, and .pdf files are supported")

    @staticmethod
    def chunk_text(text: str, size: int = 650, overlap: int = 100) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks, start = [], 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + size // 2:
                    end = boundary
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def add_document(self, document_id: str, filename: str, text: str) -> int:
        parts = self.chunk_text(text)
        with self._lock:
            self._documents[document_id] = filename
            self._chunks = [c for c in self._chunks if c.document_id != document_id]
            self._chunks.extend(
                Chunk(f"{document_id}:c{i:04d}", document_id, p)
                for i, p in enumerate(parts)
            )
            self._rebuild()
        return len(parts)

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            existed = document_id in self._documents
            self._documents.pop(document_id, None)
            self._chunks = [c for c in self._chunks if c.document_id != document_id]
            self._rebuild()
            return existed

    def list_documents(self) -> list[dict]:
        return [
            {"document_id": i, "filename": f}
            for i, f in sorted(self._documents.items())
        ]

    def _rebuild(self) -> None:
        if not self._chunks:
            self._vectorizer = self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), sublinear_tf=True
        )
        self._matrix = self._vectorizer.fit_transform([c.text for c in self._chunks])

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self._chunks or self._vectorizer is None:
            return []
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in indices if scores[i] > 0]
