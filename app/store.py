from dataclasses import dataclass
import re
from pathlib import Path
from threading import Lock
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


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
        self._embedding_model = None
        self._embedding_matrix = None
        if SentenceTransformer is not None:
            try:
                self._embedding_model = SentenceTransformer(settings.embedding_model)
            except Exception:
                self._embedding_model = None

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
        if self._embedding_model is not None:
            self._embedding_matrix = self._embedding_model.encode(
                [c.text for c in self._chunks], normalize_embeddings=True
            )
        else:
            self._embedding_matrix = None

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        results = self.search_with_scores(query, top_k)
        return [(chunk, score) for chunk, score, _, _ in results]

    def search_with_scores(
        self, query: str, top_k: int = 5
    ) -> list[tuple[Chunk, float, float, float]]:
        if not self._chunks or self._vectorizer is None:
            return []
        q = self._vectorizer.transform([query])
        lexical_scores = cosine_similarity(q, self._matrix).ravel()
        candidate_k = min(len(self._chunks), max(top_k, settings.retrieval_candidate_k))
        candidate_indices = set(lexical_scores.argsort()[::-1][:candidate_k])
        semantic_scores = lexical_scores.copy()
        if self._embedding_model is not None and self._embedding_matrix is not None:
            semantic_query = self._embedding_model.encode([query], normalize_embeddings=True)
            semantic_scores = (semantic_query @ self._embedding_matrix.T).ravel()
            candidate_indices.update(semantic_scores.argsort()[::-1][:candidate_k])
        combined = (
            settings.lexical_weight * lexical_scores
            + settings.semantic_weight * semantic_scores
        )
        indices = sorted(candidate_indices, key=lambda i: combined[i], reverse=True)[:top_k]
        return [
            (
                self._chunks[i],
                float(max(0, combined[i])),
                float(max(0, lexical_scores[i])),
                float(max(0, semantic_scores[i])),
            )
            for i in indices
            if combined[i] > 0
        ]
