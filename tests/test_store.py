from app.store import DocumentStore


def test_chunking_is_deterministic():
    text = "The quick brown fox jumps over the lazy dog. " * 100
    a = DocumentStore.chunk_text(text)
    b = DocumentStore.chunk_text(text)
    assert a == b


def test_search_returns_relevant_chunk():
    store = DocumentStore()
    store.add_document("doc1", "a.txt", "Paris is the capital of France. Berlin is the capital of Germany.")
    results = store.search("capital of France", top_k=2)
    assert results
    assert results[0][0].document_id == "doc1"
