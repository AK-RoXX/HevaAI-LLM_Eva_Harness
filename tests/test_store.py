from app.store import DocumentStore


def test_chunking_is_deterministic():
    text = "A " * 1000
    assert DocumentStore.chunk_text(text) == DocumentStore.chunk_text(text)


def test_search():
    s = DocumentStore()
    s.add_document(
        "doc1", "x.txt", "Acme was founded in 2018. Revenue was $42 million in 2024."
    )
    results = s.search("when was Acme founded?", 1)
    assert results and "2018" in results[0][0].text
