import pytest

import app.llm as llm_module
from eval.runner import result_path, select_cases


class _Response:
    def raise_for_status(self):
        return None


def test_ollama_provider_initialization_and_metadata(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(llm_module.settings, "ollama_model", "test-ollama")
    provider = llm_module.OllamaProvider()
    assert provider.provider_name == "ollama"
    assert provider.model == "test-ollama"
    assert provider.model_name == "ollama/test-ollama"


def test_gemini_provider_initialization_and_default(monkeypatch):
    class FakeClient:
        pass

    monkeypatch.setattr(llm_module.settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(llm_module.genai, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(llm_module.settings, "gemini_model", "test-gemini")
    provider = llm_module.GeminiProvider()
    assert provider.provider_name == "gemini"
    assert provider.model == "test-gemini"


def test_missing_gemini_key_is_actionable(monkeypatch):
    monkeypatch.setattr(llm_module.settings, "gemini_api_key", "")
    with pytest.raises(ValueError, match="Gemini evaluation requires GEMINI_API_KEY to be configured"):
        llm_module.GeminiProvider()


def test_provider_selection_and_model_override(monkeypatch):
    monkeypatch.setattr(llm_module.httpx, "get", lambda *args, **kwargs: _Response())
    client = llm_module.LLMClient(provider="ollama", model="explicit-model")
    assert client.provider == "ollama"
    assert client.model == "explicit-model"


def test_cases_precede_limit_and_invalid_ids_are_reported():
    rows = [{"id": "GT001"}, {"id": "GT002"}, {"id": "GT003"}]
    assert [r["id"] for r in select_cases(rows, ["GT003"], 0, 1)] == ["GT003"]
    with pytest.raises(ValueError, match="GT999"):
        select_cases(rows, ["GT999"])


def test_provider_result_filenames_are_separate():
    ollama = result_path("ground_truth.jsonl", "ollama", "qwen2.5-coder:14b")
    gemini = result_path("ground_truth.jsonl", "gemini", "gemini-3.6-flash")
    assert ollama != gemini
    assert ollama.name == "ground_truth_ollama_qwen2_5_coder_14b.jsonl"
