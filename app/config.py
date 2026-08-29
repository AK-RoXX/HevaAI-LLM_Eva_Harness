from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM provider
    llm_provider: str = "ollama"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5-coder:14b"

    # Optional automatic fallback
    llm_fallback_enabled: bool = True
    llm_fallback_provider: str = "ollama"
    llm_fallback_model: str = "qwen2.5-coder:14b"

    # Generation
    llm_temperature: float = 0.0
    llm_seed: int = 42

    # Retrieval
    top_k: int = 5
    abstain_score_threshold: float = 0.08

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()