from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    top_k: int = 5
    abstain_score_threshold: float = 0.08
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
