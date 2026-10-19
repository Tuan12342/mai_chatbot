from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = "google"
    llm_timeout_seconds: float = 60
    llm_max_retries: int = 1
    google_api_key: str = ""
    google_model: str = "gemini-3.5-flash-lite"
    google_embedding_model: str = "models/gemini-embedding-2"
    google_vlm_model: str = "gemini-3.5-flash-lite"
    skin_recommendation_max_products: int = 4
    skin_image_max_bytes: int = 8 * 1024 * 1024
    skin_vision_min_confidence: float = 0.60
    skin_image_relevance_min_confidence: float = 0.85
    skin_red_flag_min_confidence: float = 0.85
    misa_api_key: str = ""
    misa_base_url: str = "https://ai.misa.vn/nlp/llm-gateway/v1"
    misa_model: str = "misa-ai-1.1-plus"
    vietqr_bank_id: str = ""
    vietqr_account_number: str = ""
    vietqr_account_name: str = ""
    vietqr_template: str = "compact2"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_tracing_environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
