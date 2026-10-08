from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str
    google_model: str = "gemini-3.5-flash-lite"
    google_embedding_model: str = "models/gemini-embedding-001"
    vietqr_bank_id: str = ""
    vietqr_account_number: str = ""
    vietqr_account_name: str = ""
    vietqr_template: str = "compact2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
