from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = ""
    google_model: str = "gemma-4-31b-it"
    demo_mode: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
