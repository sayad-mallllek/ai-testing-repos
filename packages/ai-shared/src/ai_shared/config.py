from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://ai_lab:ai_lab_dev@localhost:55432/ai_lab"
    sql_echo: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
