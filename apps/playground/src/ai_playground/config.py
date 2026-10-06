from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    agent_max_tool_rounds: int = Field(default=6, ge=1, le=12)
    agent_max_tool_calls: int = Field(default=12, ge=1, le=24)
    agent_history_turns: int = Field(default=8, ge=0, le=30)
    agent_history_chars: int = Field(default=60_000, ge=0, le=200_000)
    agent_max_output_tokens: int = Field(default=1200, ge=100, le=4000)
    agent_timeout_seconds: float = Field(default=120, gt=0, le=300)
    model_timeout_seconds: float = Field(default=30, gt=0, le=120)
    search_timeout_seconds: float = Field(default=10, gt=0, le=30)


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()
