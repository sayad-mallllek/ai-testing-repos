from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RunRead(RunCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    output: dict[str, Any]
    created_at: datetime


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SessionCreate(InputModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime


class ChatRequest(InputModel):
    message: str = Field(min_length=1, max_length=8000)


class TurnRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    position: int
    prompt: str
    answer: str
    messages: list[dict[str, Any]]
    tool_events: list[dict[str, Any]]
    usage: dict[str, int]
    created_at: datetime


class DocumentCreate(InputModel):
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=40_000)
    source: str = Field(default="manual", min_length=1, max_length=300)

    @field_validator("content")
    @classmethod
    def limit_words(cls, value: str) -> str:
        if len(value.split()) > 5000:
            raise ValueError("Documents are limited to 5000 words in this learning app.")
        return value


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    slug: str
    title: str
    content: str
    source: str


class KnowledgeQuery(InputModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=3, ge=1, le=5)
