from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    parameters: dict[str, Any] = Field(default_factory=dict)


class RunRead(RunCreate):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    output: dict[str, Any]
    created_at: datetime
