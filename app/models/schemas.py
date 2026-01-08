from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks_created: int = 0
    message: str


class AddChunksRequest(BaseModel):
    text: str
    metadata: dict[str, Any] | None = None


class AddChunksResponse(BaseModel):
    status: str
    chunks_added: int
    chunk_ids: list[str]


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(
        default=4, description="Documents count for context", ge=1, le=10
    )
    temperature: float = Field(
        default=0.7, description="Generation temperature", ge=0.0, le=2.0
    )
    score_threshold: float = Field(
        default=0.0, description="Min threshold", ge=0.0, le=1.0
    )


class AskResponse(BaseModel):
    answer: str
    question: str
    sources: list[dict[str, Any]]
    context_used: bool
    model: str | None = None
    metrics: dict[str, Any]
    error: str | None = None
