import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CitationItem(BaseModel):
    """Structured source attribution referencing an exact document page and excerpt."""

    citation_index: int
    document_id: uuid.UUID
    document_title: str
    page_start: int
    page_end: int
    chapter: str | None = None
    section: str | None = None
    chunk_id: str
    quote_snippet: str
    score: float | None = None


class AgentTrace(BaseModel):
    """Execution telemetry emitted by the LangGraph orchestration pipeline."""

    query_type: str | None = None
    sub_queries: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    retrieved_chunks: int = 0
    reranked_chunks: int = 0
    verified_citations: int = 0
    abstained: bool = False
    latency_ms: int | None = None


class MessageOut(BaseModel):
    """Single message within a research session."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    session_id: uuid.UUID
    role: str
    content: str
    citations: list[dict] | None = Field(default=None, validation_alias="citations_json")
    agent_trace: dict | None = Field(default=None, validation_alias="agent_trace_json")
    created_at: datetime


class SessionCreate(BaseModel):
    """Payload to create a new research session."""

    title: str = Field(default="New Research Session", min_length=1, max_length=256)
    selected_document_ids: list[uuid.UUID] = Field(default_factory=list)


class SessionUpdate(BaseModel):
    """Payload to rename a session or update its document scope."""

    title: str | None = Field(default=None, min_length=1, max_length=256)
    selected_document_ids: list[uuid.UUID] | None = None


class SessionOut(BaseModel):
    """Research session summary."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    selected_document_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class SessionDetailOut(SessionOut):
    """Research session with ordered conversation history."""

    messages: list[MessageOut] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    """List wrapper for research sessions."""

    sessions: list[SessionOut]
    total: int

