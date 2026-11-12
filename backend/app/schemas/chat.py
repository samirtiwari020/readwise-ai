import uuid

from pydantic import BaseModel, Field

from app.schemas.session import MessageOut


class ChatRequest(BaseModel):
    """User research query submitted to an existing research session."""

    query: str = Field(..., min_length=1, max_length=4000)
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Optional override of scoped document IDs for this query.",
    )


class ChatResponse(BaseModel):
    """Synchronous chat response containing persisted user and grounded assistant messages."""

    session_id: uuid.UUID
    user_message: MessageOut
    assistant_message: MessageOut

