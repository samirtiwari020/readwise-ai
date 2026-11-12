import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, computed_field


def _compute_progress_percent(status: str, pages_processed: int, page_count: int | None) -> float:
    """Compute bounded 0-100 ingestion progress percentage."""
    if status == "completed":
        return 100.0
    if status in {"pending", "failed"}:
        return 0.0
    if not page_count or page_count <= 0:
        return 0.0
    pct = (pages_processed / page_count) * 100.0
    return round(min(max(pct, 0.0), 99.9), 1)


class TocEntry(BaseModel):
    """Hierarchical table of contents entry extracted from a PDF."""

    level: int
    title: str
    page: int


class DocumentOut(BaseModel):
    """Summary representation of an uploaded document."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    filename: str
    file_hash: str
    file_size_bytes: int
    page_count: int | None = None
    pages_processed: int = 0
    chunk_count: int = 0
    ingestion_status: str
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def progress_percent(self) -> float:
        """Return bounded 0-100 ingestion progress percentage."""
        return _compute_progress_percent(self.ingestion_status, self.pages_processed, self.page_count)


class DocumentDetailOut(DocumentOut):
    """Detailed document representation including extracted TOC and metadata."""

    toc_json: list[dict] | None = None
    metadata_json: dict | None = None


class DocumentStatusOut(BaseModel):
    """Lightweight polling response for document ingestion progress."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ingestion_status: str
    page_count: int | None = None
    pages_processed: int = 0
    chunk_count: int = 0
    error_message: str | None = None

    @computed_field
    @property
    def progress_percent(self) -> float:
        """Return bounded 0-100 ingestion progress percentage."""
        return _compute_progress_percent(self.ingestion_status, self.pages_processed, self.page_count)


class DocumentUploadResponse(BaseModel):
    """Response returned after uploading a PDF (including deduplication status)."""

    document: DocumentOut
    deduplicated: bool = False
    message: str


class DocumentListResponse(BaseModel):
    """List wrapper for uploaded documents."""

    documents: list[DocumentOut]
    total: int
