import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.db.models.document import Document
from app.db.models.session import ResearchSession
from app.schemas.session import (
    SessionCreate,
    SessionDetailOut,
    SessionListResponse,
    SessionOut,
    SessionUpdate,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def _validate_document_ids(
    db: AsyncSession, doc_ids: list[uuid.UUID]
) -> list[str]:
    """Verify all scoped document UUIDs exist and return deduplicated string IDs."""
    if not doc_ids:
        return []

    unique_ids = list(dict.fromkeys(doc_ids))
    found_rows = await db.scalars(select(Document.id).where(Document.id.in_(unique_ids)))
    found_set = set(found_rows.all())
    missing = [str(doc_id) for doc_id in unique_ids if doc_id not in found_set]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Referenced document(s) not found: {', '.join(missing)}",
        )
    return [str(doc_id) for doc_id in unique_ids]


async def _get_session_or_404(
    db: AsyncSession, session_id: uuid.UUID, include_messages: bool = False
) -> ResearchSession:
    """Fetch a ResearchSession by ID or raise 404."""
    stmt = select(ResearchSession).where(ResearchSession.id == session_id)
    if include_messages:
        stmt = stmt.options(selectinload(ResearchSession.messages))
    session_obj = await db.scalar(stmt)
    if session_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research session not found.",
        )
    return session_obj


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreate,
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    """Create a new research session optionally scoped to specific documents."""
    validated_ids = await _validate_document_ids(db, payload.selected_document_ids)
    session_obj = ResearchSession(
        title=payload.title.strip(),
        selected_document_ids=validated_ids,
    )
    db.add(session_obj)
    await db.commit()
    await db.refresh(session_obj)
    return SessionOut.model_validate(session_obj)


@router.get("", response_model=SessionListResponse)
async def list_sessions(db: AsyncSession = Depends(get_db)) -> SessionListResponse:
    """List all research sessions sorted by most recently updated."""
    rows = await db.scalars(select(ResearchSession).order_by(ResearchSession.updated_at.desc()))
    sessions = [SessionOut.model_validate(item) for item in rows.all()]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> SessionDetailOut:
    """Return a research session along with its ordered message history."""
    session_obj = await _get_session_or_404(db, session_id, include_messages=True)
    return SessionDetailOut.model_validate(session_obj)


@router.patch("/{session_id}", response_model=SessionOut)
async def update_session(
    session_id: uuid.UUID,
    payload: SessionUpdate,
    db: AsyncSession = Depends(get_db),
) -> SessionOut:
    """Rename a research session or update its scoped document IDs."""
    session_obj = await _get_session_or_404(db, session_id)
    if payload.title is not None:
        session_obj.title = payload.title.strip()
    if payload.selected_document_ids is not None:
        session_obj.selected_document_ids = await _validate_document_ids(
            db, payload.selected_document_ids
        )
    await db.commit()
    await db.refresh(session_obj)
    return SessionOut.model_validate(session_obj)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a research session and all associated messages."""
    session_obj = await _get_session_or_404(db, session_id)
    await db.delete(session_obj)
    await db.commit()

