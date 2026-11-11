import hashlib
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.db.models.document import Document, IngestionStatus
from app.schemas.document import (
    DocumentDetailOut,
    DocumentListResponse,
    DocumentOut,
    DocumentStatusOut,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming buffer for large 1,000+ page PDFs


def _sanitize_filename(raw_name: str | None) -> str:
    """Strip path segments and unsafe characters from an uploaded filename."""
    base = Path(raw_name or "document.pdf").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not cleaned:
        return "document.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _derive_title(filename: str) -> str:
    """Produce a readable default book/document title from a filename."""
    stem = Path(filename).stem
    return re.sub(r"[_-]+", " ", stem).strip() or "Untitled Document"


async def _stream_upload_to_disk(
    upload: UploadFile, dest_dir: Path, doc_id: uuid.UUID, safe_filename: str
) -> tuple[Path, str, int]:
    """Stream uploaded PDF to disk while verifying PDF header and computing SHA-256."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    temp_path = dest_dir / f".tmp_{doc_id}_{safe_filename}"
    final_path = dest_dir / f"{doc_id}_{safe_filename}"

    hasher = hashlib.sha256()
    total_bytes = 0
    header_checked = False

    try:
        with temp_path.open("wb") as out_file:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                if not header_checked:
                    header_checked = True
                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid file format: expected a valid PDF document.",
                        )
                out_file.write(chunk)
                hasher.update(chunk)
                total_bytes += len(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file is empty.",
            )

        temp_path.replace(final_path)
        return final_path, hasher.hexdigest(), total_bytes
    except Exception:
        temp_path.unlink(missing_ok=True)
        final_path.unlink(missing_ok=True)
        raise


async def _get_document_or_404(db: AsyncSession, document_id: uuid.UUID) -> Document:
    """Fetch a Document row or raise 404."""
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """Upload a PDF document, deduplicate via SHA-256, and queue it for ingestion."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf files are supported.",
        )

    settings = get_settings()
    safe_name = _sanitize_filename(file.filename)
    doc_id = uuid.uuid4()

    saved_path, file_hash, total_bytes = await _stream_upload_to_disk(
        upload=file,
        dest_dir=settings.resolved_upload_dir,
        doc_id=doc_id,
        safe_filename=safe_name,
    )

    existing = await db.scalar(select(Document).where(Document.file_hash == file_hash))
    if existing is not None:
        saved_path.unlink(missing_ok=True)
        return DocumentUploadResponse(
            document=DocumentOut.model_validate(existing),
            deduplicated=True,
            message="Identical PDF already exists in library (deduplicated by SHA-256).",
        )

    document = Document(
        id=doc_id,
        title=_derive_title(safe_name),
        filename=safe_name,
        file_path=str(saved_path),
        file_hash=file_hash,
        file_size_bytes=total_bytes,
        ingestion_status=IngestionStatus.PENDING.value,
    )

    try:
        db.add(document)
        await db.commit()
        await db.refresh(document)
    except Exception:
        await db.rollback()
        saved_path.unlink(missing_ok=True)
        raise

    return DocumentUploadResponse(
        document=DocumentOut.model_validate(document),
        deduplicated=False,
        message="PDF uploaded successfully and queued for ingestion.",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(db: AsyncSession = Depends(get_db)) -> DocumentListResponse:
    """Return all uploaded documents sorted newest first."""
    result = await db.scalars(select(Document).order_by(Document.created_at.desc()))
    docs = [DocumentOut.model_validate(doc) for doc in result.all()]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{document_id}", response_model=DocumentDetailOut)
async def get_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> DocumentDetailOut:
    """Return detailed metadata and extracted table of contents for a document."""
    doc = await _get_document_or_404(db, document_id)
    return DocumentDetailOut.model_validate(doc)


@router.get("/{document_id}/status", response_model=DocumentStatusOut)
async def get_document_status(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> DocumentStatusOut:
    """Return live ingestion status and page progress for a document."""
    doc = await _get_document_or_404(db, document_id)
    return DocumentStatusOut.model_validate(doc)


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> FileResponse:
    """Stream the raw PDF file inline for split-pane browser rendering."""
    doc = await _get_document_or_404(db, document_id)
    file_path = Path(doc.file_path)
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored PDF file is missing from disk.",
        )
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=doc.filename,
        content_disposition_type="inline",
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> None:
    """Delete a document record, its relational chunks, and its stored PDF file."""
    doc = await _get_document_or_404(db, document_id)
    file_path = Path(doc.file_path)
    await db.delete(doc)
    await db.commit()
    file_path.unlink(missing_ok=True)

