import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.models.document import Document, DocumentChunk, IngestionStatus
from app.services.ingestion.adaptive_chunker import AdaptiveChunker, ChunkRecord
from app.services.ingestion.pdf_parser import ParsedPdfMetadata, iter_parsed_pages, profile_pdf
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64
DB_CHUNK_BATCH_SIZE = 500


def _parse_and_chunk_pdf(
    pdf_path: Path,
    document_id: str,
    title: str,
) -> tuple[ParsedPdfMetadata, list[ChunkRecord], list[int]]:
    """Extract PDF profile, stream pages, and produce adaptive chunks synchronously."""
    settings = get_settings()
    profile = profile_pdf(pdf_path)
    empty_pages: list[int] = []

    def _tracked_pages():
        for page in iter_parsed_pages(pdf_path, profile):
            if not page.has_text:
                empty_pages.append(page.page_number)
            yield page

    chunker = AdaptiveChunker(
        document_id=document_id,
        document_title=title,
        toc=profile.toc,
        min_tokens=settings.chunk_min_tokens,
        target_tokens=settings.chunk_target_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    chunks = chunker.chunk_pages(_tracked_pages())
    return profile, chunks, empty_pages


async def _mark_failed(
    session_factory: async_sessionmaker[AsyncSession],
    document_id: uuid.UUID,
    error_message: str,
) -> None:
    """Persist FAILED status safely in a clean database session."""
    async with session_factory() as db:
        doc = await db.get(Document, document_id)
        if doc is None:
            return
        doc.ingestion_status = IngestionStatus.FAILED.value
        doc.error_message = error_message[:2000]
        await db.commit()


async def ingest_document(
    document_id: uuid.UUID,
    session_factory: async_sessionmaker[AsyncSession],
    embedding_service: EmbeddingService,
    vector_store: ChromaVectorStore,
) -> None:
    """Run full page-aware PDF ingestion, adaptive chunking, embedding, and indexing."""
    doc_id_str = str(document_id)

    async with session_factory() as db:
        doc = await db.get(Document, document_id)
        if doc is None:
            logger.warning("Ingestion skipped: document %s no longer exists.", doc_id_str)
            return

        pdf_path = Path(doc.file_path)
        title = doc.title
        doc.ingestion_status = IngestionStatus.PROCESSING.value
        doc.pages_processed = 0
        doc.error_message = None
        await db.commit()

    try:
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Uploaded PDF missing from disk: {pdf_path}")

        profile, chunks, empty_pages = await asyncio.to_thread(
            _parse_and_chunk_pdf, pdf_path, doc_id_str, title
        )

        if not chunks:
            raise ValueError(
                "No extractable text found in PDF (it may be scanned images without OCR)."
            )

        # Persist PDF profile & clear any prior chunks/vectors for idempotent re-ingestion
        async with session_factory() as db:
            doc = await db.get(Document, document_id)
            if doc is None:
                return
            doc.page_count = profile.page_count
            doc.toc_json = profile.toc
            doc.metadata_json = {
                "pdf_title": profile.pdf_title,
                "author": profile.author,
                "subject": profile.subject,
                "body_font_size": round(profile.body_font_size, 2),
                "empty_pages_count": len(empty_pages),
                "empty_pages_sample": empty_pages[:20],
            }
            doc.pages_processed = min(profile.page_count, max(1, int(profile.page_count * 0.2)))
            await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            await db.commit()

        await asyncio.to_thread(vector_store.delete_document_chunks, doc_id_str)

        # Embed & index batches while updating page progress in PostgreSQL
        total_chunks = len(chunks)
        db_chunk_rows: list[DocumentChunk] = []

        for start in range(0, total_chunks, EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            ids = [f"{doc_id_str}:{c.chunk_index}" for c in batch]
            contextual_texts = [c.contextual_text for c in batch]
            metadatas = [c.metadata for c in batch]

            embeddings = await asyncio.to_thread(embedding_service.embed_documents, contextual_texts)
            await asyncio.to_thread(
                vector_store.upsert_chunks,
                ids=ids,
                documents=contextual_texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            for chroma_id, chunk in zip(ids, batch):
                db_chunk_rows.append(
                    DocumentChunk(
                        document_id=document_id,
                        chroma_id=chroma_id,
                        chunk_index=chunk.chunk_index,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        chapter=chunk.chapter,
                        section=chunk.section,
                        chunk_type=chunk.chunk_type,
                        token_count=chunk.token_count,
                        content=chunk.content,
                        metadata_json=chunk.metadata,
                    )
                )

            # Advance progress proportionally from 20% -> 95% across embedding batches
            max_page_seen = max(c.page_end for c in batch)
            async with session_factory() as db:
                doc = await db.get(Document, document_id)
                if doc is not None:
                    doc.pages_processed = max(doc.pages_processed, min(profile.page_count, max_page_seen))
                    await db.commit()

        # Persist relational chunk mirror and finalize status
        async with session_factory() as db:
            for offset in range(0, len(db_chunk_rows), DB_CHUNK_BATCH_SIZE):
                db.add_all(db_chunk_rows[offset : offset + DB_CHUNK_BATCH_SIZE])
            doc = await db.get(Document, document_id)
            if doc is not None:
                doc.ingestion_status = IngestionStatus.COMPLETED.value
                doc.pages_processed = profile.page_count
                doc.chunk_count = total_chunks
                doc.error_message = None
            await db.commit()

    except Exception as exc:
        logger.exception("Document ingestion failed for %s", doc_id_str)
        try:
            await asyncio.to_thread(vector_store.delete_document_chunks, doc_id_str)
        except Exception:
            logger.exception("Cleanup of partial Chroma vectors failed for %s", doc_id_str)
        await _mark_failed(session_factory, document_id, str(exc))

