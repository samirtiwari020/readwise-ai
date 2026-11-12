import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.database import Base, get_db, get_session_factory
from app.main import create_app
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import _cached_vector_store


class _DeterministicEmbeddingService:
    """Fast deterministic hash-based embeddings for Phase 3 end-to-end tests."""

    def _encode(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [b / 255.0 for b in digest[:16]]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


def _make_rich_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    for page_num in range(1, 5):
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 35), "AI Systems Reference Book", fontsize=9)
        page.insert_text((50, 100), f"Chapter {page_num}: Topic {page_num}", fontsize=16)
        page.insert_text(
            (50, 145),
            f"Detailed technical discussion on page {page_num} covering RAG architectures, "
            f"page-aware chunking, metadata filtering, and grounded citations.",
            fontsize=11,
        )
        page.insert_text((280, 760), f"{page_num}", fontsize=9)
    doc.set_toc([[1, f"Chapter {i}: Topic {i}", i] for i in range(1, 5)])
    data = doc.tobytes()
    doc.close()
    return data


def _make_empty_pdf_bytes() -> bytes:
    doc = pymupdf.open()
    doc.new_page(width=612, height=792)
    data = doc.tobytes()
    doc.close()
    return data


def _test_db_url(base_url: str) -> str:
    prefix, _db = base_url.rsplit("/", 1)
    return f"{prefix}/readwise_ai_test"


def _ensure_test_database(async_test_url: str) -> None:
    sync_url = async_test_url.replace("+asyncpg", "")
    prefix, db_name = sync_url.rsplit("/", 1)
    try:
        with psycopg.connect(sync_url):
            return
    except psycopg.OperationalError as exc:
        if "does not exist" not in str(exc):
            raise
        with psycopg.connect(f"{prefix}/postgres", autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{db_name}"')


@pytest.fixture
async def phase3_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[AsyncClient, Path]]:
    get_settings.cache_clear()
    _cached_vector_store.cache_clear()
    base_settings = get_settings()
    test_url = _test_db_url(base_settings.database_url)
    _ensure_test_database(test_url)

    upload_dir = tmp_path / "uploads"
    chroma_dir = tmp_path / "chroma"
    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))
    monkeypatch.setenv("DATABASE_URL", test_url)
    get_settings.cache_clear()

    test_engine = create_async_engine(test_url, poolclass=NullPool)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    test_session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    app = create_app()

    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with test_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_session_factory] = lambda: test_session_factory
    app.dependency_overrides[get_embedding_service] = _DeterministicEmbeddingService

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, chroma_dir

    app.dependency_overrides.clear()
    _cached_vector_store.cache_clear()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    get_settings.cache_clear()


async def test_upload_triggers_page_aware_ingestion_and_delete_purges_chroma(
    phase3_client: tuple[AsyncClient, Path],
) -> None:
    client, chroma_dir = phase3_client
    pdf_bytes = _make_rich_pdf_bytes()

    upload = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("AI_Systems_Book.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["document"]["id"]

    # BackgroundTasks completes inside ASGITransport request lifecycle
    status_resp = await client.get(f"/api/v1/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["ingestion_status"] == "completed"
    assert status_data["page_count"] == 4
    assert status_data["pages_processed"] == 4
    assert status_data["progress_percent"] == 100.0
    assert status_data["chunk_count"] >= 4

    detail = (await client.get(f"/api/v1/documents/{doc_id}")).json()
    assert len(detail["toc_json"]) == 4

    page2_chunks = (await client.get(f"/api/v1/documents/{doc_id}/chunks?page=2")).json()["chunks"]
    assert len(page2_chunks) >= 1
    assert any("page 2" in chunk["content"] for chunk in page2_chunks)
    assert all(chunk["page_start"] <= 2 <= chunk["page_end"] for chunk in page2_chunks)

    # Verify Chroma vector count matches Postgres chunk count
    store = _cached_vector_store(str(chroma_dir.resolve()), get_settings().chroma_collection_name)
    assert store.count() == status_data["chunk_count"]

    # Re-ingest remains idempotent (no duplicate chunks)
    reingest = await client.post(f"/api/v1/documents/{doc_id}/reingest")
    assert reingest.status_code == 200
    status_after = (await client.get(f"/api/v1/documents/{doc_id}/status")).json()
    assert status_after["ingestion_status"] == "completed"
    assert status_after["chunk_count"] == status_data["chunk_count"]
    assert store.count() == status_data["chunk_count"]

    # Delete purges both Postgres and ChromaDB
    deleted = await client.delete(f"/api/v1/documents/{doc_id}")
    assert deleted.status_code == 204
    assert store.count() == 0


async def test_empty_scanned_pdf_marks_ingestion_failed(
    phase3_client: tuple[AsyncClient, Path],
) -> None:
    client, chroma_dir = phase3_client
    upload = await client.post(
        "/api/v1/documents/upload",
        files={"file": ("blank_scan.pdf", _make_empty_pdf_bytes(), "application/pdf")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["document"]["id"]

    status_data = (await client.get(f"/api/v1/documents/{doc_id}/status")).json()
    assert status_data["ingestion_status"] == "failed"
    assert status_data["progress_percent"] == 0.0
    assert "No extractable text" in (status_data["error_message"] or "")

    store = _cached_vector_store(str(chroma_dir.resolve()), get_settings().chroma_collection_name)
    assert store.count() == 0

