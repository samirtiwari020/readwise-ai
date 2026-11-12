import hashlib
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import psycopg
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import pymupdf
from app.core.config import get_settings
from app.core.database import Base, get_db, get_session_factory
from app.main import create_app
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.vector_store import _cached_vector_store


class _StubEmbeddingService:
    """Fast deterministic embedding stub for Phase 2 API CRUD tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [0.1] * 8


def _make_minimal_pdf_bytes(marker: str = "Readwise AI Test Page") -> bytes:
    """Create a valid minimal PDF byte payload for upload integration tests."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 100), marker, fontsize=12)
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
async def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """Provide an isolated HTTP client backed by readwise_ai_test PostgreSQL database."""
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
    app.dependency_overrides[get_embedding_service] = _StubEmbeddingService

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    _cached_vector_store.cache_clear()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    get_settings.cache_clear()


async def test_health_endpoint(api_client: AsyncClient) -> None:
    response = await api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database_connected"] is True
    assert data["llm_provider"] == "gemini"


async def test_document_upload_validation_dedup_and_crud(
    api_client: AsyncClient, tmp_path: Path
) -> None:
    # 1. Reject non-PDF filename
    bad_ext = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad_ext.status_code == 400

    # 2. Reject spoofed .pdf without %PDF- magic header
    bad_magic = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("fake.pdf", b"not actually a pdf", "application/pdf")},
    )
    assert bad_magic.status_code == 400
    assert list((tmp_path / "uploads").glob("*")) == []

    # 3. Upload valid PDF
    pdf_bytes = _make_minimal_pdf_bytes("Sample Book 1")
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()
    created = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("Clean_Architecture.pdf", pdf_bytes, "application/pdf")},
    )
    assert created.status_code == 201
    payload = created.json()
    assert payload["deduplicated"] is False
    doc = payload["document"]
    doc_id = doc["id"]
    assert doc["title"] == "Clean Architecture"
    assert doc["file_hash"] == expected_hash
    assert doc["ingestion_status"] == "pending"

    # 4. Duplicate upload returns deduplicated=True with same ID
    duplicate = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("Copy_of_Book.pdf", pdf_bytes, "application/pdf")},
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["deduplicated"] is True
    assert duplicate.json()["document"]["id"] == doc_id

    # 5. List, detail, status, and raw PDF file streaming
    listed = await api_client.get("/api/v1/documents")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    detail = await api_client.get(f"/api/v1/documents/{doc_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == doc_id

    status_resp = await api_client.get(f"/api/v1/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["ingestion_status"] == "completed"
    assert status_resp.json()["progress_percent"] == 100.0

    file_resp = await api_client.get(f"/api/v1/documents/{doc_id}/file")
    assert file_resp.status_code == 200
    assert file_resp.headers["content-type"] == "application/pdf"
    assert file_resp.content == pdf_bytes

    # 6. Delete removes DB record and file on disk
    deleted = await api_client.delete(f"/api/v1/documents/{doc_id}")
    assert deleted.status_code == 204
    assert list((tmp_path / "uploads").glob("*")) == []
    assert (await api_client.get(f"/api/v1/documents/{doc_id}")).status_code == 404


async def test_session_crud_and_document_scope_validation(api_client: AsyncClient) -> None:
    # 1. Invalid document ID rejected cleanly
    missing_doc = str(uuid.uuid4())
    invalid_resp = await api_client.post(
        "/api/v1/sessions",
        json={"title": "Bad Session", "selected_document_ids": [missing_doc]},
    )
    assert invalid_resp.status_code == 404

    # 2. Upload real doc and create scoped session
    upload = await api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("book.pdf", _make_minimal_pdf_bytes("Session Book"), "application/pdf")},
    )
    doc_id = upload.json()["document"]["id"]

    created = await api_client.post(
        "/api/v1/sessions",
        json={"title": "Chapter 3 Comparison", "selected_document_ids": [doc_id]},
    )
    assert created.status_code == 201
    session = created.json()
    session_id = session["id"]
    assert session["selected_document_ids"] == [doc_id]

    # 3. List, get detail, patch, delete
    listed = await api_client.get("/api/v1/sessions")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    detail = await api_client.get(f"/api/v1/sessions/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["messages"] == []

    patched = await api_client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"title": "Updated Title", "selected_document_ids": []},
    )
    assert patched.status_code == 200
    assert patched.json()["title"] == "Updated Title"
    assert patched.json()["selected_document_ids"] == []

    deleted = await api_client.delete(f"/api/v1/sessions/{session_id}")
    assert deleted.status_code == 204
    assert (await api_client.get(f"/api/v1/sessions/{session_id}")).status_code == 404

