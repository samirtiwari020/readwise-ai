import uuid
from datetime import UTC, datetime

from app.core.config import PROJECT_ROOT, Settings, normalize_database_url
from app.schemas.document import DocumentStatusOut
from app.schemas.session import MessageOut


def test_normalize_database_url_encodes_extra_at_sign() -> None:
    raw = "postgresql+asyncpg://postgres:my@pass@localhost:5433/readwise_ai"
    assert (
        normalize_database_url(raw)
        == "postgresql+asyncpg://postgres:my%40pass@localhost:5433/readwise_ai"
    )


def test_normalize_database_url_preserves_clean_and_encoded_urls() -> None:
    clean = "postgresql+asyncpg://postgres:secret@localhost:5433/readwise_ai"
    already_encoded = "postgresql+asyncpg://postgres:my%40pass@localhost:5433/readwise_ai"
    assert normalize_database_url(clean) == clean
    assert normalize_database_url(already_encoded) == already_encoded


def test_settings_resolves_data_directories_relative_to_project() -> None:
    settings = Settings()
    assert settings.resolved_upload_dir == (PROJECT_ROOT / "data" / "uploads").resolve()
    assert settings.resolved_chroma_dir == (PROJECT_ROOT / "data" / "chroma").resolve()


def test_document_status_progress_percent_bounds() -> None:
    doc_id = uuid.uuid4()
    pending = DocumentStatusOut(
        id=doc_id, ingestion_status="pending", page_count=None, pages_processed=0
    )
    processing = DocumentStatusOut(
        id=doc_id, ingestion_status="processing", page_count=200, pages_processed=50
    )
    completed = DocumentStatusOut(
        id=doc_id, ingestion_status="completed", page_count=200, pages_processed=180
    )
    assert pending.progress_percent == 0.0
    assert processing.progress_percent == 25.0
    assert completed.progress_percent == 100.0


def test_message_out_maps_jsonb_columns_to_api_fields() -> None:
    now = datetime.now(UTC)
    msg = MessageOut.model_validate(
        {
            "id": uuid.uuid4(),
            "session_id": uuid.uuid4(),
            "role": "assistant",
            "content": "Grounded answer [1].",
            "citations_json": [{"citation_index": 1, "page_start": 42}],
            "agent_trace_json": {"query_type": "single_fact", "verified_citations": 1},
            "created_at": now,
        }
    )
    assert msg.citations == [{"citation_index": 1, "page_start": 42}]
    assert msg.agent_trace == {"query_type": "single_fact", "verified_citations": 1}

