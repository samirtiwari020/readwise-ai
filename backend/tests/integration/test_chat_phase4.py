import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import psycopg
import pymupdf
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.database import Base, get_db, get_session_factory
from app.main import create_app
from app.services.agents.llm import get_chat_llm
from app.services.agents.state import QueryPlan, RetrievalAssessment, SubQuery
from app.services.retrieval.embeddings import get_embedding_service
from app.services.retrieval.reranker import get_reranker_service
from app.services.retrieval.vector_store import _cached_vector_store


class _SemanticStubEmbedder:
    _TERMS = ("vector", "chroma", "bm25", "agent", "langgraph", "citation", "page")

    def _encode(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [1.0 if term in lowered else 0.05 for term in self._TERMS]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


class _SimpleOverlapReranker:
    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        q_words = set(query.lower().split())
        return [
            min(1.0, 0.25 + 0.15 * len(q_words & set(passage.lower().split())))
            for passage in passages
        ]


class _FakeGeminiChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-gemini-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = (
            "Book One uses ChromaDB for page-aware vector indexing [1], "
            "while Book Two uses LangGraph for query decomposition [2]."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(self, schema: type, **kwargs: Any):
        if schema is QueryPlan:
            return RunnableLambda(
                lambda _msgs: QueryPlan(
                    query_type="cross_document_compare",
                    needs_comparison_table=True,
                    sub_queries=[SubQuery(query="ChromaDB vector indexing LangGraph decomposition")],
                )
            )
        if schema is RetrievalAssessment:
            def _assess(msgs: list[BaseMessage]) -> RetrievalAssessment:
                text = str(msgs[-1].content)
                # Extract first two chroma_id values mentioned in grader prompt
                import re

                ids = re.findall(r"\[chroma_id=([^\]]+)\]", text)
                return RetrievalAssessment(
                    sufficient=bool(ids),
                    relevant_chunk_ids=ids[:2],
                    reason="Relevant passages found.",
                )

            return RunnableLambda(_assess)
        raise ValueError(f"Unsupported schema: {schema}")


def _build_pdf(title: str, body: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 100), title, fontsize=16)
    page.insert_text((50, 145), body, fontsize=11)
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
async def phase4_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    get_settings.cache_clear()
    _cached_vector_store.cache_clear()
    base_settings = get_settings()
    test_url = _test_db_url(base_settings.database_url)
    _ensure_test_database(test_url)

    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
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
    app.dependency_overrides[get_embedding_service] = lambda: _SemanticStubEmbedder()
    app.dependency_overrides[get_reranker_service] = lambda: _SimpleOverlapReranker()
    app.dependency_overrides[get_chat_llm] = lambda: _FakeGeminiChatModel()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()
    _cached_vector_store.cache_clear()
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    get_settings.cache_clear()


async def test_chat_and_sse_streaming_end_to_end(phase4_client: AsyncClient) -> None:
    # 1. Session fails cleanly if no completed documents exist yet
    empty_session = (
        await phase4_client.post("/api/v1/sessions", json={"title": "New Research Session"})
    ).json()
    no_docs_resp = await phase4_client.post(
        f"/api/v1/sessions/{empty_session['id']}/chat",
        json={"query": "Compare Book One and Book Two"},
    )
    assert no_docs_resp.status_code == 400

    # 2. Upload two books (BackgroundTasks runs ingestion cleanly)
    up1 = await phase4_client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "Book_One.pdf",
                _build_pdf(
                    "Chapter 1: Vector Storage",
                    "Book One uses ChromaDB for page-aware vector indexing and metadata search.",
                ),
                "application/pdf",
            )
        },
    )
    up2 = await phase4_client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "Book_Two.pdf",
                _build_pdf(
                    "Chapter 1: Orchestration",
                    "Book Two uses LangGraph for agentic query decomposition and verification.",
                ),
                "application/pdf",
            )
        },
    )
    doc1_id = up1.json()["document"]["id"]
    doc2_id = up2.json()["document"]["id"]

    # 3. Create scoped session and run synchronous chat
    session = (
        await phase4_client.post(
            "/api/v1/sessions",
            json={"title": "New Research Session", "selected_document_ids": [doc1_id, doc2_id]},
        )
    ).json()
    session_id = session["id"]

    chat_resp = await phase4_client.post(
        f"/api/v1/sessions/{session_id}/chat",
        json={"query": "Compare Book One and Book Two architectures"},
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["user_message"]["role"] == "user"
    assistant_msg = chat_data["assistant_message"]
    assert assistant_msg["role"] == "assistant"
    assert "[1]" in assistant_msg["content"]
    assert len(assistant_msg["citations"]) == 2
    assert assistant_msg["agent_trace"]["query_type"] == "cross_document_compare"
    assert assistant_msg["agent_trace"]["verified_citations"] == 2

    # Session auto-titled from first query & messages persisted
    detail = (await phase4_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert detail["title"] == "Compare Book One and Book Two architectures"
    assert len(detail["messages"]) == 2

    # 4. Run SSE streaming chat endpoint
    async with phase4_client.stream(
        "POST",
        f"/api/v1/sessions/{session_id}/chat/stream",
        json={"query": "Summarize citation tracking across both books"},
    ) as stream_resp:
        assert stream_resp.status_code == 200
        body = (await stream_resp.aread()).decode("utf-8")

    assert "event: status" in body
    assert "event: trace" in body
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body

    # Verify streamed assistant message persisted to PostgreSQL
    detail_after = (await phase4_client.get(f"/api/v1/sessions/{session_id}")).json()
    assert len(detail_after["messages"]) == 4
    last_done_line = [ln for ln in body.splitlines() if ln.startswith("data:")][-1]
    parsed_done = json.loads(last_done_line.removeprefix("data:").strip())
    assert parsed_done["agent_trace"]["verified_citations"] == 2

