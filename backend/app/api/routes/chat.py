import json
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from app.core.database import get_db, get_session_factory
from app.db.models.document import Document, IngestionStatus
from app.db.models.session import Message, MessageRole, ResearchSession
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.session import MessageOut
from app.services.agents.graph import build_agent_trace, build_research_graph
from app.services.agents.llm import get_chat_llm
from app.services.agents.state import DocumentCatalogEntry, ResearchAgentState
from app.services.retrieval.embeddings import EmbeddingService, get_embedding_service
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.reranker import RerankerService, get_reranker_service
from app.services.retrieval.vector_store import ChromaVectorStore, get_vector_store

router = APIRouter(prefix="/sessions", tags=["chat"])

_STAGE_LABELS = {
    "decompose": ("decomposing", "Analyzing intent and decomposing sub-queries..."),
    "retrieve": ("retrieving", "Searching ChromaDB + BM25 and reranking passages..."),
    "grade": ("grading", "Grading retrieved passages for evidence sufficiency..."),
    "synthesize": ("synthesizing", "Generating grounded answer with inline citations..."),
    "verify": ("verifying", "Verifying page-level source attributions..."),
}


def _format_history(messages: list[Message], max_turns: int = 6) -> str:
    """Format recent session turns concisely for query coreference resolution."""
    recent = messages[-max_turns:]
    lines: list[str] = []
    for msg in recent:
        role_label = "User" if msg.role == MessageRole.USER.value else "Assistant"
        snippet = msg.content.strip().replace("\n", " ")[:400]
        lines.append(f"{role_label}: {snippet}")
    return "\n".join(lines)


def _auto_session_title(query: str) -> str:
    cleaned = " ".join(query.split()).strip()
    if len(cleaned) <= 64:
        return cleaned
    return f"{cleaned[:61].rsplit(' ', 1)[0]}..."


async def _prepare_chat_execution(
    db: AsyncSession,
    session_id: uuid.UUID,
    payload: ChatRequest,
    retriever: HybridRetriever,
) -> tuple[ResearchSession, Message, ResearchAgentState]:
    """Validate session & indexed documents, persist user message, and build initial graph state."""
    stmt = (
        select(ResearchSession)
        .where(ResearchSession.id == session_id)
        .options(selectinload(ResearchSession.messages))
    )
    session_obj = await db.scalar(stmt)
    if session_obj is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Research session not found.",
        )

    raw_scope = (
        [str(doc_id) for doc_id in payload.document_ids]
        if payload.document_ids is not None
        else [str(doc_id) for doc_id in (session_obj.selected_document_ids or [])]
    )

    doc_stmt = select(Document).where(Document.ingestion_status == IngestionStatus.COMPLETED.value)
    if raw_scope:
        scope_uuids = [uuid.UUID(item) for item in raw_scope]
        doc_stmt = doc_stmt.where(Document.id.in_(scope_uuids))

    completed_docs = (await db.scalars(doc_stmt.order_by(Document.created_at.desc()))).all()
    if not completed_docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No completed indexed documents are available in the selected scope.",
        )

    scoped_ids = [str(doc.id) for doc in completed_docs]
    catalog = [
        DocumentCatalogEntry(
            document_id=str(doc.id),
            title=doc.title,
            page_count=doc.page_count,
            toc_titles=[str(item.get("title", "")) for item in (doc.toc_json or [])[:12]],
        )
        for doc in completed_docs
    ]
    history_text = _format_history(session_obj.messages)
    corpus = await retriever.load_corpus(db, scoped_ids)

    clean_query = payload.query.strip()
    user_message = Message(
        session_id=session_obj.id,
        role=MessageRole.USER.value,
        content=clean_query,
    )
    db.add(user_message)
    if session_obj.title == "New Research Session" and not session_obj.messages:
        session_obj.title = _auto_session_title(clean_query)
    session_obj.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(user_message)

    initial_state: ResearchAgentState = {
        "query": clean_query,
        "history_text": history_text,
        "document_catalog": catalog,
        "scoped_document_ids": scoped_ids,
        "corpus": corpus,
        "steps": [],
    }
    return session_obj, user_message, initial_state


def _get_hybrid_retriever(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    reranker_service: RerankerService = Depends(get_reranker_service),
) -> HybridRetriever:
    return HybridRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        reranker_service=reranker_service,
    )


@router.post("/{session_id}/chat", response_model=ChatResponse)
async def chat_with_documents(
    session_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    retriever: HybridRetriever = Depends(_get_hybrid_retriever),
    llm: BaseChatModel = Depends(get_chat_llm),
) -> ChatResponse:
    """Execute the agentic RAG pipeline and persist the cited assistant response."""
    session_obj, user_message, initial_state = await _prepare_chat_execution(
        db=db,
        session_id=session_id,
        payload=payload,
        retriever=retriever,
    )

    start_time = time.perf_counter()
    graph = build_research_graph(llm=llm, retriever=retriever)
    final_state: ResearchAgentState = await graph.ainvoke(initial_state)
    trace = build_agent_trace(final_state, start_time)

    assistant_message = Message(
        session_id=session_obj.id,
        role=MessageRole.ASSISTANT.value,
        content=final_state.get("final_answer", ""),
        citations_json=final_state.get("citations", []),
        agent_trace_json=trace,
    )
    db.add(assistant_message)
    session_obj.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(assistant_message)

    return ChatResponse(
        session_id=session_obj.id,
        user_message=MessageOut.model_validate(user_message),
        assistant_message=MessageOut.model_validate(assistant_message),
    )


@router.post("/{session_id}/chat/stream")
async def stream_chat_with_documents(
    session_id: uuid.UUID,
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
    retriever: HybridRetriever = Depends(_get_hybrid_retriever),
    llm: BaseChatModel = Depends(get_chat_llm),
) -> EventSourceResponse:
    """Stream live LangGraph agent stage updates, verified answer text, and citations over SSE."""
    session_obj, user_message, initial_state = await _prepare_chat_execution(
        db=db,
        session_id=session_id,
        payload=payload,
        retriever=retriever,
    )
    graph = build_research_graph(llm=llm, retriever=retriever)
    session_uuid = session_obj.id

    async def _event_generator() -> AsyncIterator[dict[str, str]]:
        start_time = time.perf_counter()
        merged_state: dict[str, Any] = dict(initial_state)

        yield {
            "event": "user_message",
            "data": json.dumps(MessageOut.model_validate(user_message).model_dump(mode="json")),
        }

        try:
            async for update in graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in update.items():
                    merged_state.update(node_output)
                    stage, detail = _STAGE_LABELS.get(node_name, (node_name, "Processing..."))
                    yield {
                        "event": "status",
                        "data": json.dumps({"stage": stage, "detail": detail}),
                    }
                    if node_name == "decompose":
                        yield {
                            "event": "trace",
                            "data": json.dumps(
                                {
                                    "query_type": node_output.get("query_type"),
                                    "sub_queries": [
                                        sq.query for sq in node_output.get("sub_queries", [])
                                    ],
                                }
                            ),
                        }

            final_answer = str(merged_state.get("final_answer", ""))
            citations = merged_state.get("citations", [])
            trace = build_agent_trace(merged_state, start_time)  # type: ignore[arg-type]

            # Stream verified answer in smooth token-like chunks after citation guard passes
            chunk_size = 48
            for idx in range(0, len(final_answer), chunk_size):
                yield {
                    "event": "token",
                    "data": json.dumps({"delta": final_answer[idx : idx + chunk_size]}),
                }

            yield {"event": "citations", "data": json.dumps(citations)}

            async with session_factory() as stream_db:
                assistant_message = Message(
                    session_id=session_uuid,
                    role=MessageRole.ASSISTANT.value,
                    content=final_answer,
                    citations_json=citations,
                    agent_trace_json=trace,
                )
                stream_db.add(assistant_message)
                target_session = await stream_db.get(ResearchSession, session_uuid)
                if target_session is not None:
                    target_session.updated_at = datetime.now(UTC)
                await stream_db.commit()
                await stream_db.refresh(assistant_message)
                msg_payload = MessageOut.model_validate(assistant_message).model_dump(mode="json")

            yield {
                "event": "done",
                "data": json.dumps({"assistant_message": msg_payload, "agent_trace": trace}),
            }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)}),
            }

    return EventSourceResponse(_event_generator())

