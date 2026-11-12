import asyncio
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.services.agents.citation_guard import ABSTENTION_MESSAGE, verify_and_build_citations
from app.services.agents.prompts import (
    GRADER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    format_catalog_for_prompt,
    format_passages_for_grading,
    format_passages_for_synthesis,
)
from app.services.agents.state import (
    QueryPlan,
    ResearchAgentState,
    RetrievalAssessment,
    SubQuery,
)
from app.services.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk


def _interleave_by_document(
    chunks: list[RetrievedChunk],
    limit: int,
) -> list[RetrievedChunk]:
    """Balance top retrieved chunks round-robin across documents for fair comparison."""
    buckets: dict[str, list[RetrievedChunk]] = {}
    for chunk in sorted(chunks, key=lambda item: item.score, reverse=True):
        buckets.setdefault(chunk.document_id, []).append(chunk)

    ordered: list[RetrievedChunk] = []
    doc_queues = list(buckets.values())
    while doc_queues and len(ordered) < limit:
        next_queues: list[list[RetrievedChunk]] = []
        for queue in doc_queues:
            if queue and len(ordered) < limit:
                ordered.append(queue.pop(0))
            if queue:
                next_queues.append(queue)
        doc_queues = next_queues
    return ordered


def _merge_retrieved_chunks(
    existing: list[RetrievedChunk],
    incoming: list[RetrievedChunk],
    balance_docs: bool,
    limit: int,
) -> list[RetrievedChunk]:
    """Deduplicate chunks by chroma_id retaining highest reranker score."""
    best_by_id: dict[str, RetrievedChunk] = {}
    for chunk in [*existing, *incoming]:
        prev = best_by_id.get(chunk.chroma_id)
        if prev is None or chunk.score > prev.score:
            best_by_id[chunk.chroma_id] = chunk

    merged = list(best_by_id.values())
    if balance_docs:
        return _interleave_by_document(merged, limit=limit)
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:limit]


def build_research_graph(
    llm: BaseChatModel,
    retriever: HybridRetriever,
    top_k: int | None = None,
    rerank_top_n: int | None = None,
):
    """Compile the stateful LangGraph agentic RAG workflow."""
    settings = get_settings()
    k = top_k or settings.retrieval_top_k
    n = rerank_top_n or settings.rerank_top_n

    planner_llm = llm.with_structured_output(QueryPlan)
    grader_llm = llm.with_structured_output(RetrievalAssessment)

    async def decompose_node(state: ResearchAgentState) -> dict[str, Any]:
        query = state["query"]
        catalog = state.get("document_catalog", [])
        scoped_set = set(state.get("scoped_document_ids", []))
        history_text = state.get("history_text", "").strip()

        prompt_parts = [
            f"Available Scoped Documents:\n{format_catalog_for_prompt(catalog)}",
        ]
        if history_text:
            prompt_parts.append(f"Recent Conversation History:\n{history_text}")
        prompt_parts.append(f"Current User Question:\n{query}")

        plan: QueryPlan = await planner_llm.ainvoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content="\n\n".join(prompt_parts)),
            ]
        )

        # Sanitize target_document_ids to stay strictly within scoped documents
        sanitized_sub_queries: list[SubQuery] = []
        for sq in plan.sub_queries:
            valid_targets = [doc_id for doc_id in sq.target_document_ids if doc_id in scoped_set]
            sanitized_sub_queries.append(
                SubQuery(
                    query=sq.query.strip() or query,
                    target_document_ids=valid_targets,
                    page_start=sq.page_start,
                    page_end=sq.page_end,
                )
            )

        if plan.query_type != "out_of_scope" and not sanitized_sub_queries:
            sanitized_sub_queries = [SubQuery(query=query)]

        steps = list(state.get("steps", []))
        steps.append(
            f"decompose:{plan.query_type} ({len(sanitized_sub_queries)} sub-queries)"
        )
        return {
            "query_type": plan.query_type,
            "sub_queries": sanitized_sub_queries,
            "needs_comparison_table": plan.needs_comparison_table,
            "retry_count": 0,
            "steps": steps,
        }

    async def retrieve_node(state: ResearchAgentState) -> dict[str, Any]:
        sub_queries = state.get("sub_queries", [])
        scoped_ids = state.get("scoped_document_ids", [])
        corpus = state.get("corpus", [])
        query_type = state.get("query_type", "single_fact")
        retry_count = state.get("retry_count", 0)

        if query_type == "out_of_scope" or not sub_queries:
            steps = list(state.get("steps", []))
            steps.append("retrieve:skipped (out_of_scope)")
            return {
                "retrieved_chunks": [],
                "total_retrieved_count": 0,
                "steps": steps,
            }

        # On retry, search with the rewritten query across all scoped documents
        if retry_count > 0 and state.get("rewritten_query"):
            active_queries = [SubQuery(query=state["rewritten_query"] or state["query"])]
        else:
            active_queries = sub_queries

        per_query_n = max(4, n if len(active_queries) == 1 else max(3, n // len(active_queries) + 2))
        tasks = [
            retriever.retrieve(
                query=sq.query,
                corpus=corpus,
                document_ids=sq.target_document_ids or scoped_ids,
                page_start=sq.page_start,
                page_end=sq.page_end,
                top_k=k,
                rerank_top_n=per_query_n,
            )
            for sq in active_queries
        ]
        batches = await asyncio.gather(*tasks)
        incoming = [chunk for batch in batches for chunk in batch]

        max_pool = max(n, n + 4 if query_type == "cross_document_compare" else n)
        merged = _merge_retrieved_chunks(
            existing=state.get("retrieved_chunks", []),
            incoming=incoming,
            balance_docs=(query_type == "cross_document_compare"),
            limit=max_pool,
        )

        steps = list(state.get("steps", []))
        steps.append(f"retrieve:pass_{retry_count + 1} ({len(merged)} chunks)")
        return {
            "retrieved_chunks": merged,
            "total_retrieved_count": state.get("total_retrieved_count", 0) + len(incoming),
            "steps": steps,
        }

    async def grade_node(state: ResearchAgentState) -> dict[str, Any]:
        retrieved = state.get("retrieved_chunks", [])
        retry_count = state.get("retry_count", 0)
        steps = list(state.get("steps", []))

        if not retrieved:
            steps.append("grade:no_candidates")
            return {
                "relevant_chunks": [],
                "needs_retry": False,
                "steps": steps,
            }

        prompt = (
            f"User Question:\n{state['query']}\n\n"
            f"Retrieved Passages:\n{format_passages_for_grading(retrieved)}"
        )
        assessment: RetrievalAssessment = await grader_llm.ainvoke(
            [
                SystemMessage(content=GRADER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        selected_ids = set(assessment.relevant_chunk_ids)
        relevant = [chunk for chunk in retrieved if chunk.chroma_id in selected_ids]

        should_retry = (
            not assessment.sufficient
            and retry_count == 0
            and bool((assessment.rewritten_query or "").strip())
        )
        if should_retry:
            steps.append("grade:insufficient -> rewriting query")
            return {
                "relevant_chunks": relevant,
                "needs_retry": True,
                "retry_count": retry_count + 1,
                "rewritten_query": (assessment.rewritten_query or "").strip(),
                "steps": steps,
            }

        if not assessment.sufficient and not relevant:
            steps.append("grade:abstain (no grounded evidence)")
            return {
                "relevant_chunks": [],
                "needs_retry": False,
                "steps": steps,
            }

        # If grader deemed sufficient without listing explicit IDs, keep top retrieved chunks
        final_relevant = relevant if relevant else (retrieved if assessment.sufficient else [])
        steps.append(f"grade:accepted ({len(final_relevant)} chunks)")
        return {
            "relevant_chunks": final_relevant,
            "needs_retry": False,
            "steps": steps,
        }

    async def synthesize_node(state: ResearchAgentState) -> dict[str, Any]:
        relevant = state.get("relevant_chunks", [])
        steps = list(state.get("steps", []))

        if not relevant:
            steps.append("synthesize:abstained")
            return {
                "raw_answer": ABSTENTION_MESSAGE,
                "steps": steps,
            }

        passages_block = format_passages_for_synthesis(relevant)
        comparison_hint = (
            "\nInclude a concise Markdown comparison table comparing each document's position."
            if state.get("needs_comparison_table") or state.get("query_type") == "cross_document_compare"
            else ""
        )
        history_block = (
            f"Conversation Context:\n{state['history_text']}\n\n"
            if state.get("history_text", "").strip()
            else ""
        )

        system_prompt = SYNTHESIS_SYSTEM_PROMPT.format(max_index=len(relevant))
        user_prompt = (
            f"{history_block}"
            f"User Question:\n{state['query']}{comparison_hint}\n\n"
            f"Numbered Source Passages:\n{passages_block}"
        )

        ai_msg = await llm.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
        steps.append("synthesize:generated")
        return {
            "raw_answer": content,
            "steps": steps,
        }

    async def verify_node(state: ResearchAgentState) -> dict[str, Any]:
        raw_answer = state.get("raw_answer", "")
        relevant = state.get("relevant_chunks", [])
        final_answer, citations, abstained = verify_and_build_citations(raw_answer, relevant)
        steps = list(state.get("steps", []))
        steps.append(f"verify:{len(citations)} citations")
        return {
            "final_answer": final_answer,
            "citations": citations,
            "abstained": abstained,
            "steps": steps,
        }

    def _route_after_grade(state: ResearchAgentState) -> str:
        return "retry" if state.get("needs_retry") else "proceed"

    builder = StateGraph(ResearchAgentState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("verify", verify_node)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"retry": "retrieve", "proceed": "synthesize"},
    )
    builder.add_edge("synthesize", "verify")
    builder.add_edge("verify", END)

    return builder.compile()


def build_agent_trace(state: ResearchAgentState, start_time: float) -> dict[str, Any]:
    """Assemble persisted AgentTrace metadata from final graph state."""
    return {
        "query_type": state.get("query_type"),
        "sub_queries": [sq.query for sq in state.get("sub_queries", [])],
        "steps": state.get("steps", []),
        "retrieved_chunks": state.get("total_retrieved_count", 0),
        "reranked_chunks": len(state.get("retrieved_chunks", [])),
        "verified_citations": len(state.get("citations", [])),
        "abstained": bool(state.get("abstained", False)),
        "latency_ms": int((time.perf_counter() - start_time) * 1000),
    }

