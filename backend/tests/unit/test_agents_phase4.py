from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.services.agents.citation_guard import ABSTENTION_MESSAGE, verify_and_build_citations
from app.services.agents.graph import build_research_graph
from app.services.agents.state import DocumentCatalogEntry, QueryPlan, RetrievalAssessment, SubQuery
from app.services.retrieval.hybrid_retriever import CorpusChunkRecord, HybridRetriever, RetrievedChunk, tokenize_for_bm25
from app.services.retrieval.vector_store import ChromaVectorStore


class _KeywordEmbeddingService:
    """Tiny keyword-aligned embeddings for deterministic unit tests."""

    _VOCAB = ("attention", "transformer", "bm25", "vector", "theorem", "reranker")

    def _vec(self, text: str) -> list[float]:
        lowered = text.lower()
        vec = [1.0 if term in lowered else 0.05 for term in self._VOCAB]
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


class _KeywordReranker:
    """Scores passages by token overlap against the query."""

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        q_tokens = set(tokenize_for_bm25(query))
        scores: list[float] = []
        for passage in passages:
            p_tokens = set(tokenize_for_bm25(passage))
            overlap = len(q_tokens & p_tokens)
            scores.append(min(1.0, 0.2 + 0.2 * overlap))
        return scores


class _ScriptedResearchLLM(BaseChatModel):
    """Minimal LangChain BaseChatModel stub supporting structured outputs and async generation."""

    planner_fn: Any
    grader_fn: Any
    synth_fn: Any

    @property
    def _llm_type(self) -> str:
        return "scripted-research-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self.synth_fn(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

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
            return RunnableLambda(lambda msgs: self.planner_fn(msgs))
        if schema is RetrievalAssessment:
            return RunnableLambda(lambda msgs: self.grader_fn(msgs))
        raise ValueError(f"Unsupported structured schema: {schema}")


def test_citation_guard_strips_invalid_and_renumbers_in_appearance_order() -> None:
    chunks = [
        RetrievedChunk(
            chroma_id="doc-a:0",
            document_id="doc-a",
            document_title="Book A",
            page_start=12,
            page_end=12,
            chapter="Ch 1",
            section=None,
            chunk_type="narrative",
            content="Book A explains dense vector indexing using cosine similarity.",
            score=0.92,
            source_method="hybrid",
        ),
        RetrievedChunk(
            chroma_id="doc-b:3",
            document_id="doc-b",
            document_title="Book B",
            page_start=45,
            page_end=46,
            chapter="Ch 3",
            section="BM25",
            chunk_type="narrative",
            content="Book B uses BM25 lexical scoring to match exact theorem identifiers.",
            score=0.89,
            source_method="bm25",
        ),
    ]

    raw = "Book B uses BM25 [2, 99], whereas Book A uses cosine similarity [1] [0]."
    cleaned, citations, abstained = verify_and_build_citations(raw, chunks)

    assert abstained is False
    assert "[99]" not in cleaned
    assert "[0]" not in cleaned
    assert cleaned == "Book B uses BM25 [1], whereas Book A uses cosine similarity [2]."
    assert len(citations) == 2
    assert citations[0]["citation_index"] == 1
    assert citations[0]["document_title"] == "Book B"
    assert citations[0]["page_start"] == 45
    assert citations[1]["citation_index"] == 2
    assert citations[1]["document_title"] == "Book A"
    assert citations[1]["page_start"] == 12


def test_citation_guard_abstains_when_no_relevant_chunks() -> None:
    cleaned, citations, abstained = verify_and_build_citations("Unsupported guess [1].", [])
    assert abstained is True
    assert citations == []
    assert cleaned == ABSTENTION_MESSAGE


async def test_hybrid_retriever_fuses_dense_and_bm25_with_page_filters(tmp_path: Path) -> None:
    embedder = _KeywordEmbeddingService()
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma", collection_name="hybrid_test")
    reranker = _KeywordReranker()

    docs = [
        ("doc-1:0", "doc-1", "Systems Book", 5, "Theorem 4.2 proves BM25 preserves exact identifiers."),
        ("doc-1:1", "doc-1", "Systems Book", 18, "Attention mechanisms compute scaled dot products."),
        ("doc-2:0", "doc-2", "ML Book", 30, "Cross-encoder reranker refines top candidate passages."),
    ]
    embeddings = embedder.embed_documents([item[4] for item in docs])
    store.upsert_chunks(
        ids=[item[0] for item in docs],
        documents=[item[4] for item in docs],
        embeddings=embeddings,
        metadatas=[
            {"document_id": item[1], "page_start": item[3], "page_end": item[3]}
            for item in docs
        ],
    )

    corpus = [
        CorpusChunkRecord(
            chroma_id=cid,
            document_id=did,
            document_title=title,
            page_start=pg,
            page_end=pg,
            chapter=None,
            section=None,
            chunk_type="narrative",
            content=text,
            tokens=tokenize_for_bm25(text),
        )
        for cid, did, title, pg, text in docs
    ]

    retriever = HybridRetriever(embedder, store, reranker)
    hits = await retriever.retrieve(
        query="What does Theorem 4.2 prove about BM25?",
        corpus=corpus,
        document_ids=["doc-1", "doc-2"],
        top_k=5,
        rerank_top_n=2,
    )
    assert hits[0].chroma_id == "doc-1:0"
    assert hits[0].source_method == "hybrid"

    page_filtered = await retriever.retrieve(
        query="attention",
        corpus=corpus,
        document_ids=["doc-1"],
        page_start=15,
        page_end=20,
    )
    assert [h.chroma_id for h in page_filtered] == ["doc-1:1"]


async def test_langgraph_cross_doc_comparison_and_retry_loop(tmp_path: Path) -> None:
    embedder = _KeywordEmbeddingService()
    store = ChromaVectorStore(persist_dir=tmp_path / "chroma", collection_name="agent_test")
    reranker = _KeywordReranker()

    docs = [
        ("doc-a:0", "doc-a", "Book Alpha", 10, "Book Alpha defines vector retrieval using HNSW graphs."),
        ("doc-b:0", "doc-b", "Book Beta", 22, "Book Beta combines BM25 sparse scoring with rerankers."),
    ]
    store.upsert_chunks(
        ids=[item[0] for item in docs],
        documents=[item[4] for item in docs],
        embeddings=embedder.embed_documents([item[4] for item in docs]),
        metadatas=[
            {"document_id": item[1], "page_start": item[3], "page_end": item[3]}
            for item in docs
        ],
    )
    corpus = [
        CorpusChunkRecord(
            chroma_id=cid,
            document_id=did,
            document_title=title,
            page_start=pg,
            page_end=pg,
            chapter="Ch 1",
            section=None,
            chunk_type="narrative",
            content=text,
            tokens=tokenize_for_bm25(text),
        )
        for cid, did, title, pg, text in docs
    ]

    grade_calls = {"count": 0}

    def _planner(_msgs: list[BaseMessage]) -> QueryPlan:
        return QueryPlan(
            query_type="cross_document_compare",
            needs_comparison_table=True,
            sub_queries=[
                SubQuery(query="vector retrieval", target_document_ids=["doc-a"]),
                SubQuery(query="BM25 reranker", target_document_ids=["doc-b"]),
            ],
        )

    def _grader(_msgs: list[BaseMessage]) -> RetrievalAssessment:
        grade_calls["count"] += 1
        if grade_calls["count"] == 1:
            return RetrievalAssessment(
                sufficient=False,
                relevant_chunk_ids=[],
                rewritten_query="HNSW vector retrieval BM25 rerankers",
                reason="Need stronger combined query.",
            )
        return RetrievalAssessment(
            sufficient=True,
            relevant_chunk_ids=["doc-a:0", "doc-b:0"],
            reason="Both books retrieved.",
        )

    def _synth(_msgs: list[BaseMessage]) -> str:
        return (
            "| Book | Method |\n| --- | --- |\n"
            "| Book Alpha | HNSW vector retrieval [1] |\n"
            "| Book Beta | BM25 and rerankers [2] |"
        )

    llm = _ScriptedResearchLLM(planner_fn=_planner, grader_fn=_grader, synth_fn=_synth)
    retriever = HybridRetriever(embedder, store, reranker)
    graph = build_research_graph(llm=llm, retriever=retriever, top_k=5, rerank_top_n=4)

    result = await graph.ainvoke(
        {
            "query": "Compare retrieval in Book Alpha and Book Beta",
            "history_text": "",
            "document_catalog": [
                DocumentCatalogEntry(document_id="doc-a", title="Book Alpha", page_count=100),
                DocumentCatalogEntry(document_id="doc-b", title="Book Beta", page_count=120),
            ],
            "scoped_document_ids": ["doc-a", "doc-b"],
            "corpus": corpus,
            "steps": [],
        }
    )

    assert grade_calls["count"] == 2
    assert result["retry_count"] == 1
    assert result["abstained"] is False
    assert len(result["citations"]) == 2
    assert {c["document_title"] for c in result["citations"]} == {"Book Alpha", "Book Beta"}

