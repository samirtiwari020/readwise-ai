import asyncio
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document, DocumentChunk
from app.services.retrieval.embeddings import EmbeddingService
from app.services.retrieval.reranker import RerankerService
from app.services.retrieval.vector_store import ChromaVectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms preserving identifiers."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class RetrievedChunk:
    """Enriched chunk returned by hybrid retrieval and cross-encoder reranking."""

    chroma_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    chapter: str | None
    section: str | None
    chunk_type: str
    content: str
    score: float
    source_method: str  # "dense" | "bm25" | "hybrid"


@dataclass(slots=True)
class CorpusChunkRecord:
    """Lightweight in-memory chunk projection loaded from PostgreSQL for BM25 & hydration."""

    chroma_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    chapter: str | None
    section: str | None
    chunk_type: str
    content: str
    tokens: list[str]


class HybridRetriever:
    """Combines ChromaDB dense retrieval + BM25 lexical search (RRF) + CrossEncoder reranking."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        reranker_service: RerankerService,
        rrf_k: int = 60,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker_service = reranker_service
        self.rrf_k = rrf_k

    async def load_corpus(
        self, db: AsyncSession, document_ids: list[str]
    ) -> list[CorpusChunkRecord]:
        """Load relational chunks for the scoped documents once per research query."""
        if not document_ids:
            return []

        stmt = (
            select(
                DocumentChunk.chroma_id,
                DocumentChunk.document_id,
                Document.title,
                DocumentChunk.page_start,
                DocumentChunk.page_end,
                DocumentChunk.chapter,
                DocumentChunk.section,
                DocumentChunk.chunk_type,
                DocumentChunk.content,
            )
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(DocumentChunk.document_id.in_(document_ids))
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        rows = (await db.execute(stmt)).all()
        return [
            CorpusChunkRecord(
                chroma_id=row.chroma_id,
                document_id=str(row.document_id),
                document_title=row.title,
                page_start=row.page_start,
                page_end=row.page_end,
                chapter=row.chapter,
                section=row.section,
                chunk_type=row.chunk_type,
                content=row.content,
                tokens=tokenize_for_bm25(row.content),
            )
            for row in rows
        ]

    @staticmethod
    def _filter_corpus(
        corpus: list[CorpusChunkRecord],
        document_ids: list[str] | None,
        page_start: int | None,
        page_end: int | None,
    ) -> list[CorpusChunkRecord]:
        doc_set = set(document_ids) if document_ids else None
        filtered: list[CorpusChunkRecord] = []
        for item in corpus:
            if doc_set is not None and item.document_id not in doc_set:
                continue
            if page_start is not None and item.page_end < page_start:
                continue
            if page_end is not None and item.page_start > page_end:
                continue
            filtered.append(item)
        return filtered

    def _bm25_search(
        self,
        query: str,
        subset: list[CorpusChunkRecord],
        top_k: int,
    ) -> list[str]:
        if not subset or top_k <= 0:
            return []
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        tokenized_docs = [item.tokens if item.tokens else ["_"] for item in subset]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            (idx for idx, val in enumerate(scores) if val > 0.0),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]
        return [subset[idx].chroma_id for idx in ranked_indices]

    def _fuse_and_rerank(
        self,
        query: str,
        dense_ids: list[str],
        bm25_ids: list[str],
        by_chroma_id: dict[str, CorpusChunkRecord],
        candidate_k: int,
        rerank_top_n: int,
    ) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}

        for rank, cid in enumerate(dense_ids, start=1):
            if cid not in by_chroma_id:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            sources.setdefault(cid, set()).add("dense")

        for rank, cid in enumerate(bm25_ids, start=1):
            if cid not in by_chroma_id:
                continue
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            sources.setdefault(cid, set()).add("bm25")

        if not rrf_scores:
            return []

        fused_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[
            :candidate_k
        ]
        passages = [by_chroma_id[cid].content for cid in fused_ids]
        rerank_scores = self.reranker_service.score_pairs(query, passages)

        reranked: list[RetrievedChunk] = []
        for cid, rerank_score in zip(fused_ids, rerank_scores):
            rec = by_chroma_id[cid]
            method_set = sources.get(cid, set())
            source_method = "hybrid" if len(method_set) == 2 else next(iter(method_set), "dense")
            reranked.append(
                RetrievedChunk(
                    chroma_id=rec.chroma_id,
                    document_id=rec.document_id,
                    document_title=rec.document_title,
                    page_start=rec.page_start,
                    page_end=rec.page_end,
                    chapter=rec.chapter,
                    section=rec.section,
                    chunk_type=rec.chunk_type,
                    content=rec.content,
                    score=float(rerank_score),
                    source_method=source_method,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:rerank_top_n]

    async def retrieve(
        self,
        query: str,
        corpus: list[CorpusChunkRecord],
        document_ids: list[str] | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        top_k: int = 20,
        rerank_top_n: int = 6,
    ) -> list[RetrievedChunk]:
        """Run dense Chroma search + BM25 lexical search + RRF + CrossEncoder reranking."""
        subset = self._filter_corpus(corpus, document_ids, page_start, page_end)
        if not subset:
            return []

        by_chroma_id = {item.chroma_id: item for item in subset}

        def _sync_search() -> list[RetrievedChunk]:
            query_vec = self.embedding_service.embed_query(query)
            dense_hits = self.vector_store.query(
                query_embedding=query_vec,
                top_k=top_k,
                document_ids=document_ids,
                page_start=page_start,
                page_end=page_end,
            )
            dense_ids = [hit["chroma_id"] for hit in dense_hits]
            bm25_ids = self._bm25_search(query=query, subset=subset, top_k=top_k)
            return self._fuse_and_rerank(
                query=query,
                dense_ids=dense_ids,
                bm25_ids=bm25_ids,
                by_chroma_id=by_chroma_id,
                candidate_k=top_k,
                rerank_top_n=rerank_top_n,
            )

        return await asyncio.to_thread(_sync_search)

