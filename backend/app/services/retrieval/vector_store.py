from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from app.core.config import get_settings


class ChromaVectorStore:
    """Persistent ChromaDB store supporting metadata-filtered multi-document retrieval."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        """Upsert embedded chunks in batches to safely handle 1,000+ page books."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def delete_document_chunks(self, document_id: str) -> None:
        """Remove all indexed vector chunks belonging to a document."""
        self._collection.delete(where={"document_id": document_id})

    def count(self) -> int:
        """Return total number of stored chunk vectors."""
        return int(self._collection.count())

    @staticmethod
    def _build_where_filter(
        document_ids: list[str] | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if document_ids:
            if len(document_ids) == 1:
                clauses.append({"document_id": document_ids[0]})
            else:
                clauses.append({"document_id": {"$in": document_ids}})
        if page_start is not None:
            clauses.append({"page_end": {"$gte": page_start}})
        if page_end is not None:
            clauses.append({"page_start": {"$lte": page_end}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        document_ids: list[str] | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query top-K vector matches with optional document and page-range filters."""
        total_vectors = self.count()
        if total_vectors == 0 or top_k <= 0:
            return []

        where_filter = self._build_where_filter(
            document_ids=document_ids,
            page_start=page_start,
            page_end=page_end,
        )
        n_results = min(top_k, total_vectors)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = (raw.get("ids") or [[]])[0]
        docs_list = (raw.get("documents") or [[]])[0]
        metas_list = (raw.get("metadatas") or [[]])[0]
        dists_list = (raw.get("distances") or [[]])[0]

        results: list[dict[str, Any]] = []
        for chroma_id, text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            similarity = max(0.0, 1.0 - float(dist))
            results.append(
                {
                    "chroma_id": chroma_id,
                    "contextual_text": text,
                    "metadata": meta or {},
                    "score": similarity,
                }
            )
        return results


@lru_cache(maxsize=4)
def _cached_vector_store(persist_dir: str, collection_name: str) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir=Path(persist_dir), collection_name=collection_name)


def get_vector_store() -> ChromaVectorStore:
    """FastAPI dependency returning the persistent ChromaDB vector store."""
    settings = get_settings()
    return _cached_vector_store(
        str(settings.resolved_chroma_dir),
        settings.chroma_collection_name,
    )

