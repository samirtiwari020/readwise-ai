from functools import lru_cache
from typing import Protocol

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings


class EmbeddingService(Protocol):
    """Interface for dense passage and query embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks into normalized dense vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a user search query into a normalized dense vector."""


class HuggingFaceEmbeddingService:
    """GPU-accelerated local HuggingFace embedding wrapper (BGE-optimized)."""

    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.device = device
        self._query_prefix = (
            self._BGE_QUERY_PREFIX
            if "bge-" in model_name.lower() and "reranker" not in model_name.lower()
            else ""
        )
        self._client = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document passages in batches."""
        if not texts:
            return []
        return self._client.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query, applying retrieval prompt prefix when required."""
        formatted = f"{self._query_prefix}{query}" if self._query_prefix else query
        return self._client.embed_query(formatted)


@lru_cache(maxsize=2)
def _cached_hf_service(model_name: str, device: str) -> HuggingFaceEmbeddingService:
    return HuggingFaceEmbeddingService(model_name=model_name, device=device)


def resolve_embedding_device(preferred: str) -> str:
    """Resolve effective compute device, falling back to CPU when CUDA is unavailable."""
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embedding_service() -> EmbeddingService:
    """FastAPI dependency returning the configured HuggingFace embedding service."""
    settings = get_settings()
    device = resolve_embedding_device(settings.embedding_device)
    return _cached_hf_service(settings.hf_embedding_model, device)

