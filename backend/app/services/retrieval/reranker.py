import math
from functools import lru_cache
from typing import Protocol

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.services.retrieval.embeddings import resolve_embedding_device


class RerankerService(Protocol):
    """Interface for cross-encoder pairwise relevance scoring."""

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Return relevance scores in [0, 1] aligned with input passages."""


class CrossEncoderRerankerService:
    """GPU-accelerated HuggingFace CrossEncoder reranker (BGE-reranker)."""

    def __init__(self, model_name: str, device: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    @staticmethod
    def _sigmoid(logit: float) -> float:
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_val = math.exp(logit)
        return exp_val / (1.0 + exp_val)

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Score (query, passage) pairs and normalize logits into [0, 1]."""
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        raw_scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [round(self._sigmoid(float(score)), 6) for score in raw_scores]


@lru_cache(maxsize=2)
def _cached_reranker(model_name: str, device: str) -> CrossEncoderRerankerService:
    return CrossEncoderRerankerService(model_name=model_name, device=device)


def get_reranker_service() -> RerankerService:
    """FastAPI dependency returning the configured HuggingFace CrossEncoder reranker."""
    settings = get_settings()
    device = resolve_embedding_device(settings.embedding_device)
    return _cached_reranker(settings.hf_reranker_model, device)

