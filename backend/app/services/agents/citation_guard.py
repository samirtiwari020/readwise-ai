import re
from typing import Any

from app.services.retrieval.hybrid_retriever import RetrievedChunk

ABSTENTION_MESSAGE = (
    "I could not find sufficient evidence in the selected documents to answer this question reliably."
)

_CITE_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _extract_quote_snippet(content: str, max_len: int = 240) -> str:
    """Return a clean concise excerpt from a cited chunk for UI hover/jump display."""
    normalized = re.sub(r"\s+", " ", content).strip()
    if len(normalized) <= max_len:
        return normalized
    clipped = normalized[: max_len - 3].rsplit(" ", 1)[0]
    return f"{clipped}..."


def verify_and_build_citations(
    raw_answer: str,
    relevant_chunks: list[RetrievedChunk],
) -> tuple[str, list[dict[str, Any]], bool]:
    """Validate inline [N] markers against retrieved chunks and renumber sequentially.

    Returns (cleaned_answer, verified_citation_dicts, abstained).
    """
    text = raw_answer.strip()
    if not text or not relevant_chunks:
        return ABSTENTION_MESSAGE, [], True

    max_idx = len(relevant_chunks)
    seen_order: dict[int, int] = {}

    def _rewrite_group(match: re.Match[str]) -> str:
        raw_nums = [int(part.strip()) for part in match.group(1).split(",")]
        valid_new_nums: list[int] = []
        for orig_idx in raw_nums:
            if 1 <= orig_idx <= max_idx:
                if orig_idx not in seen_order:
                    seen_order[orig_idx] = len(seen_order) + 1
                mapped = seen_order[orig_idx]
                if mapped not in valid_new_nums:
                    valid_new_nums.append(mapped)
        if not valid_new_nums:
            return ""
        return "".join(f"[{num}]" for num in valid_new_nums)

    cleaned = _CITE_GROUP_RE.sub(_rewrite_group, text)
    # Clean up redundant spaces left when invalid citations are stripped
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    citations: list[dict[str, Any]] = []
    for orig_idx, new_idx in seen_order.items():
        chunk = relevant_chunks[orig_idx - 1]
        citations.append(
            {
                "citation_index": new_idx,
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "chapter": chunk.chapter,
                "section": chunk.section,
                "chunk_id": chunk.chroma_id,
                "quote_snippet": _extract_quote_snippet(chunk.content),
                "score": round(chunk.score, 4),
            }
        )

    lowered = cleaned.lower()
    abstained = (
        "could not find sufficient evidence" in lowered
        or "not mentioned in the provided" in lowered
    ) and len(citations) == 0

    return cleaned, citations, abstained

