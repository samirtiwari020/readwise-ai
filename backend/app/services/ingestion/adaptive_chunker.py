import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import tiktoken

from app.services.ingestion.pdf_parser import ParsedPage, TextBlock

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count BPE tokens for adaptive chunk sizing."""
    if not text:
        return 0
    return len(_TOKENIZER.encode(text, disallowed_special=()))


@dataclass(slots=True)
class ChunkRecord:
    """Single page-aware, hierarchy-annotated chunk ready for DB & ChromaDB."""

    chunk_index: int
    content: str
    contextual_text: str
    page_start: int
    page_end: int
    chapter: str | None
    section: str | None
    chunk_type: str
    token_count: int
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class _BufferItem:
    text: str
    page_number: int
    block_type: str
    tokens: int


class AdaptiveChunker:
    """Build hierarchy- and page-aware chunks with adaptive token targets per content type."""

    def __init__(
        self,
        document_id: str,
        document_title: str,
        toc: list[dict] | None = None,
        min_tokens: int = 200,
        target_tokens: int = 600,
        max_tokens: int = 1000,
        overlap_tokens: int = 80,
    ) -> None:
        self.document_id = document_id
        self.document_title = document_title
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        # Map 1-based page number -> ordered TOC entries starting on that page
        self._toc_by_page: dict[int, list[dict]] = {}
        for entry in toc or []:
            page = int(entry.get("page", 1))
            self._toc_by_page.setdefault(page, []).append(entry)

        self._current_chapter: str | None = None
        self._current_section: str | None = None
        self._buffer: list[_BufferItem] = []
        self._buffer_tokens = 0
        self._has_new_content = False
        self._chunk_counter = 0

    def _apply_toc_for_page(self, page_number: int) -> list[ChunkRecord]:
        """Update chapter/section context from PDF TOC entries on this page."""
        emitted: list[ChunkRecord] = []
        entries = self._toc_by_page.get(page_number, [])
        for entry in entries:
            level = int(entry.get("level", 1))
            title = str(entry.get("title", "")).strip()
            if not title:
                continue
            if level <= 1 and title != self._current_chapter:
                emitted.extend(self._flush_buffer(carry_overlap=False))
                self._current_chapter = title
                self._current_section = None
            elif level >= 2 and title != self._current_section:
                emitted.extend(self._flush_buffer(carry_overlap=False))
                self._current_section = title
        return emitted

    def _split_oversized_text(self, text: str, max_chunk_tokens: int) -> list[str]:
        """Split a single block exceeding max_chunk_tokens along sentence/newline boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        parts: list[str] = []
        current_pieces: list[str] = []
        current_tokens = 0

        for piece in sentences:
            piece = piece.strip()
            if not piece:
                continue
            piece_tokens = count_tokens(piece)
            if piece_tokens > max_chunk_tokens:
                if current_pieces:
                    parts.append(" ".join(current_pieces))
                    current_pieces = []
                    current_tokens = 0
                # Hard token-window fallback for massive unbroken blocks
                raw_ids = _TOKENIZER.encode(piece, disallowed_special=())
                for start in range(0, len(raw_ids), max_chunk_tokens):
                    parts.append(_TOKENIZER.decode(raw_ids[start : start + max_chunk_tokens]))
                continue

            if current_tokens + piece_tokens > max_chunk_tokens and current_pieces:
                parts.append(" ".join(current_pieces))
                current_pieces = [piece]
                current_tokens = piece_tokens
            else:
                current_pieces.append(piece)
                current_tokens += piece_tokens

        if current_pieces:
            parts.append(" ".join(current_pieces))
        return parts

    def _target_bounds_for_type(self, block_type: str) -> tuple[int, int]:
        """Return (target_tokens, max_tokens) tuned for dense vs narrative blocks."""
        if block_type in {"table", "list", "code"}:
            dense_target = max(self.min_tokens, int(self.target_tokens * 0.55))
            dense_max = max(dense_target + 80, int(self.max_tokens * 0.55))
            return dense_target, dense_max
        return self.target_tokens, self.max_tokens

    def _dominant_chunk_type(self, items: list[_BufferItem]) -> str:
        """Pick dominant block type by token weight."""
        totals: dict[str, int] = {}
        for item in items:
            totals[item.block_type] = totals.get(item.block_type, 0) + item.tokens
        return max(totals.items(), key=lambda pair: pair[1])[0]

    def _build_contextual_text(
        self,
        content: str,
        page_start: int,
        page_end: int,
        chapter: str | None,
        section: str | None,
    ) -> str:
        """Prepend structural attribution header for embedding quality."""
        page_label = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
        crumbs = [self.document_title]
        if chapter:
            crumbs.append(chapter)
        if section:
            crumbs.append(section)
        crumbs.append(page_label)
        return f"[{' | '.join(crumbs)}]\n{content}"

    def _extract_overlap_items(self) -> list[_BufferItem]:
        """Retain tail items up to overlap_tokens while preserving true page numbers."""
        if self.overlap_tokens <= 0 or not self._buffer:
            return []
        retained: list[_BufferItem] = []
        accumulated = 0
        for item in reversed(self._buffer):
            if accumulated >= self.overlap_tokens:
                break
            if item.tokens <= self.overlap_tokens - accumulated:
                retained.append(item)
                accumulated += item.tokens
            else:
                # Trim tail sentence/words cleanly
                words = item.text.split()
                tail_words = words[-max(15, self.overlap_tokens // 2) :]
                tail_text = " ".join(tail_words)
                retained.append(
                    _BufferItem(
                        text=tail_text,
                        page_number=item.page_number,
                        block_type=item.block_type,
                        tokens=count_tokens(tail_text),
                    )
                )
                break
        retained.reverse()
        return retained

    def _flush_buffer(self, carry_overlap: bool = True) -> list[ChunkRecord]:
        """Materialize the current buffer into a ChunkRecord."""
        if not self._buffer or not self._has_new_content:
            if not carry_overlap:
                self._buffer = []
                self._buffer_tokens = 0
                self._has_new_content = False
            return []

        items = self._buffer
        content = "\n\n".join(item.text for item in items).strip()
        if not content:
            self._buffer = []
            self._buffer_tokens = 0
            self._has_new_content = False
            return []

        pages = [item.page_number for item in items]
        page_start = min(pages)
        page_end = max(pages)
        chunk_type = self._dominant_chunk_type(items)
        token_count = count_tokens(content)

        record = ChunkRecord(
            chunk_index=self._chunk_counter,
            content=content,
            contextual_text=self._build_contextual_text(
                content=content,
                page_start=page_start,
                page_end=page_end,
                chapter=self._current_chapter,
                section=self._current_section,
            ),
            page_start=page_start,
            page_end=page_end,
            chapter=self._current_chapter,
            section=self._current_section,
            chunk_type=chunk_type,
            token_count=token_count,
            metadata={
                "document_id": self.document_id,
                "document_title": self.document_title,
                "chunk_index": self._chunk_counter,
                "page_start": page_start,
                "page_end": page_end,
                "chapter": self._current_chapter or "",
                "section": self._current_section or "",
                "chunk_type": chunk_type,
                "token_count": token_count,
            },
        )
        self._chunk_counter += 1

        if carry_overlap:
            overlap = self._extract_overlap_items()
            self._buffer = overlap
            self._buffer_tokens = sum(item.tokens for item in overlap)
            self._has_new_content = False
        else:
            self._buffer = []
            self._buffer_tokens = 0
            self._has_new_content = False

        return [record]

    def _append_to_buffer(self, item: _BufferItem) -> None:
        """Append a newly parsed block item to the buffer and mark fresh content."""
        self._buffer.append(item)
        self._buffer_tokens += item.tokens
        self._has_new_content = True

    def _process_block(self, block: TextBlock) -> list[ChunkRecord]:
        """Consume a single classified TextBlock and emit zero or more chunks."""
        emitted: list[ChunkRecord] = []

        if block.block_type == "heading":
            # Flush before starting a new chapter/section boundary
            emitted.extend(self._flush_buffer(carry_overlap=False))
            heading_text = re.sub(r"\s+", " ", block.text).strip()
            if block.heading_level == 1:
                self._current_chapter = heading_text
                self._current_section = None
            else:
                self._current_section = heading_text
            # Keep heading text prefixed in the next buffer so section title is searchable
            self._append_to_buffer(
                _BufferItem(
                    text=heading_text,
                    page_number=block.page_number,
                    block_type="heading",
                    tokens=count_tokens(heading_text),
                )
            )
            return emitted

        target_limit, max_limit = self._target_bounds_for_type(block.block_type)
        block_tokens = count_tokens(block.text)

        # Split oversized blocks before buffering
        if block_tokens > max_limit:
            emitted.extend(self._flush_buffer(carry_overlap=True))
            for sub_text in self._split_oversized_text(block.text, target_limit):
                sub_tokens = count_tokens(sub_text)
                self._append_to_buffer(
                    _BufferItem(
                        text=sub_text,
                        page_number=block.page_number,
                        block_type=block.block_type,
                        tokens=sub_tokens,
                    )
                )
                if self._buffer_tokens >= target_limit:
                    emitted.extend(self._flush_buffer(carry_overlap=True))
            return emitted

        # Flush before adding if appending would cross adaptive max limit
        if self._buffer and self._has_new_content and (self._buffer_tokens + block_tokens > max_limit):
            emitted.extend(self._flush_buffer(carry_overlap=True))

        self._append_to_buffer(
            _BufferItem(
                text=block.text,
                page_number=block.page_number,
                block_type=block.block_type,
                tokens=block_tokens,
            )
        )

        if self._buffer_tokens >= target_limit:
            emitted.extend(self._flush_buffer(carry_overlap=True))

        return emitted

    def chunk_pages(self, pages: Iterable[ParsedPage]) -> list[ChunkRecord]:
        """Chunk an entire stream of parsed pages into adaptive ChunkRecords."""
        chunks: list[ChunkRecord] = []
        for page in pages:
            chunks.extend(self._apply_toc_for_page(page.page_number))
            for block in page.blocks:
                chunks.extend(self._process_block(block))
        chunks.extend(self._flush_buffer(carry_overlap=False))
        return chunks
